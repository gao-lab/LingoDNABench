import os
import sys
import time
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# =========================
# Argument Parsing
# =========================

input_seq = sys.argv[1]
output_dir = sys.argv[2]
embedding_len = int(sys.argv[3])
layer = int(sys.argv[4])
epoch = int(sys.argv[5])

mean_mode = False

name = os.path.basename(input_seq).split(".")[0]
output_file = f"{output_dir}/{name}-embedding-layer_{layer}.npy"

if os.path.exists(output_file) and os.path.getsize(output_file) != 0:
    sys.exit(0)


# =========================
# Environment
# =========================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# =========================
# Utilities
# =========================

def find_files_with_prefix(directory, prefix):
    if not os.path.exists(directory):
        raise ValueError(f"dir {directory} not exists")
    return [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f))
    ]


# =========================
# Model & Tokenizer Config
# =========================

batch_size = 4
kmer = 1

sys.path.append("./BERT-155M-Series")
from dpb_bert.data import DNADataset, DNATokenizer
from dpb_bert.model import DNALingo

model_dir = "./BERT-155M-Series/MS7-4K-8-K1"
model_version = find_files_with_prefix(model_dir, f"model_{epoch}_")[0]
model_checkpoint = f"{model_dir}/{model_version}"

# model hyper-params (unchanged)
d_kv = 64
n_heads = 16
n_layers = 12
max_vocab = 16
d_model = n_heads * d_kv


# =========================
# Model Loading
# =========================

def load_model():
    print("Initializing model and loading checkpoint...")

    model = DNALingo(
        max_vocab=max_vocab,
        n_heads=n_heads,
        d_kv=d_kv,
        n_layers=n_layers,
        eval_mode=True,
    ).to(device)

    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    model_state.update(state_dict)
    model.load_state_dict(model_state)

    return model


model = load_model()
print(model)


# =========================
# Dataset & Dataloader
# =========================

seq_path = input_seq
input_seq_arr = np.loadtxt(seq_path, dtype=str, delimiter="\t")
seq_num = input_seq_arr.shape[0]
print(seq_num)

dna_tokenizer = DNATokenizer(kmer=kmer)
seq_dataset = DNADataset(dna_tokenizer, seq_path, data_type="seq")

seq_dataloader = DataLoader(
    seq_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)


# =========================
# Output Buffer
# =========================

if mean_mode:
    output_embedding = np.empty((seq_num, d_model), dtype=np.float32)
else:
    output_embedding = np.empty(
        (seq_num, embedding_len, d_model), dtype=np.float32
    )


# =========================
# Embedding Extraction
# =========================

print("Extracting embedding...")
model.eval().cuda()

# flash attention only supports fp16 / bf16
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        for idx, batch in enumerate(seq_dataloader):
            input_ids = batch[0].cuda()
            embedding = model(input_ids)

            start = idx * batch_size
            end = start + embedding[layer].shape[0]

            if mean_mode:
                # exclude CLS & SEP
                emb = embedding[layer][:, 1:1 + embedding_len, :]
                emb = torch.mean(emb, dim=1)
                output_embedding[start:end] = (
                    emb.cpu().numpy().astype(np.float32)
                )
            else:
                output_embedding[start:end] = (
                    embedding[layer][:, 1:1 + embedding_len, :]
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )


# =========================
# Save
# =========================

np.save(output_file, output_embedding.astype(np.float32))
