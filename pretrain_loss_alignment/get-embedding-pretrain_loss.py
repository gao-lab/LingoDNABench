
# ============================================================
# Imports & Environment
# ============================================================

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# ============================================================
# Argument Parsing (unchanged)
# ============================================================

input_seq = sys.argv[1]
output_dir = sys.argv[2]
embedding_len = int(sys.argv[3])
layer = int(sys.argv[4])
epoch = int(sys.argv[5])

mean_mode = False

name = os.path.basename(input_seq).split(".")[0]
output_file = os.path.join(
    output_dir, f"{name}-embedding-layer_{layer}.npy"
)

if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    sys.exit(0)

# ============================================================
# Device & Global Config
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

batch_size = 4
kmer = 1

# ============================================================
# Utility Functions
# ============================================================

def find_files_with_prefix(directory, prefix):
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    return [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f))
    ]

# ============================================================
# Model / Tokenizer Setup
# ============================================================

sys.path.append("./BERT-155M-Series")
from dpb_bert.data import DNADataset, DNATokenizer
from dpb_bert.model import DNALingo

model_dir = "./BERT-155M-Series/MS7-4K-8-K1"
model_version = find_files_with_prefix(
    model_dir, f"model_{epoch}_"
)[0]
model_checkpoint = os.path.join(model_dir, model_version)

# model hyper-parameters (unchanged)
d_kv = 64
n_heads = 16
n_layers = 12
max_vocab = 16
d_model = n_heads * d_kv

# ============================================================
# Model Loader
# ============================================================

def load_model(checkpoint_path):
    """
    Initialize DNALingo model and load checkpoint.
    Logic identical to original script.
    """
    print("Initializing model and loading checkpoint...")

    model = DNALingo(
        max_vocab=max_vocab,
        n_heads=n_heads,
        d_kv=d_kv,
        n_layers=n_layers,
        eval_mode=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    model_state = model.state_dict()
    filtered_state = {
        k: v for k, v in state_dict.items() if k in model_state
    }
    model_state.update(filtered_state)
    model.load_state_dict(model_state)

    return model

# ============================================================
# Dataset & DataLoader
# ============================================================

def build_dataloader(seq_path):
    """
    Build DNADataset and DataLoader.
    """
    seq_arr = np.loadtxt(seq_path, dtype=str, delimiter="\t")
    seq_num = seq_arr.shape[0]
    print(seq_num)

    tokenizer = DNATokenizer(kmer=kmer)
    dataset = DNADataset(tokenizer, seq_path, data_type="seq")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    return dataloader, seq_num

# ============================================================
# Output Buffer Allocation
# ============================================================

def allocate_output_buffer(seq_num):
    """
    Allocate output embedding buffer.
    """
    if mean_mode:
        return np.empty((seq_num, d_model), dtype=np.float32)
    else:
        return np.empty(
            (seq_num, embedding_len, d_model),
            dtype=np.float32,
        )

# ============================================================
# Embedding Extraction
# ============================================================

def extract_embeddings(model, dataloader, output_buffer):
    """
    Extract embeddings from pretrained model.
    Logic strictly matches original implementation.
    """
    print("Extracting embedding...")
    model.eval().cuda()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                input_ids = batch[0].cuda()
                embedding = model(input_ids)

                start = idx * batch_size
                end = start + embedding[layer].shape[0]

                if mean_mode:
                    emb = embedding[layer][:, 1 : 1 + embedding_len, :]
                    emb = torch.mean(emb, dim=1)
                    output_buffer[start:end] = (
                        emb.cpu().numpy().astype(np.float32)
                    )
                else:
                    output_buffer[start:end] = (
                        embedding[layer][:, 1 : 1 + embedding_len, :]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

# ============================================================
# Main Pipeline
# ============================================================

def main():
    model = load_model(model_checkpoint)
    print(model)

    dataloader, seq_num = build_dataloader(input_seq)
    output_embedding = allocate_output_buffer(seq_num)

    extract_embeddings(model, dataloader, output_embedding)

    np.save(output_file, output_embedding.astype(np.float32))

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
