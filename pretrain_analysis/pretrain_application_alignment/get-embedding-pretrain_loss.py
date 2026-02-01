
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
# Argument Parsing (compatible with original)
# ============================================================

input_seq = sys.argv[1]
output_dir = sys.argv[2]
embedding_len = int(sys.argv[3])
layer = int(sys.argv[4])
epoch = int(sys.argv[5])

# If True: mean over token dimension, output shape (N, d_model)
mean_mode = False

name = os.path.basename(input_seq).split(".")[0]
output_file = os.path.join(output_dir, f"{name}-embedding-layer_{layer}.npy")

# Skip if already produced
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    sys.exit(0)

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Device & Tunables
# ============================================================

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# -------- performance / memory knobs (env override) --------
# Batch size: keep small to avoid OOM. Increase if you have headroom.
batch_size = int(os.environ.get("BATCH_SIZE", "1"))

# Dataloader workers: >0 may speed up, but also increases RAM usage.
num_workers = int(os.environ.get("NUM_WORKERS", "0"))

# Autocast: greatly reduces GPU memory for large models; usually safe for inference.
use_autocast = os.environ.get("USE_AUTOCAST", "1") == "1"
autocast_dtype = os.environ.get("AUTOCAST_DTYPE", "bf16").lower()  # bf16 | fp16
if autocast_dtype not in ("bf16", "fp16"):
    autocast_dtype = "bf16"

# Output dtype on disk: float16 is faster/smaller; float32 is safer for downstream.
save_dtype = os.environ.get("SAVE_DTYPE", "float32").lower()  # float16 | float32
if save_dtype not in ("float16", "float32"):
    save_dtype = "float32"
save_np_dtype = np.float16 if save_dtype == "float16" else np.float32

# Optional: skip counting by setting TOTAL_SEQ
total_seq_env = os.environ.get("TOTAL_SEQ")

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


def count_sequences_fast(path: str) -> int:
    """Count lines in a text file without loading it into memory."""
    if total_seq_env is not None:
        try:
            n = int(total_seq_env)
            if n > 0:
                return n
        except Exception:
            pass

    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def get_autocast_dtype():
    if autocast_dtype == "fp16":
        return torch.float16
    return torch.bfloat16


# ============================================================
# Model / Tokenizer Setup
# ============================================================

sys.path.append('../../pretrain_gLM')
from BERT.data import DNADataset, DNATokenizer
from BERT.model import BaselineBERT
from BERT.utils import load_config

config = load_config("../../pretrain_gLM/config/Baseline_gLM.config.json")
model_dir = "../../pretrain_analysis/pretrain_application_alignment/BERT-155M-Series/model_M"
model_version = find_files_with_prefix(model_dir, f"model_{epoch}_")[0]
model_checkpoint = os.path.join(model_dir, model_version)

model_config = config["model_config"]
d_kv = model_config["d_kv"]
n_heads = model_config["n_heads"]
n_layers = model_config["n_layers"]
max_vocab = model_config["max_vocab"]
d_model = n_heads * d_kv
kmer = model_config["kmer"]

# ============================================================
# Model Loader
# ============================================================

def load_model():
    print("Initializing model and loading checkpoint...")

    model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers, eval_mode=True)
    model.to(device)

    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # remove unused keys
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    model_state.update(state_dict)
    model.load_state_dict(model_state)

    model.eval()
    return model


# ============================================================
# Dataset & DataLoader
# ============================================================

def build_dataloader(seq_path):
    """
    Build DNADataset and DataLoader, without np.loadtxt (which loads everything into RAM).
    """
    seq_num = count_sequences_fast(seq_path)
    print(f"[INFO] seq_num={seq_num} (batch_size={batch_size}, num_workers={num_workers})")

    tokenizer = DNATokenizer(kmer=kmer)
    dataset = DNADataset(tokenizer, seq_path, data_type="seq", eval_mode=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,   # pin_memory=True can speed up but increases RAM pressure
        drop_last=False,
    )
    return dataloader, seq_num


# ============================================================
# Output Buffer Allocation (memmap .npy)
# ============================================================

def allocate_output_memmap(seq_num):
    """
    Allocate output as a .npy memmap so we don't keep a huge array in RAM.
    """
    if mean_mode:
        shape = (seq_num, d_model)
    else:
        shape = (seq_num, embedding_len, d_model)

    # Create .npy with header + memmap view
    mm = np.lib.format.open_memmap(
        output_file, mode="w+", dtype=save_np_dtype, shape=shape
    )
    return mm


# ============================================================
# Embedding Extraction (streaming write)
# ============================================================

def extract_embeddings(model, dataloader, out_mm):
    """
    Stream embeddings to disk via memmap.
    Keeps peak RAM low; GPU memory mainly depends on model+batch+seq_len.
    """
    print("Extracting embedding...")
    model.to(device)
    model.eval()

    # Choose autocast context
    if device.type == "cuda" and use_autocast:
        ac = torch.autocast(device_type="cuda", dtype=get_autocast_dtype())
    else:
        # no-op context manager
        from contextlib import nullcontext
        ac = nullcontext()

    write_pos = 0

    with torch.inference_mode():
        with ac:
            for idx, batch in enumerate(dataloader):
                # DNADataset returns tuple; original used batch[0]
                input_ids = batch[0].to(device, non_blocking=False)

                # BaselineBERT returns per-layer embeddings (list/tuple)
                embedding = model(input_ids)

                # Slice: remove [CLS] and keep first embedding_len tokens
                if mean_mode:
                    emb = embedding[layer][:, 1 : 1 + embedding_len, :]
                    emb = torch.mean(emb, dim=1)  # (B, d_model)
                    emb_np = emb.detach().cpu().to(torch.float32).numpy()
                    bsz = emb_np.shape[0]
                    out_mm[write_pos : write_pos + bsz] = emb_np.astype(save_np_dtype, copy=False)
                else:
                    emb = embedding[layer][:, 1 : 1 + embedding_len, :]  # (B, L, d_model)
                    emb_np = emb.detach().cpu().to(torch.float32).numpy()
                    bsz = emb_np.shape[0]
                    out_mm[write_pos : write_pos + bsz] = emb_np.astype(save_np_dtype, copy=False)

                write_pos += bsz

                # Free references ASAP (helps reduce peak memory)
                del embedding, emb, emb_np, input_ids

                if device.type == "cuda" and (idx + 1) % 50 == 0:
                    # Occasionally clear cache to reduce fragmentation (don't do every step)
                    torch.cuda.empty_cache()

    if write_pos != out_mm.shape[0]:
        print(f"[WARN] wrote {write_pos} samples, expected {out_mm.shape[0]}")

    out_mm.flush()


# ============================================================
# Main
# ============================================================

def main():
    model = load_model()
    print(model)

    dataloader, seq_num = build_dataloader(input_seq)
    out_mm = allocate_output_memmap(seq_num)

    extract_embeddings(model, dataloader, out_mm)

    # memmap already saved as output_file; flush ensures data is written.
    print(f"[INFO] Saved embeddings to: {output_file} (dtype={save_dtype})")


if __name__ == "__main__":
    main()
