
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader


def _parse_args(argv):
    input_seq = argv[1]
    output_dir = argv[2]
    embedding_len = int(argv[3])
    layer = int(argv[4])
    epoch = int(argv[5])

    # default format from env
    save_format = "npy"

    # parse extra args
    extra = argv[6:]
    for i, tok in enumerate(extra):
        if tok in ("--save_format", "--format"):
            if i + 1 >= len(extra):
                raise SystemExit(f"Error: {tok} requires a value (npy or h5).")
            save_format = extra[i + 1].strip().lower()

    if save_format not in ("npy", "h5"):
        raise SystemExit("Error: --save_format/--format must be one of: npy, h5")

    return input_seq, output_dir, embedding_len, layer, epoch, save_format


input_seq, output_dir, embedding_len, layer, epoch, save_format = _parse_args(sys.argv)

# If True: mean over token dimension, output shape (N, d_model)
mean_mode = False

name = os.path.basename(input_seq).split(".")[0]
ext = ".npy" if save_format == "npy" else ".h5"
output_file = os.path.join(output_dir, f"{name}-embedding-layer_{layer}{ext}")

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
# Output Allocation (memmap .npy OR streaming .h5)
# ============================================================

def get_output_shape(seq_num: int):
    if mean_mode:
        return (seq_num, d_model)
    return (seq_num, embedding_len, d_model)


def allocate_output_npy(seq_num: int):
    """
    Allocate output as a .npy memmap so we don't keep a huge array in RAM.
    """
    shape = get_output_shape(seq_num)
    mm = np.lib.format.open_memmap(
        output_file, mode="w+", dtype=save_np_dtype, shape=shape
    )
    return mm


def allocate_output_h5(seq_num: int):
    """
    Allocate output as an HDF5 dataset and stream-write slices.
    """
    try:
        import h5py
    except Exception as e:
        raise RuntimeError(
            "Saving to .h5 requires 'h5py'. Install it via: pip install h5py"
        ) from e

    shape = get_output_shape(seq_num)

    # A small, safe default chunk size along the first dimension.
    chunk0 = max(1, min(batch_size, seq_num))
    if mean_mode:
        chunks = (chunk0, d_model)
    else:
        chunks = (chunk0, embedding_len, d_model)

    f = h5py.File(output_file, "w")
    dset = f.create_dataset(
        "embeddings",
        shape=shape,
        dtype=save_np_dtype,
        chunks=chunks,
    )

    # Basic metadata (kept minimal)
    f.attrs["input_seq"] = str(input_seq)
    f.attrs["layer"] = int(layer)
    f.attrs["epoch"] = int(epoch)
    f.attrs["embedding_len"] = int(embedding_len)
    f.attrs["mean_mode"] = bool(mean_mode)
    f.attrs["save_dtype"] = str(save_dtype)

    return f, dset


# ============================================================
# Embedding Extraction (streaming write)
# ============================================================

def extract_embeddings(model, dataloader, seq_num, out_npy_mm=None, out_h5_f=None, out_h5_ds=None):
    """
    Stream embeddings to disk. Keeps peak RAM low; GPU memory mainly depends on model+batch+seq_len.
    """
    print("Extracting embedding...")
    model.to(device)
    model.eval()

    # Choose autocast context
    if device.type == "cuda" and use_autocast:
        ac = torch.autocast(device_type="cuda", dtype=get_autocast_dtype())
    else:
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
                else:
                    emb = embedding[layer][:, 1 : 1 + embedding_len, :]  # (B, L, d_model)
                    emb_np = emb.detach().cpu().to(torch.float32).numpy()

                bsz = emb_np.shape[0]

                # Write slice
                if save_format == "npy":
                    out_npy_mm[write_pos : write_pos + bsz] = emb_np.astype(save_np_dtype, copy=False)
                else:
                    out_h5_ds[write_pos : write_pos + bsz] = emb_np.astype(save_np_dtype, copy=False)

                write_pos += bsz

                # Free references ASAP (helps reduce peak memory)
                del embedding, emb, emb_np, input_ids

                if device.type == "cuda" and (idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()

    if write_pos != seq_num:
        print(f"[WARN] wrote {write_pos} samples, expected {seq_num}")

    # Flush
    if save_format == "npy":
        out_npy_mm.flush()
    else:
        out_h5_f.flush()


# ============================================================
# Main
# ============================================================

def main():
    model = load_model()
    print(model)

    dataloader, seq_num = build_dataloader(input_seq)

    if save_format == "npy":
        out_mm = allocate_output_npy(seq_num)
        extract_embeddings(model, dataloader, seq_num, out_npy_mm=out_mm)
        print(f"[INFO] Saved embeddings to: {output_file} (dtype={save_dtype}, format=npy)")
    else:
        out_f, out_ds = allocate_output_h5(seq_num)
        try:
            extract_embeddings(model, dataloader, seq_num, out_h5_f=out_f, out_h5_ds=out_ds)
            print(f"[INFO] Saved embeddings to: {output_file} (dtype={save_dtype}, format=h5, dataset='embeddings')")
        finally:
            out_f.close()


if __name__ == "__main__":
    main()
