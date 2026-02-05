
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

# -------------------------------
# Device & runtime knobs
# -------------------------------
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here; let launcher decide.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Control memory via env vars
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))          # default 1 to avoid OOM
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))        # default 0 (multi-worker can increase RAM)
PIN_MEMORY = bool(int(os.getenv("PIN_MEMORY", "1"))) if device.type == "cuda" else False

# Mixed precision for CUDA inference
USE_AUTOCAST = bool(int(os.getenv("USE_AUTOCAST", "1"))) and device.type == "cuda"
AUTOCAST_DTYPE = os.getenv("AUTOCAST_DTYPE", "bf16").lower()  # bf16 or fp16
if AUTOCAST_DTYPE in ("bf16", "bfloat16"):
    _autocast_dtype = torch.bfloat16
else:
    _autocast_dtype = torch.float16

# Optional: avoid fragmentation (helpful on some clusters)
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# -------------------------------
# CLI arguments
# -------------------------------
parser = argparse.ArgumentParser(
    description="Stream embeddings to disk (.npy memmap by default; optional .h5)."
)
parser.add_argument("model_dir")
parser.add_argument("model_type")
parser.add_argument("model_name")
parser.add_argument("input_seq")
parser.add_argument("output_dir")
parser.add_argument("embedding_len", type=int, help="token length expected by benchmark (fixed output length)")
parser.add_argument("layer", type=int)

parser.add_argument(
    "--output_file",
    default=None,
    help="optional explicit output file path (overrides default naming).",
)
parser.add_argument(
    "--save_format",
    "--format",
    dest="save_format",
    choices=["npy", "h5"],
    default="npy",
    help="output format: npy (default) or h5.",
)
args = parser.parse_args()

model_dir = args.model_dir
model_type = args.model_type
model_name = args.model_name
input_seq = args.input_seq
output_dir = args.output_dir
embedding_len = int(args.embedding_len)   # token length expected by benchmark (fixed output length)
layer = int(args.layer)                  # same semantics as old code: hidden_states[layer]
SAVE_FORMAT = args.save_format

name = os.path.basename(input_seq).split(".")[0]
default_ext = ".npy" if SAVE_FORMAT == "npy" else ".h5"
output_file = args.output_file or os.path.join(output_dir, f"{name}-embedding-layer_{layer}{default_ext}")

os.makedirs(output_dir, exist_ok=True)

# If already done, exit quickly
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    sys.exit(0)

# -------------------------------
# IO helpers: count & iterate sequences without loading all
# -------------------------------
def _normalize_line_to_seq(line: str):
    """Best-effort extraction of sequence from a text/tsv/csv line."""
    line = line.strip()
    if not line:
        return None
    # skip common headers
    low = line.lower()
    if low.startswith("sequence") or low.startswith("seq") and "," in line:
        # e.g. deepgene csv header "sequence,..."
        return None
    if line.startswith(">"):  # fasta header
        return None

    # prefer first column before tab/comma
    if "\t" in line:
        seq = line.split("\t", 1)[0].strip()
    elif "," in line:
        seq = line.split(",", 1)[0].strip()
    else:
        seq = line

    if not seq:
        return None
    return seq


def count_sequences(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            seq = _normalize_line_to_seq(line)
            if seq is None:
                continue
            n += 1
    return n


def iter_sequences(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            seq = _normalize_line_to_seq(line)
            if seq is None:
                continue
            yield seq


# -------------------------------
# Optional: capture a single layer via forward hook to avoid output_hidden_states=True
# Works best for BERT-like encoders. Falls back to hidden_states when not supported.
# -------------------------------
def _get_encoder_layers(model):
    # try common attribute paths (HF BERT/Roberta/Deberta/etc.)
    candidates = [
        ("encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("roberta", "encoder", "layer"),
        ("deberta", "encoder", "layer"),
        ("backbone", "encoder", "layer"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (list, torch.nn.ModuleList)):
            return obj
    return None


def make_extract_fn(model, kind: str):
    """
    Return a function extract(batch)->Tensor[B, embedding_len, hidden]
    kind: 'nt', 'caduceus', 'hyenadna', 'omnina', ...
    """
    use_hook = bool(int(os.getenv("CAPTURE_LAYER_HOOK", "1")))
    layers = _get_encoder_layers(model) if use_hook else None

    # The old code used hidden_states[layer]. In HF, hidden_states[0] is embeddings,
    # hidden_states[1] is output of first transformer block, ... hidden_states[-1] last block.
    # Hook can only capture transformer blocks, not embedding output; for layer==0 we fallback.
    wants_embedding_output = (layer == 0)

    if layers is None or wants_embedding_output:
        # Fallback: request all hidden_states (higher memory)
        def _extract_hidden_states(outputs):
            hs = outputs.hidden_states
            if kind == "nt":
                return hs[layer][:, 1:1 + embedding_len, :]
            elif kind in ("caduceus", "hyenadna"):
                return hs[layer][:, :-1, :]
            elif kind == "omnina":
                # already tokenized to embedding_len
                return hs[layer][:, :embedding_len, :]
            else:
                return hs[layer][:, :embedding_len, :]

        def extract(batch):
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            out = _extract_hidden_states(outputs)
            return out

        return extract

    # Hook path: capture a single transformer block output
    # Map hidden_states index -> encoder layer index
    if layer > 0:
        enc_idx = layer - 1
    else:
        enc_idx = layer  # negative indexing works for ModuleList
    # Normalize negative index
    if enc_idx < 0:
        enc_idx = len(layers) + enc_idx
    enc_idx = max(0, min(enc_idx, len(layers) - 1))

    captured = {"tensor": None}

    def _hook(_module, _inp, out):
        # out may be tuple or tensor
        t = out[0] if isinstance(out, (tuple, list)) else out
        captured["tensor"] = t

    handle = layers[enc_idx].register_forward_hook(_hook)

    def extract(batch):
        captured["tensor"] = None
        _ = model(**batch, output_hidden_states=False, return_dict=True)
        t = captured["tensor"]
        if t is None:
            # safety fallback
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            t = outputs.hidden_states[layer]
        if kind == "nt":
            t = t[:, 1:1 + embedding_len, :]
        elif kind in ("caduceus", "hyenadna"):
            t = t[:, :-1, :]
        elif kind == "omnina":
            t = t[:, :embedding_len, :]
        else:
            t = t[:, :embedding_len, :]
        return t

    # ensure hook removed at process exit
    import atexit
    atexit.register(lambda: handle.remove())
    return extract


# -------------------------------
# Iterable datasets (stream tokenization)
# -------------------------------
class TextSeqIterable(IterableDataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __iter__(self):
        yield from iter_sequences(self.path)


class TokenizeIterable(IterableDataset):
    """
    Stream sequences -> tokenized batch items (dict of tensors).
    tokenizer: HF tokenizer-like
    max_length: int
    """
    def __init__(self, seq_path: str, tokenizer, max_length: int, add_pad_token: bool = False):
        super().__init__()
        self.seq_path = seq_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_pad_token = add_pad_token
        if add_pad_token:
            # Some tokenizers (OmniNA) may need explicit pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def __iter__(self):
        for seq in iter_sequences(self.seq_path):
            out = self.tokenizer(
                seq,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            # squeeze batch dim
            yield {k: v.squeeze(0) for k, v in out.items()}


def _collate_dict(batch_list):
    # batch_list is list of dict[tensor]
    keys = batch_list[0].keys()
    return {k: torch.stack([b[k] for b in batch_list], dim=0) for k in keys}


# -------------------------------
# Core: stream embeddings to disk (default .npy memmap; optional .h5)
# -------------------------------
from numpy.lib.format import open_memmap


class OutputWriter:
    """Write embeddings incrementally to .npy (memmap) or .h5."""

    def __init__(
        self,
        path: str,
        fmt: str,
        total: int,
        emb_len: int,
        hidden_size: int,
        *,
        chunk_rows: int = 1,
    ):
        self.path = path
        self.fmt = fmt

        if fmt == "npy":
            self.mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(total, emb_len, hidden_size))
            self.mm[:] = 0.0  # initialize (also ensures deterministic padding)
            self._total = total
            self._emb_len = emb_len
            self._hidden_size = hidden_size
            self.h5 = None
            self.dset = None
            return

        if fmt == "h5":
            try:
                import h5py
            except Exception as e:
                raise RuntimeError(
                    "Saving to .h5 requires the 'h5py' package. Install with: pip install h5py"
                ) from e

            compression = os.getenv("H5_COMPRESSION", "gzip").lower()
            if compression in ("none", "null", "false", "0", ""):
                compression = None
                compression_opts = None
                shuffle = False
            else:
                compression_opts = int(os.getenv("H5_COMPRESSION_LEVEL", "4"))
                shuffle = True

            chunk0 = min(max(1, int(chunk_rows)), total)

            self.h5 = h5py.File(path, "w")
            self.dset = self.h5.create_dataset(
                "embeddings",
                shape=(total, emb_len, hidden_size),
                dtype="float32",
                chunks=(chunk0, emb_len, hidden_size),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
                fillvalue=0.0,
            )

            # Lightweight metadata (optional, but helpful)
            self.h5.attrs["embedding_len"] = emb_len
            self.h5.attrs["hidden_size"] = hidden_size
            self.h5.attrs["layer"] = layer
            self.h5.attrs["model_type"] = model_type
            self.h5.attrs["model_name"] = model_name
            self.h5.attrs["input_seq"] = input_seq

            self.mm = None
            self._total = total
            self._emb_len = emb_len
            self._hidden_size = hidden_size
            return

        raise ValueError(f"Unsupported save format: {fmt}")

    def write(self, start: int, batch_arr: np.ndarray):
        """batch_arr: float32 numpy array with shape (B, emb_len, H)."""
        if batch_arr.dtype != np.float32:
            batch_arr = batch_arr.astype(np.float32, copy=False)
        bsz = batch_arr.shape[0]
        if self.fmt == "npy":
            self.mm[start: start + bsz, :, :] = batch_arr
        else:
            self.dset[start: start + bsz, :, :] = batch_arr

    def flush(self):
        if self.fmt == "npy":
            self.mm.flush()
        else:
            self.h5.flush()

    def close(self):
        self.flush()
        if self.fmt == "h5":
            self.h5.close()


def open_output_writer(total: int, hidden_size: int) -> OutputWriter:
    return OutputWriter(output_file, SAVE_FORMAT, total, embedding_len, hidden_size, chunk_rows=BATCH_SIZE)


def _pad_to_fixed(arr_2d: np.ndarray, hidden_size: int) -> np.ndarray:
    """Pad a [L, H] array to [1, embedding_len, H] with zeros."""
    out = np.zeros((1, embedding_len, hidden_size), dtype=np.float32)
    if arr_2d is None:
        return out
    L = min(arr_2d.shape[0], embedding_len)
    out[0, :L, :] = arr_2d[:L, :].astype(np.float32, copy=False)
    return out


def stream_to_memmap(total: int, hidden_size: int, dataloader, extract_fn):
    """
    Write embeddings directly to output_file.

    - SAVE_FORMAT=npy (default): writes a .npy memmap (fastest).
    - SAVE_FORMAT=h5: writes an HDF5 file with dataset key 'embeddings'.
    """
    writer = open_output_writer(total, hidden_size)
    idx = 0

    # Use inference_mode for best memory behavior
    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                    emb = extract_fn(batch)
            else:
                emb = extract_fn(batch)

            emb = emb.detach().float().cpu().numpy()  # float32 on disk
            bsz = emb.shape[0]

            # emb is expected to be [B, embedding_len, H]
            writer.write(idx, emb[:, :embedding_len, :])

            idx += bsz

            # Help free GPU memory quickly
            del emb, batch

    writer.close()
    return output_file


# -------------------------------
# Model runners (memory-optimized)
# -------------------------------
def run_caduceus():
    # Caduceus uses custom tokenizer in your repo
    sys.path.append(f"{model_dir}/{model_name}")
    from tokenization_caduceus import CaduceusTokenizer

    tokenizer = CaduceusTokenizer.from_pretrained(
        f"{model_dir}/{model_name}",
        padding="max_length",
        max_length=embedding_len + 1,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    ds = TokenizeIterable(input_seq, tokenizer, max_length=embedding_len + 1)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=_collate_dict,
    )

    extract_fn = make_extract_fn(model, kind="caduceus")
    return stream_to_memmap(total, hidden_size, dl, extract_fn)


def run_hyenadna():
    sys.path.append(f"{model_dir}/{model_name}")
    from tokenization_hyena import HyenaDNATokenizer

    tokenizer = HyenaDNATokenizer.from_pretrained(
        f"{model_dir}/{model_name}",
        padding="max_length",
        max_length=embedding_len + 1,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    ds = TokenizeIterable(input_seq, tokenizer, max_length=embedding_len + 1)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=_collate_dict,
    )

    extract_fn = make_extract_fn(model, kind="hyenadna")
    return stream_to_memmap(total, hidden_size, dl, extract_fn)


def run_nt():
    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    ds = TokenizeIterable(input_seq, tokenizer, max_length=embedding_len + 1)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=_collate_dict,
    )

    extract_fn = make_extract_fn(model, kind="nt")
    return stream_to_memmap(total, hidden_size, dl, extract_fn)


def run_omnina():
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}")
    model = AutoModel.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    ds = TokenizeIterable(input_seq, tokenizer, max_length=embedding_len, add_pad_token=True)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=_collate_dict,
    )

    extract_fn = make_extract_fn(model, kind="omnina")
    return stream_to_memmap(total, hidden_size, dl, extract_fn)


def run_bert_series():
    # Your internal BERT-series code path (keep logic, but stream & memmap)
    sys.path.append("../pretrain_gLM")
    from BERT.data import DNADataset, DNATokenizer
    from BERT.model import BaselineBERT
    from BERT.utils import load_config

    config = load_config("../pretrain_gLM/config/Baseline_gLM.config.json")
    if model_name == "RandomWeight":
        model_checkpoint = "../pretrain_analysis/pretrain_application_alignment/BERT-155M-Series/RandomWeights/model_init_666.pt"
    elif model_name == "model_M":
        model_checkpoint = "../pretrain_analysis/pretrain_application_alignment/BERT-155M-Series/model_M/model_8_711981.pt"
    elif model_name == "model_H":
        model_checkpoint = "../pretrain_analysis/pretrain_application_alignment/BERT-155M-Series/model_H/model_13_610919.pt"
    else:
        raise ValueError(model_name)

    model_config = config["model_config"]
    d_kv = model_config["d_kv"]
    n_heads = model_config["n_heads"]
    n_layers = model_config["n_layers"]
    max_vocab = model_config["max_vocab"]
    d_model = n_heads * d_kv
    kmer = model_config["kmer"]

    def load_model():
        model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers, eval_mode=True).to(device)
        checkpoint = torch.load(model_checkpoint, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        model_state = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_state}
        model_state.update(state_dict)
        model.load_state_dict(model_state)
        return model.eval()

    dna_tokenizer = DNATokenizer(kmer=kmer)
    seq_dataset = DNADataset(dna_tokenizer, input_seq, data_type="seq", eval_mode=True)

    # NOTE: num_workers>0 often increases RAM usage due to dataset pickling/caching
    dl = DataLoader(seq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    total = len(seq_dataset)
    writer = open_output_writer(total, d_model)

    model = load_model()

    with torch.inference_mode():
        if USE_AUTOCAST:
            with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                for i, batch in enumerate(dl):
                    emb = model(batch[0].to(device))[layer][:, 1:1 + embedding_len, :]
                    emb = emb.detach().float().cpu().numpy()
                    bsz = emb.shape[0]
                    writer.write(i * BATCH_SIZE, emb[:bsz, :embedding_len, :])
        else:
            for i, batch in enumerate(dl):
                emb = model(batch[0].to(device))[layer][:, 1:1 + embedding_len, :]
                emb = emb.detach().float().cpu().numpy()
                bsz = emb.shape[0]
                writer.write(i * BATCH_SIZE, emb[:bsz, :embedding_len, :])

    writer.close()
    return output_file


def run_dnabert():
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}", local_files_only=True)
    model = AutoModel.from_pretrained(f"{model_dir}/{model_name}", local_files_only=True).to(device).eval()

    k_mers = int(model_name.split("_")[-1])
    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    writer = open_output_writer(total, hidden_size)

    def tokenize(seq):
        kmers = " ".join(seq[i:i + k_mers] for i in range(len(seq) - k_mers + 1))
        return tokenizer(kmers, return_tensors="pt", padding="max_length", truncation=True, max_length=embedding_len)

    idx = 0
    with torch.inference_mode():
        for seq in iter_sequences(input_seq):
            x = tokenize(seq)
            x = {k: v.to(device) for k, v in x.items()}
            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                    outputs = model(**x, output_hidden_states=True, return_dict=True)
            else:
                outputs = model(**x, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states
            # keep same slicing as old code
            temp = hs[layer][0, 1:-1, :].detach().float().cpu().numpy()
            L = min(temp.shape[0], embedding_len)
            writer.write(idx, _pad_to_fixed(temp, hidden_size))
            idx += 1

    writer.close()
    return output_file


def run_deepgene():
    sys.path.append(f"{model_dir}/DeepGene-main/PanGeneGraphTrans")
    from modeling_roformer import RoFormerForMaskedLM
    from tokenizers import Tokenizer

    param_file = f"{model_dir}/DeepGene-main/model/pretrain_params_epoch_20"
    model = RoFormerForMaskedLM.from_pretrained(param_file).to(device).eval()

    tok = Tokenizer.from_file(f"{model_dir}/DeepGene-main/data/vocab/tokenizer.json")
    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    writer = open_output_writer(total, hidden_size)

    idx = 0
    with torch.inference_mode():
        for line in open(input_seq, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("sequence"):
                continue
            seq = line.split(",", 1)[0].strip()
            if not seq:
                continue

            ids = tok.encode(seq).ids[:embedding_len]
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            attn = torch.ones_like(input_ids)

            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                    outputs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)

            hs = outputs.hidden_states
            temp = hs[layer][0, :embedding_len, :].detach().float().cpu().numpy()
            L = min(temp.shape[0], embedding_len)
            writer.write(idx, _pad_to_fixed(temp, hidden_size))
            idx += 1

    writer.close()
    return output_file


def run_dnabert2():
    model_path = f"{model_dir}/DNABERT-2-117M"

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, model_max_length=embedding_len, local_files_only=True)

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    writer = open_output_writer(total, hidden_size)

    idx = 0
    with torch.inference_mode():
        for seq in iter_sequences(input_seq):
            x = tokenizer(seq, return_tensors="pt", truncation=True, max_length=embedding_len + 2)
            x = {k: v.to(device) for k, v in x.items()}

            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                    outputs = model(**x, return_dict=True)
            else:
                outputs = model(**x, return_dict=True)

            # outputs[0] is last_hidden_state: [1, L, H]
            # old code used encoded_layers[0][0,1:-1,:]
            last = outputs[0][0, 1:-1, :].detach().float().cpu().numpy()
            L = min(last.shape[0], embedding_len)
            writer.write(idx, _pad_to_fixed(last, hidden_size))
            idx += 1

    writer.close()
    return output_file


def run_generator():
    model_path = f"{model_dir}/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    writer = open_output_writer(total, hidden_size)

    tokenizer.padding_side = "right"

    idx = 0
    with torch.inference_mode():
        for seq in iter_sequences(input_seq):
            inputs = tokenizer(
                [seq],
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(model.config.max_position_embeddings, embedding_len + 2),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            else:
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            hs = outputs.hidden_states
            temp = hs[layer][0, 1:1 + embedding_len, :].detach().float().cpu().numpy()
            L = min(temp.shape[0], embedding_len)
            writer.write(idx, _pad_to_fixed(temp, hidden_size))
            idx += 1

    writer.close()
    return output_file


def run_evo2():
    from evo2 import Evo2
    model_path = f"{model_dir}/{model_name}"
    model = Evo2("evo2_7b", local_path=f"{model_path}.pt").cuda().eval()

    total = count_sequences(input_seq)
    hidden_size = 4096

    writer = open_output_writer(total, hidden_size)

    idx = 0
    with torch.inference_mode():
        for seq in iter_sequences(input_seq):
            input_ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int, device="cuda:0").unsqueeze(0)
            outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer])
            temp = embeddings[layer][0, 1:1 + embedding_len, :].detach().float().cpu().numpy()
            L = min(temp.shape[0], embedding_len)
            writer.write(idx, _pad_to_fixed(temp, hidden_size))
            idx += 1

    writer.close()
    return output_file


def run_lucaone_hf():
    from lucagplm import LucaGPLMModel, LucaGPLMTokenizer
    model_name_hf = f"{model_dir}/LucaOne-default-step36M"

    model = LucaGPLMModel.from_pretrained(model_name_hf).to(device).eval()
    tokenizer = LucaGPLMTokenizer.from_pretrained(model_name_hf)

    total = count_sequences(input_seq)
    hidden_size = model.config.hidden_size

    writer = open_output_writer(total, hidden_size)

    idx = 0
    with torch.inference_mode():
        for seq in iter_sequences(input_seq):
            inputs = tokenizer(
                seq,
                seq_type="gene",
                return_tensors="pt",
                truncation=True,
                max_length=embedding_len + 2,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if USE_AUTOCAST:
                with torch.autocast(device_type="cuda", dtype=_autocast_dtype):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            token_emb = outputs.last_hidden_state[:, 1:-1, :]
            temp = token_emb[0, :embedding_len, :].detach().float().cpu().numpy()
            L = min(temp.shape[0], embedding_len)
            writer.write(idx, _pad_to_fixed(temp, hidden_size))
            idx += 1

    writer.close()
    return output_file


# -------------------------------
# Dispatcher
# -------------------------------
RUNNERS = {
    "bert-series": run_bert_series,
    "dnabert": run_dnabert,
    "dnabert2": run_dnabert2,
    "caduceus": run_caduceus,
    "hyenadna": run_hyenadna,
    "nt": run_nt,
    "omnina": run_omnina,
    "deepgene": run_deepgene,
    "lucaone": run_lucaone_hf,
    "GEN": run_generator,
    "evo2": run_evo2,
}

if model_type not in RUNNERS:
    raise ValueError(f"Unsupported model_type: {model_type}")

RUNNERS[model_type]()

# output already written at output_file
print(f"[OK] saved: {output_file}")
