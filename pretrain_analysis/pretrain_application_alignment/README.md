# Evaluating gLM Pretraining Checkpoints Across Genomic Applications

If pretraining progress meaningfully improves downstream genomic tasks, then **later checkpoints (lower pretraining loss)** should tend to yield **better downstream performance** after adapter training.

This document shows how to evaluate checkpoints from different pretraining epochs for a multi-species BERT-style gLM:

- **Model family:** `BERT-Series: model_M`
- **Checkpoint naming:** `model_M-<epoch>` (e.g., epoch 0 → `model_M-0`)

> Inspired by: https://doi.org/10.1101/2024.02.05.578959

---

## Overview

For each epoch checkpoint:

1. **Extract embeddings** for all `*data*` files in a dataset directory (e.g., `train_data_0.txt`, `dev_data_0.txt`, `test_data_0.txt`, etc.).
2. **Run downstream evaluation** (e.g., promoter classification) using the generated embeddings.

---

## 1) Extract embeddings for a specific epoch

Example dataset: **Proximal TATA-promoter** (`prom_300_tata`).

Create a script file (e.g., `run-embed-pretrain-loss.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Function: run_embedding_extraction
# Arguments:
#   $1: data_dir       Dataset directory (contains *data* files)
#   $2: epoch          Pretraining epoch (integer)
#   $3: layer          Embedding layer index (-1 for last layer)
#   $4: embedding_len  Input sequence length in bp (e.g., 300)
# =========================================================
run_embedding_extraction() {
  local data_dir="$1"
  local epoch="$2"
  local layer="$3"
  local embedding_len="$4"

  local model_name="model_M-${epoch}"
  local output_dir="${data_dir}/${model_name}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "Error: data_dir does not exist: ${data_dir}" >&2
    return 1
  fi

  mkdir -p "${output_dir}"

  # Process all files containing "data" in filename at the top level of data_dir
  find "${data_dir}" -maxdepth 1 -type f -name "*data*" | while read -r file; do
    echo "[epoch ${epoch}] Processing: ${file}"
    python get-embedding-pretrain_loss.py \
      "${file}" \
      "${output_dir}" \
      "${embedding_len}" \
      "${layer}" \
      "${epoch}"
  done

  echo "[epoch ${epoch}] Embeddings saved to: ${output_dir}"
}

# -------------------------
# Example usage
# -------------------------
data_dir="../benchmark/datasets/promoter/prom_300_tata"
epoch=0
layer=-1
embedding_len=300

run_embedding_extraction "${data_dir}" "${epoch}" "${layer}" "${embedding_len}"
```

Run:

```bash
bash run-embed-pretrain-loss.sh
```

> Output embeddings will be stored under:
> `../benchmark/datasets/promoter/prom_300_tata/MS7-4K-8-K1-<epoch>/`

---

## 2) Loop over epochs: extract embeddings + evaluate

Below is a **single end-to-end loop** that:
- extracts embeddings for each epoch, then
- runs the downstream benchmark script using that epoch’s `model_name`.

```bash
#!/usr/bin/env bash
set -euo pipefail

data_dir="../benchmark/datasets/promoter/prom_300_tata"
layer=-1
embedding_len=300

random_seed=42
output_dim=1
regression=False   # classification

for epoch in {0..8}; do
  model_name="model_M-${epoch}"
  output_dir="${data_dir}/${model_name}"
  mkdir -p "${output_dir}"

  echo "=============================="
  echo "Epoch: ${epoch}  Model: ${model_name}"
  echo "=============================="

  # 1) embeddings
  find "${data_dir}" -maxdepth 1 -type f -name "*data*" | while read -r file; do
    python get-embedding-pretrain_loss.py \
      "${file}" \
      "${output_dir}" \
      "${embedding_len}" \
      "${layer}" \
      "${epoch}"
  done

  # 2) downstream evaluation
  python script/benchmark-default-applications.py \
    "${model_name}" \
    "${data_dir}" \
    "${layer}" \
    "${random_seed}" \
    "${output_dim}" \
    "${regression}"
done
```

---

## 3) Apply to other tasks

Follow the standard benchmark workflow described in `../../benchmark/README.md`, but **replace** `model_name` with:

- `model_M-0`, `model_M-1`, …, `model_M-8`

and ensure embeddings for that `model_name` exist under the dataset directory before running evaluation.

