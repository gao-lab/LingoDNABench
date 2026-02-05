# Benchmark: Genomic Language Models (gLMs)

This repository benchmarks multiple genomic language models (gLMs) by **standardizing datasets, evaluation protocols, and metrics**, while allowing **model-specific environments** (due to substantial differences in architectures, dependencies, and CUDA requirements).

---

## Table of Contents
- [1. Supported Models](#1-supported-models)
- [2. Quickstart](#2-quickstart)
  - [2.1 Download Models](#21-download-models)
- [3. Model-Specific Environments](#3-model-specific-environments)
  - [3.1 bert-series Environment](#31-bert-series-environment)
  - [3.2 Model References](#32-model-references)
- [4. Extract Embeddings](#4-extract-embeddings)
- [5. Downstream Evaluation](#5-downstream-evaluation)
  - [5.1 Evaluation Environment Setup](#51-evaluation-environment-setup)
  - [5.2 Default Applications](#52-default-applications)
  - [5.3 Exon PSI](#53-exon-psi)
  - [5.4 Gene Expression](#54-gene-expression)
  - [5.5 PPI / EPI](#55-ppi--epi)
  - [5.6 TFBS / Accessibility / Histone](#56-tfbs--accessibility--histone)
  - [5.7 Variant Effect Prediction](#57-variant-effect-prediction)

---

## 1. Supported Models

| Model | Pretraining Data | Architecture | Tokenization | Objective | Params |
|---|---|---|---|---|---:|
| Caduceus-ps | Human genome | Mamba | 1-mer | MLM | 1.9M |
| HyenaDNA-1M | Human genome | Hyena | 1-mer | CLM | 6.6M |
| DeepGene | Human pan-genome | Transformer | BPE | MLM | 85M |
| GPN-MSA | MSA from 100 vertebrates | Transformer | 1-mer | MLM | 86M |
| DNABERT (3-mer) | Human genome | Transformer | overlapped 3-mer | MLM | 89M |
| DNABERT-2 | 135 species genomes | Transformer | BPE (byte-level) | MLM | 117M |
| OmniNA-220M | 172 species genomes | Transformer (LLaMA) | BPE | CLM | 220M |
| LucaOne | DNA/RNA/protein from 169,861 species | Transformer | 1-mer | MLM | 1.8B |
| NT-2.5B-MS | 850 species genomes | Transformer | non-overlapped 6-mer | MLM | 2.5B |
| GENERator-3B | Multi-species genomes | Transformer | non-overlapped 6-mer | CLM | 3B |
| Evo2-7B | Genome sequences from 128M species | SSM | 1-mer | CLM | 7B |

> Notes:
> - Some models require custom CUDA extensions; do not expect a single unified environment.

---

## 2. Quickstart

### 2.1 Download Models

```bash
pip install huggingface-cli==0.36.0
sh models-download.sh ./models
```

---

## 3. Model-Specific Environments

We **do not** unify model environments.  
Instead, each gLM is evaluated in its own validated environment, following the official implementation released by the original authors.

### 3.1 bert-series Environment

```bash
#download the checkpoints of BERT-Series models
wget -r -np -nH --cut-dirs=2 -R "index.html*"  \
  "http://ftp.cbi.pku.edu.cn/pub/LingoDNABench/BERT-155M-Series/"

mv BERT-155M-Series ../pretrain_analysis/pretrain_application_alignment/
```


```bash
conda create -n bert-series python=3.9 pip -c conda-forge
conda activate bert-series

pip install   torch==2.6.0+cu124   --index-url https://download.pytorch.org/whl/cu124

# flash-attn (from Dao-AILab)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```

### 3.2 Model References

#### Caduceus-ps
- Official repo: https://github.com/kuleshov-group/caduceus
- HF: https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
- Inference: official pipeline
- Notes: state-space based sequence modeling

#### HyenaDNA-1M
- Official repo: https://github.com/HazyResearch/hyena-dna
- HF: https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf
- Inference: official implementation
- Notes: requires custom CUDA extensions (e.g., `causal_conv1d`)

#### DeepGene
- Official repo: https://github.com/wds-seu/DeepGene/blob/main/README.md
- Inference: official inference scripts
- Notes: evaluated using pretrained checkpoints released by the authors

#### DNABERT (3-mer)
- Official repo: https://github.com/jerryji1993/DNABERT
- HF: https://huggingface.co/zhihan1996/DNA_bert_3
- Inference: Hugging Face pipeline
- Notes: k-mer tokenization (k=3)

#### DNABERT-2
- Official repo: https://github.com/MAGICS-LAB/DNABERT_2
- HF: https://huggingface.co/zhihan1996/DNABERT-2-117M
- Inference: Hugging Face pipeline
- Notes: byte-level tokenizer

#### GPN-MSA
- Official repo: https://github.com/songlab-cal/gpn
- Inference: official inference code
- Notes: MSA-aware objectives; follow authors’ settings

#### OmniNA
- HF: https://huggingface.co/XLS/OmniNA-220m
- Inference: Hugging Face pipeline
- Notes: long-range genomic modeling; environment coupled to released codebase

#### LucaOne
- HF: https://huggingface.co/LucaGroup/LucaOne-default-step36M
- Inference: official inference pipeline
- Notes: multi-task and multi-modal objectives

#### NT-2.5B-MS
- Official repo: https://github.com/instadeepai/nucleotide-transformer/tree/main/nucleotide_transformer
- HF: https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
- Inference: official scripts
- Notes: large model; pay attention to GPU memory

#### GENERator
- Official repo: https://github.com/GenerTeam/GENERator
- HF: https://huggingface.co/GenerTeam/GENERanno-eukaryote-0.5b-base
- Inference: official generative pipeline
- Notes: generative objective

#### Evo2
- Official repo: https://github.com/arcinstitute/evo2
- Inference: official implementation
- Notes: mixed genomic sources (eukaryotic + prokaryotic)

---

## 4. Extract Embeddings

Example: extracting embeddings for the **TATA promoter (proximal)** prediction dataset.

```bash
input_seq=datasets/promoter/prom_300_tata/dev_data_0.txt
model_type=nt
model_name=nucleotide-transformer-2.5b-multi-species
embedding_len=300
out_dir=datasets/promoter/prom_300_tata/${model_name}
mkdir -p "${out_dir}"

layer=-1  # last layer
# NOTE: for NT with non-overlapped 6-mer tokenization, token length is approx embedding_len/6 (+1)
python get-embeddings.py ./models "${model_type}" "${model_name}" "${input_seq}" "${out_dir}" "$((embedding_len/6+1))" "${layer}"
```

Batch script (for quick testing, only NT-2.5B-MS and DNABERT-2 are enabled by default in the script):
```bash
sh run-get-embedding.sh ./datasets/promoter/prom_300_tata 300
```

---

## 5. Downstream Evaluation

### 5.1 Evaluation Environment Setup

#### PyTorch evaluation env
```bash
conda create -n torch-eval python=3.9 pip -c conda-forge
conda activate torch-eval

pip install pandas scikit-learn biopython
pip install   torch==2.6.0+cu124   --index-url https://download.pytorch.org/whl/cu124
```

#### TensorFlow evaluation env (For TFBS/HM/DNA_accessibility)
```bash
conda create -n tf-eval python=3.10 pip -c conda-forge
conda activate tf-eval

pip install "tensorflow[and-cuda]"==2.16.1 scikit-learn
```
---

### 5.2 Default Applications

```bash
data_dir="./datasets/promoter/prom_300_tata"
model_name="nucleotide-transformer-2.5b-multi-species"
layer=-1
random_seed=42
output_dim=1
regression=False  # classification

python script/benchmark-default-applications.py   "${model_name}" "${data_dir}" "${layer}" "${random_seed}" "${output_dim}" "${regression}"
```

### 5.3 Exon PSI

```bash
data_dir="./datasets/Exon_PSI"
model_name="nucleotide-transformer-2.5b-multi-species"
layer=-1
random_seed=42
output_dim=56
regression=True

python script/benchmark-Exon_PSI.py   "${model_name}" "${data_dir}/${model_name}" "${layer}" "${random_seed}" "${output_dim}" "${regression}"
```

### 5.4 Gene Expression

```bash
data_dir="./datasets/exp"
model_name="nucleotide-transformer-2.5b-multi-species"
layer=-1
random_seed=42
output_dim=53
regression=True

python script/benchmark-exp.py   "${model_name}" "${data_dir}" "${layer}" "${random_seed}" "${output_dim}" "${regression}"
```

### 5.5 PPI / EPI

```bash
model_name="nucleotide-transformer-2.5b-multi-species"
data_dir="./datasets/PPI_EPI/tB/P-P"
layer=-1
random_seed=42
output_dim=1
regression=False

python script/benchmark-PPI-PEI.py   "${model_name}" "${data_dir}/${model_name}" "${layer}" "${random_seed}" "${output_dim}" "${regression}"
```

### 5.6 TFBS / Accessibility / Histone

```bash
model_name="nucleotide-transformer-2.5b-multi-species"
layer=-1
embedding_len=510
model_dim=2560

dataset_dir="./datasets/TFBS"
checkpoint_dir="./datasets/TFBS/checkpoint"
eval_dir="./datasets/TFBS/eval"

python script/TFBS_510-embedding-all_model.py   "${model_name}" "${layer}" "$((embedding_len/6+1))" "${model_dim}"   "${dataset_dir}" "${checkpoint_dir}" "${eval_dir}"
```

### 5.7 Variant Effect Prediction (zero-shot)

```bash
data_dir="./datasets/variant/var_disease_noncoding"
model_name="nucleotide-transformer-2.5b-multi-species"
layer=-1

python script/variant_effect_prediction-zero-shot.py   "${data_dir}" "${model_name}" "${layer}"
```
