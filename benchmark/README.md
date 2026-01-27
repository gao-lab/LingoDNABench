# Benchmark
## Genomic language model preparation


| **Model**          | **Pretraining Data**                             | **Model Architecture** | **Tokenization Strategy** | **Pretraining Objectives** | **Number of Parameters** |
| ------------------ | ------------------------------------------------ | ---------------------- | ------------------------- | -------------------------- | ------------------------ |
| **Caduceus-ps**   | Human genome                                     | Mamba                  | 1-mer                     | MLM                        | 1.9M                     |
| **HyenaDNA-1M**   | Human genome                                     | Hyena                  | 1-mer                     | CLM                        | 6.6M                     |
| **DeepGene**      | Human pan-genome                                 | Transformer            | BPE                       | MLM                        | 85M                      |
| **GPN-MSA**       | MSA from 100 vertebrates                         | Transformer            | 1-mer                     | MLM                        | 86M                      |
| **DNABERT-3mer**  | Human genome                                     | Transformer            | Overlapped 3-mer          | MLM                        | 89M                      |
| **DNABERT-2**     | 135 species genomes                              | Transformer            | BPE                       | MLM                        | 117M                     |
| **OmniNA-220M**   | 172 species genomes                              | Transformer (LLaMA)    | BPE                       | CLM                        | 220M                     |
| **LucaOne**       | DNA, RNA, protein sequences from 169,861 species | Transformer            | 1-mer                     | MLM                        | 1.8B                     |
| **NT-2.5B-MS**   | 850 species genomes                              | Transformer            | Non-overlapped 6-mer      | MLM                        | 2.5B                     |
| **GENERator-3B** | Multi-species genomes                            | Transformer            | Non-overlapped 6-mer      | CLM                        | 3B                       |
| **Evo2-7B**      | Genome sequences from 128M species                                           | SSM                    | 1-mer                     | CLM                        | 7B                       |



Download the genomic language model to benchmarking
```
pip install huggingface-cli==0.36.0
sh models-download.sh ./models
```

## Model-Specific Environments and References

Due to substantial differences in architectures, dependencies, and CUDA requirements,
 **each genomic language model (gLM) was evaluated in its own validated software environment**, following the official implementation released by the original authors.

We do **not** attempt to unify model environments.
 Instead, we standardize **evaluation protocols, datasets, and metrics**, while respecting model-specific execution requirements.

Below we list the models evaluated in this benchmark, together with the corresponding reference implementations.

------

### bert-series

``` bash
conda create -n bert-series python=3.9 pip -c conda-forge
conda activate bert-series

pip install \
  torch==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# download from https://github.com/Dao-AILab
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl  
```

### Caduceus-ps

- **Official repository:** https://github.com/kuleshov-group/caduceus
- **Hugging Face:** https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16
- **Inference:** official inference pipeline
- **Notes:** state-space–based sequence modeling

------

### HyenaDNA-1M

- **Official repository:** https://github.com/HazyResearch/hyena-dna
- **Hugging Face:** https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf
- **Inference:** official implementation
- **Notes:** requires custom CUDA extensions (e.g., `causal_conv1d`)

------

### DeepGene

- **Official repository:** https://github.com/wds-seu/DeepGene/blob/main/README.md
- **Inference:** official inference scripts
- **Notes:** evaluated using pretrained checkpoints released by the authors

------

### DNABERT-3mer

- **Official repository:** https://github.com/jerryji1993/DNABERT
- **Hugging Face:** https://huggingface.co/zhihan1996/DNA_bert_3
- **Inference:** Hugging Face pipeline
- **Notes:** k-mer–based tokenization (k=3)

------

### DNABERT-2

- **Official repository:** https://github.com/MAGICS-LAB/DNABERT_2
- **Hugging Face:** https://huggingface.co/zhihan1996/DNABERT-2-117M
- **Inference:** Hugging Face pipeline
- **Notes:** byte-level tokenizer

------

### GPN-MSA

- **Official repository:** https://github.com/songlab-cal/gpn
- **Inference:** official inference code
- **Notes:** pretrained with MSA-aware objectives; evaluated following authors’ recommended settings

------

### OmniNA

- **Hugging Face:** https://huggingface.co/XLS/OmniNA-220m
- **Inference:** Hugging Face pipeline
- **Notes:** long-range genomic sequence modeling; environment tightly coupled to released codebase

------

### LucaOne

- **Hugging Face:** https://huggingface.co/LucaGroup/LucaOne-default-step36M
- **Inference:** official inference pipeline
- **Notes:** multi-task and multi-modal pretraining objectives

------

### NT-MS-2.5B

- **Official repository:** https://github.com/instadeepai/nucleotide-transformer/tree/main/nucleotide_transformer
- **Hugging face:** https://huggingface.co/InstaDeepAI/nucleotide-transformer-2.5b-multi-species
- **Inference:** official inference scripts
- **Notes:** large-scale parameterization

------

### GENERator

- **Official repository:** https://github.com/GenerTeam/GENERator
- **Hugging face:** https://huggingface.co/GenerTeam/GENERanno-eukaryote-0.5b-base
- **Inference:** official generative inference pipeline
- **Notes:** generative genomic modeling objective

------

### Evo2

- **Official repository:** https://github.com/arcinstitute/evo2
- **Inference:** official inference implementation
- **Notes:** pretrained on mixed genomic sources, including eukaryotic and prokaryotic genomes

------

## Extracting embeddings
An example for extracting embedding in TATA-promoter (proximal) prediction dataset.
```
input_seq=datasets/promoter/prom_300_tata/dev_data_0.txt
model_type=nt
model_name=nucleotide-transformer-2.5b-multi-species
embedding_len=300
out_dir=datasets/promoter/prom_300_tata/$model_name
mkdir -p $out_dir
layer=-1
python get-embeddings.py ./models $model_type $model_name $input_seq $out_dir $((embedding_len/6+1)) -1
```
A script for extracting embedding in gLMs (For quick testing, only NT-2.5b-MS and DNABERT2 are not commented out in the script)
```
sh run-get-embedding.sh ./datasets/promoter/prom_300_tata 300
```


## Downstream applications
### default applications:
```
# default applications
data_dir="./datasets/promoter/prom_300_tata"
model_name=nucleotide-transformer-2.5b-multi-species
layer=-1 # the last layer
random_seed=42
output_dim=1
regression=False #classification application
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression
```

### Exon PSI applications
```
data_dir="./datasets/Exon_PSI"
layer=-1 
model_name=nucleotide-transformer-2.5b-multi-species
random_seed=42
output_dim=56
regression=True 
python scirpt/benchmark-Exon_PSI.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Gene expression level prediction applications
```
data_dir="./datasets/exp"
model_name=nucleotide-transformer-2.5b-multi-species
layer=-1
random_seed=42
output_dim=53
regression=True
python scirpt/benchmark-exp.py  $model_name ${data_dir} $layer  $random_seed $output_dim $regression
```

### Promoter-promoter/Enhancer-promoter interaction applications
```
model_name=nucleotide-transformer-2.5b-multi-species
data_dir="./datasets/PPI_EPI/tB/P-P"
layer=-1
random_seed=42
output_dim=1
regression=False
python benchmark-PPI-PEI.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### TFBS/DNA accessibility/Histone modification
```
model_name=nucleotide-transformer-2.5b-multi-species
layer=-1
embedding_len=510
model_dim=2560
dataset_dir=./datasets/TFBS
checkpoint_dir=../datasets/TFBS/checkpoint
eval_dir=../datasets/TFBS/eval 
python TFBS_510-embedding-all_model.py nucleotide-transformer-2.5b-multi-species -1 $((embedding_len / 6 + 1 )) $model_dim $data_dir $checkpoint_dir $eval_dir

```

### Variant effect prediction
```
data_dir=.datasets/variant/var_disease_noncoding
model_name=nucleotide-transformer-2.5b-multi-species
layer=-1
python variant_effect_prediction-zero-shot.py $data_dir $model_name $layer
```
