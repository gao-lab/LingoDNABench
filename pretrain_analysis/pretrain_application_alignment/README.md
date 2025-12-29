## Evaluating Checkpoints from Different Pre-Training Stages of a gLM Across Genomic Applications
If the pre-training stage significantly benefits genomic applications, we expect that checkpoints with lower pre-training loss will correspond to adapter models with better performance. Here, we used a gLM trained on genome sequences from multiple species (`BERT-Series: MS7-4K-8-K1`). In the scripts below, checkpoints from different training epochs are referenced as `MS7-4K-8-K1-epoch`(e.g., epoch 0 is denoted as `MS7-4K-8-K1-0`). (This idea was inspired by https://doi.org/10.1101/2024.02.05.578959)
### Specify a checkpoint from a pre-training epoch and extract the corresponding sequence embeddings
Demonstration using the Proximal TATA-promoter dataset:
```
#!/bin/bash

# ========================================================
# Function: run_embedding_extraction
# Parameters:
#   $1: data_dir      dataset directory
#   $2: epoch         the specified epoch of pretrain checkpoint
#   $3: layer         embedding layer
#   $4: embedding_len the length of input embedding
# ========================================================
run_embedding_extraction() {
    local data_dir="$1"
    local epoch="$2"
    local layer="$3"
    local embedding_len="$4"
    
    local model_name="MS7-4K-8-K1-${epoch}"
    if [ ! -d "$data_dir" ]; then
        echo "Error: Directory $data_dir does not exist."
        return 1
    fi
    find "$data_dir" -mindepth 0 -maxdepth 0 -type d | while read -r dir; do
        
        local output_dir="$dir/$model_name"
        
        
        if [ ! -d "$output_dir" ]; then
            mkdir -p "$output_dir"
        fi

        find "$dir" -maxdepth 1 -type f -name "*data*" | while read -r file; do
            echo "Processing: $file in $dir"
            python get-embedding-pretrain_loss.py \
                "$file" \
                "$output_dir" \
                "$embedding_len" \
                "$layer" \
                "$epoch"
        done
    done
}

# =========================
# Usage example
# =========================

data_dir="../benchmark/datasets/promoter/prom_300_tata"
epoch=0
layer=-1
embedding_len=300

for EPOCH in {0..8}:
do
run_embedding_extraction "$data_dir" "$epoch" "$layer" 
random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression
done

```

### Apply embedding to genomic applications
Follow the process described in `../benchmark/README.md`, replacing model_namewith `MS7-4K-8-K1-epoch` accordingly.
