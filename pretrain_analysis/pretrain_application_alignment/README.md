## Evaluating Checkpoints from Different Pre-Training Stages of a gLM Across Genomic Applications
If the pre-training stage significantly benefits genomic applications, we expect that checkpoints with lower pre-training loss will correspond to adapter models with better performance. Here, we used a gLM trained on genome sequences from multiple species (BERT-Series: MS7-4K-8-K1). In the scripts below, checkpoints from different training epochs are referenced as MS7-4K-8-K1-epoch(e.g., epoch 0 is denoted as MS7-4K-8-K1-0).
### Specify a checkpoint from a pre-training epoch and extract the corresponding sequence embeddings
Demonstration using the Proximal TATA-promoter dataset:
```
data_dir="../benchmark/datasets/promoter/prom_300_tata"
epoch=0
model_name="MS7-4K-8-K1-${epoch}"

for dir in $(find "$data_dir" -mindepth 0 -maxdepth 0 -type d); do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir "$dir/$model_name"
    fi
    for file in $(find "$dir" -type f -name "*data*"); do
        echo "$file $dir"
        python get-embedding-pretrain_loss.py "$file" "$dir/$model_name" $embedding_len -1 $epoch
    done
done
```

### Apply embedding to genomic applications
Follow the process described in ../benchmark/README.md, replacing model_namewith MS7-4K-8-K1-epoch accordingly.
