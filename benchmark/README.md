# Benchmark
## Genomic language model preparation
Download the genomic language model to benchmarking
```
sh models-download.sh
```

## Extracting embeddings
An example for extracting embedding in TATA-promoter (proximal) prediction dataset.
```
sh run-get-embedding.sh ./dataset/promoter/prom_300_tata 300
```

## Downstream applications
### default applications:
```
data_dir="./dataset/promoter/prom_300_tata"
layer=-1 
random_seed=42
model_name=DNABERT-2
output_dim=1
regression=False 
python benchmark-default-tasks.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Exon PSI applications
```
data_dir="./dataset/Exon_PSI"
layer=-1 
model_name=DNABERT-2
random_seed=42
output_dim=56
regression=True 
python benchmark-Exon_PSI.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Gene expression level prediction applications
```
data_dir="./dataset/exp"
model_name=DNABERT-2
layer=-1
random_seed=42
output_dim=53
regression=True
python benchmark-exp.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Gene expression level prediction applications
```
data_dir="./dataset/exp"
layer=-1
random_seed=42
output_dim=53
regression=True
python benchmark-exp.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Promoter-promoter/Enhancer-promoter interaction applications
```
data_dir="./dataset/PPI_EPI/tB/P-P"
layer=-1
random_seed=42
output_dim=1
regression=False
python benchmark-PPI-PEI.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### TFBS/DNA accessibility/Histone modification
```

model_name=DNABERT-2
layer=-1
embedding_len=510
model_dim=768
dataset_dir=./dataset/TFBS
checkpoint_dir=your/path/to/save/checkpoints
eval_dir=your/path/to/save/evaluation/results
python TFBS_510-embedding-all_model.py $model_name $layer 510 ${dataset_dir}/${model_name} $checkpoint_dir $eval_dir
```

### Variant effect prediction
```
data_dir=./variant/var_disease_noncoding
model_name=DNABERT-2
layer=-1
python variant_effect_prediction-zero-shot.py $data_dir $model_name $layer
```
