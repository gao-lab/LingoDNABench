# Benchmark
## Genomic language model preparation
Download the genomic language model to benchmarking
```
sh models-download.sh ./models
```

## Extracting embeddings
An example for extracting embedding in TATA-promoter (proximal) prediction dataset.
```
input_seq= datasets/promoter/prom_300_tata/dev_data_0.txt
model_type=nt
model_name=nucleotide-transformer-2.5b-multi-species
embedding_len=300
out_dir=datasets/promoter/prom_300_tata/$model_name
mkdir -p $out_dir
layer=-1
python get-embeddings.py $model_type $model_name $input_seq $out_dir $((embedding_len/6+1)) -1
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
