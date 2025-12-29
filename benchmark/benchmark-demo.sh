
#######################################
# Common embedding runner
# Arguments:
#   $1: backend name (omnina / nt / dnabert / ...)
#   $2: model name
#   $3: embedding length
#   $4: embedding layer, -1 means the last layer
#   $5: file filter (grep chain)
#######################################
run_embeddings () {
    local data_dir=$1
    local model_dir=$2
    local model_type=$3
    local model_name=$4
    local emb_len=$5
    local layer=$6
    local filter_cmd=$7

    for dir in $(find "$data_dir" -mindepth 0 -maxdepth 0 -type d); do
        out_dir="$dir/$model_name"
        mkdir -p "$out_dir"

        for file in $(find "$dir" -type f -name "*data*" | eval "$filter_cmd"); do
            echo "$file $dir"
            python get-embeddings.py \
                "$data_dir" \
                "$model_dir" \
                "$model_type" \
                "$model_name" \
                "$file" \
                "$out_dir" \
                "$emb_len" \
                "$layer"
        done
    done
}


#######################################
# Nucleotide Transformer
#######################################
model_type=nt
model_name=nucleotide-transformer-2.5b-multi-species
model_dir=./models
layer=-1

# Promoter
## Promoter proximal all
data_dir=./datasets/promoter/prom_300_all
embedding_len=300
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

#Promoter proximal no-TATA
data_dir=./datasets/promoter/prom_300_notata
embedding_len=300
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

#Promoter proximal TATA
data_dir=./datasets/promoter/prom_300_tata
embedding_len=300
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

#Promoter core all
data_dir=./datasets/promoter/prom_core_all
embedding_len=70
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

#Promoter core no TATA
data_dir=./datasets/promoter/prom_core_notata
embedding_len=70
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

#Promoter core TATA
data_dir=./datasets/promoter/prom_core_tata
embedding_len=70
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Enhancer
## human
data_dir=./datasets/enhancer/human
embedding_len=200
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression


## mouse
data_dir=./datasets/enhancer/mouse
embedding_len=200
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# DNA methylation
## 5mC
data_dir=./datasets/DNA_methylation/5mC
embedding_len=41
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

## 6mA
data_dir=./datasets/DNA_methylation/6mA
embedding_len=41
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Silencer
data_dir=./datasets/silencer
embedding_len=200
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# CRE activity
## HepG2
data_dir=./datasets/CRE_activity/HepG2
embedding_len=230
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=True 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

## K562
data_dir=./datasets/CRE_activity/K562
embedding_len=230
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=True 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Splice site
## acceptor
data_dir=./datasets/splice_site/acceptor
embedding_len=400
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

## donor
data_dir=./datasets/splice_site/donor
embedding_len=400
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# PAS all
data_dir=./datasets/PAS
embedding_len=600
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# PAS AATAAA only
data_dir=./datasets/PAS_AATAAA
embedding_len=600
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# TIS
data_dir=./datasets/TIS
embedding_len=600
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Intronretention
data_dir=./datasets/TIS
embedding_len=600
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=False 
python scirpt/benchmark-default-applications.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Bulk RNA-seq
data_dir=./datasets/exp
embedding_len=2000
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=53
regression=True 
python scirpt/benchmark-exp.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Exon_PSI
data_dir=./datasets/Exon_PSI
embedding_len=600
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=56
regression=True 
python scirpt/benchmark-Exon_PSI.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# PPI
## tB
data_dir=./datasets/PPI/tB
embedding_len=1000
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding"

random_seed=42
output_dim=1
regression=True 
python scirpt/benchmark-PPI-EPI.py  $model_name $data_dir $layer  $random_seed $output_dim 


# EPI
## tB
data_dir=./datasets/EPI/tB

embedding_len=2000
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding |grep data_0"

embedding_len=1000
run_embeddings \
    "$data_dir"\
    "$model_dir" \
    "$model_type" \
    "$model_name" \
    "$((embedding_len/6+1))" \
    "$layer" \
    "grep -v embedding |grep data_1"

random_seed=42
output_dim=1
regression=True 
python scirpt/benchmark-PPI-EPI.py  $model_name $data_dir $layer  $random_seed $output_dim 


# Variant
## disease-related 
data_dir=./datasets/variant/var_disease_coding
variant_effect_prediction-zero-shot.py $data_dir $model_name $layer

data_dir=./datasets/variant/var_disease_noncoding
variant_effect_prediction-zero-shot.py $data_dir $model_name $layer

## transcript-related
data_dir=./datasets/variant/var_expression_eQTL
variant_effect_prediction-zero-shot.py $data_dir $model_name $layer

data_dir=./datasets/variant/var_expression_MPRA
variant_effect_prediction-zero-shot.py $data_dir $model_name $layer



