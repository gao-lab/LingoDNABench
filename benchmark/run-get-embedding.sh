#!/bin/bash
#SBATCH -J get-embedding-cfdna
#SBATCH -p gpu32
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o ../logs/get_embedding_%j.%N.out
#SBATCH --gres=gpu:1
#SBATCH -x c05b26n01


# ## enhancer

data_dir="./dataset/promoter/prom_300_tata"
embedding_len=300

### omnina
model_name="OmniNA-220m"
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`   
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding`
        do
        echo "$file $dir"
        python get-embeddings.py omnina $model_name $file $dir/$model_name $embedding_len 16
    done
done

### nt
model_name="nucleotide-transformer-2.5b-multi-species"
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding` 
        do
        echo "$file $dir"
        python get-embeddings.py nt $model_name $file $dir/$model_name $((embedding_len/6+1)) 32
    done
done

### dnabert2
model_name='dnabert2'
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding` 
        do
        echo "$file $dir"
        python get-embeddings.py $model_name $model_name $file $dir/$model_name $embedding_len 11
    done
done

### dnabert
model_name='DNA_bert_3'
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding` 
        do
        echo "$file $dir"
        python get-embeddings.py dnabert $model_name $file $dir/$model_name $((embedding_len-3+1)) 12
    done
done


model_name=hyenadna-large-1m-seqlen-hf
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding` 
        do
        echo "$file $dir"
        python get-embeddings.py hyenadna $model_name $file $dir/$model_name $embedding_len 9
    done
done


model_name='deepgene'
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding|grep -v part` 
    do
        echo "$file $dir"
        python get-embeddings.py deepgene $model_name $file $dir/$model_name $embedding_len -1
    done
done


model_name="caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding |grep -v part|grep -v enformer` 
        do
        echo "$file $dir"
        python get-embeddings.py caduceus $model_name $file $dir/$model_name $embedding_len -1
    done
done

model_name='lucaone'
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding|grep -v part` 
        do
        echo "$file $dir"
        python get-embeddings.py lucaone $model_name $file $dir/$model_name $embedding_len -1
    done
done



model_name=GENERator-eukaryote-3b-base
for dir in `find $data_dir -mindepth 0 -maxdepth 0 -type d`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"|grep -v embedding|grep -v part` 
        do
        echo "$file $dir"
        python get-embeddings.py GEN $model_name $file $dir/$model_name $((embedding_len/6+1)) -1
    done
done