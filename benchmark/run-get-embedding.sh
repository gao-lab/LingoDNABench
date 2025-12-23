#!/usr/bin/env bash
set -e

#######################################
# Global config
#######################################
data_dir="./dataset/promoter/prom_300_tata"
embedding_len=300

#######################################
# Common embedding runner
# Arguments:
#   $1: backend name (omnina / nt / dnabert / ...)
#   $2: model name
#   $3: embedding length
#   $4: extra argument (e.g. -1 / 9)
#   $5: file filter (grep chain)
#######################################
run_embeddings () {
    local backend=$1
    local model_name=$2
    local emb_len=$3
    local extra_arg=$4
    local filter_cmd=$5

    for dir in $(find "$data_dir" -mindepth 0 -maxdepth 0 -type d); do
        out_dir="$dir/$model_name"
        mkdir -p "$out_dir"

        for file in $(find "$dir" -type f -name "*data*" | eval "$filter_cmd"); do
            echo "$file $dir"
            python get-embeddings.py \
                "$backend" \
                "$model_name" \
                "$file" \
                "$out_dir" \
                "$emb_len" \
                "$extra_arg"
        done
    done
}

#######################################
# OmniNA
#######################################
run_embeddings \
    "omnina" \
    "OmniNA-220m" \
    "$embedding_len" \
    "-1" \
    "grep -v embedding"

#######################################
# Nucleotide Transformer
#######################################
run_embeddings \
    "nt" \
    "nucleotide-transformer-2.5b-multi-species" \
    "$((embedding_len/6+1))" \
    "-1" \
    "grep -v embedding"

#######################################
# DNABERT2
#######################################
run_embeddings \
    "dnabert2" \
    "dnabert2" \
    "$embedding_len" \
    "-1" \
    "grep -v embedding"

#######################################
# DNABERT
#######################################
run_embeddings \
    "dnabert" \
    "DNA_bert_3" \
    "$((embedding_len-3+1))" \
    "-1" \
    "grep -v embedding"

#######################################
# HyenaDNA
#######################################
run_embeddings \
    "hyenadna" \
    "hyenadna-large-1m-seqlen-hf" \
    "$embedding_len" \
    "9" \
    "grep -v embedding"

#######################################
# DeepGene
#######################################
run_embeddings \
    "deepgene" \
    "deepgene" \
    "$embedding_len" \
    "-1" \
    "grep -v embedding | grep -v part"

#######################################
# Caduceus
#######################################
run_embeddings \
    "caduceus" \
    "caduceus-ps_seqlen-131k_d_model-256_n_layer-16" \
    "$embedding_len" \
    "-1" \
    "grep -v embedding | grep -v part | grep -v enformer"

#######################################
# LucaOne
#######################################
run_embeddings \
    "lucaone" \
    "lucaone" \
    "$embedding_len" \
    "-1" \
    "grep -v embedding | grep -v part"

#######################################
# GENERator
#######################################
run_embeddings \
    "GEN" \
    "GENERator-eukaryote-3b-base" \
    "$((embedding_len/6+1))" \
    "-1" \
    "grep -v embedding | grep -v part"
