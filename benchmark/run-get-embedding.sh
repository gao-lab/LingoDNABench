
#######################################
# Global config
#######################################
data_dir=$1
embedding_len=$2

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
    local model_dir=$1
    local model_type=$2
    local model_name=$3
    local emb_len=$4
    local layer=$5
    local filter_cmd=$6

    for dir in $(find "$data_dir" -mindepth 0 -maxdepth 0 -type d); do
        out_dir="$dir/$model_name"
        mkdir -p "$out_dir"

        for file in $(find "$dir" -type f -name "*data*" | eval "$filter_cmd"); do
            echo "$file $dir"
            python get-embeddings-new.py \
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
run_embeddings \
    "./models" \
    "nt" \
    "nucleotide-transformer-2.5b-multi-species" \
    "$((embedding_len/6+1))" \
    "-1" \
    "grep -v embedding"

#######################################
# DNABERT2
#######################################
run_embeddings \
    "./models" \
    "dnabert2" \
    "dnabert2" \
    "$embedding_len" \
    "-1" \
    "grep -v embedding"


# #######################################
# # OmniNA
# #######################################
# run_embeddings \
#     "./models" \
#     "omnina" \
#     "OmniNA-220m" \
#     "$embedding_len" \
#     "-1" \
#     "grep -v embedding"



# #######################################
# # DNABERT
# #######################################
# run_embeddings \
#     "./models" \
#     "dnabert" \
#     "DNA_bert_3" \
#     "$((embedding_len-3+1))" \
#     "-1" \
#     "grep -v embedding"

# #######################################
# # HyenaDNA
# #######################################
# run_embeddings \
#     "./models" \
#     "hyenadna" \
#     "hyenadna-large-1m-seqlen-hf" \
#     "$embedding_len" \
#     "9" \
#     "grep -v embedding"

# #######################################
# # DeepGene
# #######################################
# run_embeddings \
#     "./models" \
#     "deepgene" \
#     "deepgene" \
#     "$embedding_len" \
#     "-1" \
#     "grep -v embedding"

# #######################################
# # Caduceus
# #######################################
# run_embeddings \
#     "./models" \
#     "caduceus" \
#     "caduceus-ps_seqlen-131k_d_model-256_n_layer-16" \
#     "$embedding_len" \
#     "-1" \
#     "grep -v embedding "

# #######################################
# # LucaOne
# #######################################
# run_embeddings \
#     "./models" \
#     "lucaone" \
#     "lucaone" \
#     "$embedding_len" \
#     "-1" \
#     "grep -v embedding "

# #######################################
# # GENERator
# #######################################
# run_embeddings \
#     "./models" \
#     "GEN" \
#     "GENERator-eukaryote-3b-base" \
#     "$((embedding_len/6+1))" \
#     "-1" \
#     "grep -v embedding"

# #######################################
# # Evo2
# #######################################
# run_embeddings \
#     "./models" \
#     "evo2" \
#     "evo2_7b" \
#     "$((embedding_len/6+1))" \
#     "blocks.28.mlp.l3" \
#     "grep -v embedding"
