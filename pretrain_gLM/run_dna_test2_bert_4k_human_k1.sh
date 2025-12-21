#!/bin/bash
#SBATCH -J dna_test2_bert_4k_human_k1
#SBATCH -p gpu32
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -x c05b26n01
#SBATCH -o ./log/dna_test2_bert_4k_human_k1_%j.%N.out
#SBATCH --gres=gpu:1

source activate
conda activate dnalingo

torchrun --nnodes 1 --nproc_per_node 1 --master-port 12670 dna_test2_bert_4k_human_k1.py