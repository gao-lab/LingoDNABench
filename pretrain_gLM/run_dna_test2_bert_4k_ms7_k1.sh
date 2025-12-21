#!/bin/bash
#SBATCH -J dna_test2_bert_4k_ms7_k1
#SBATCH -p gpu32
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o ./log/dna_test2_bert_4k_ms7_k1_%j.%N.out
#SBATCH -x c05b27n04
#SBATCH --gres=gpu:4

source activate
conda activate dnalingo

torchrun --nnodes 1 --nproc_per_node 4 --master-port 19690 dna_test2_bert_4k_ms7_k1.py