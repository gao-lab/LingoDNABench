#!/bin/bash


feature_name=$1
data_dir=$2
output_dir=$3
ref_genome=$4



for elem in train valid test; do bedtools getfasta -fi $ref_genome -bed ${output_dir}/${feature_name}_${elem}.bed -name -fo ${output_dir}/${feature_name}_${elem}.fasta; done


