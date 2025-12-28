#!/bin/bash


feature_name=$1
data_dir=$2
output_dir=$3
blacklist_dir=$4
ref_genome=$5


bedtools intersect -wa -a ${data_dir}/${feature_name}.bed -b $blacklist_dir > ${output_dir}/${feature_name}_blacklist.bed
touch ${output_dir}/${feature_name}_trim.bed
grep -vf ${output_dir}/${feature_name}_blacklist.bed ${data_dir}/$feature_name.bed > ${output_dir}/${feature_name}_trim.bed

touch ${output_dir}/${feature_name}_train.bed ${output_dir}/${feature_name}_valid.bed ${output_dir}/${feature_name}_test.bed

awk '{if($1 =="chr1" || $1 =="chr8" || $1 =="chr9" ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${output_dir}/${feature_name}_trim.bed > ${output_dir}/${feature_name}_test.bed

awk '{if($1 =="chr2" || $1 =="chr4"  ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${output_dir}/${feature_name}_trim.bed > ${output_dir}/${feature_name}_valid.bed

awk '{if($1 !="chr1" && $1 !="chr2" && $1 !="chr4" && $1 !="chr8" && $1 !="chr9" ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${output_dir}/${feature_name}_trim.bed > ${output_dir}/${feature_name}_train.bed



for elem in train valid test; do bedtools getfasta -fi $ref_genome -bed ${output_dir}/${feature_name}_${elem}.bed -name -fo ${output_dir}/${feature_name}_${elem}.fasta; done


