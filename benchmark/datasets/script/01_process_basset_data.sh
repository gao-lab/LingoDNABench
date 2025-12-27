#!/bin/bash


feature_name=$1
data_dir=$2


bedtools intersect -wa -a ${data_dir}/${feature_name}.bed -b ../../../data/hg38-blacklist.v2.bed > ${data_dir}/${feature_name}_blacklist.bed
touch ${data_dir}/${feature_name}_trim.bed
grep -vf ${data_dir}/${feature_name}_blacklist.bed ${data_dir}/$feature_name.bed > ${data_dir}/${feature_name}_trim.bed

touch ${data_dir}/${feature_name}_train.bed ${data_dir}/${feature_name}_valid.bed ${data_dir}/${feature_name}_test.bed

awk '{if($1 =="chr1" || $1 =="chr8" || $1 =="chr9" ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${data_dir}/${feature_name}_trim.bed > ${data_dir}/${feature_name}_test.bed

awk '{if($1 =="chr2" || $1 =="chr4"  ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${data_dir}/${feature_name}_trim.bed > ${data_dir}/${feature_name}_valid.bed

awk '{if($1 !="chr1" && $1 !="chr2" && $1 !="chr4" && $1 !="chr8" && $1 !="chr9" ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${data_dir}/${feature_name}_trim.bed > ${data_dir}/${feature_name}_train.bed



for elem in train valid test; do bedtools getfasta -fi ../../../data/GRCh38.primary_assembly.genome.fa -bed ${data_dir}/${feature_name}_${elem}.bed -name -fo ${data_dir}/${feature_name}_${elem}.fasta; done


