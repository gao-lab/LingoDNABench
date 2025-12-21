#!/bin/bash


feature_name="TFBS"
data_dir="../dataset/TFBS"


bedtools intersect -wa -a $feature_name.bed -b /path/to/blacklist/hg38-blacklist.v2.bed > ${feature_name}_blacklist.bed
touch ${feature_name}_trim.bed
grep -vf ${feature_name}_blacklist.bed $feature_name.bed > ${feature_name}_trim.bed

touch ${feature_name}_train.bed ${feature_name}_valid.bed ${feature_name}_test.bed

awk '{if($1 =="chr1" || $1 =="chr8" || $1 =="chr9" ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${feature_name}_trim.bed > ${feature_name}_test.bed

awk '{if($1 =="chr2" || $1 =="chr4"  ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${feature_name}_trim.bed > ${feature_name}_valid.bed

awk '{if($1 !="chr1" && $1 !="chr2" && $1 !="chr4" && $1 !="chr8" && $1 !="chr9" ) printf "%s\t%s\t%s\t%s\n",$1,$2,$3,$7}' ${feature_name}_trim.bed > ${feature_name}_train.bed



for elem in train valid test; do bedtools getfasta -fi /path/to/reference/genome/GRCh38.primary_assembly.genome.fa -bed ${feature_name}_${elem}.bed -name -fo ${feature_name}_${elem}.fasta; done


