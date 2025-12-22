#!/bin/sh
# This script is for processing the downloaded reference genomes

cd data
python 01_split_genome.py
python 02_remove_dup.py
python 03_convert_hdf.py
echo "Genome dataset preparation completed."
