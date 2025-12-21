import subprocess
import os
from Bio import SeqIO
import re
import gzip


max_seq_length = 4096
n_ratio = 0.01

raw_genome_dir = "./raw/"
log_dir = "./log/"
processed_dir = "./processed_" + str(max_seq_length) + "/"
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)


def split_genome(input_genome):
    print("Processing " + input_genome)
    removed_seq_num = 0
    seq_num = 0
    out_file = input_genome.replace(".fna.gz", "_" + str(max_seq_length) + ".seq")
    with open(processed_dir + out_file, 'w') as fw:     
        with gzip.open(raw_genome_dir + input_genome, 'rt') as f:
            for record in SeqIO.parse(f, "fasta"):
                for i in range(0, len(record.seq), max_seq_length):
                    if i + max_seq_length > len(record.seq):
                        break
                    sub_seq = str(record.seq[i:i+max_seq_length]).upper()

                    # convert non-ATCGN characters to N
                    sub_seq = re.sub("[^AGCTN]", "N", sub_seq)
                    # check the ratio of Ns in the sub_seq
                    if float(sub_seq.count("N")/len(sub_seq)) > n_ratio:
                        removed_seq_num += 1
                        continue
                    
                    seq_num += 1
                    fw.write(sub_seq + "\n")
    return seq_num, removed_seq_num

if __name__ == "__main__":
    info_log = open(log_dir + "split_genome_info.log", 'w')
    for file in os.listdir(raw_genome_dir):
        if not file.endswith(".fna.gz"):
            continue
        seq_num, removed_seq_num = split_genome(file)
        print("Removed " + str(removed_seq_num) + " seqs.")
        print("Total seqs: " + str(seq_num))
        info_log.write(file + "\t" + str(seq_num) + "\t" + str(removed_seq_num) + "\n")
    info_log.close()
    print("Done.")
