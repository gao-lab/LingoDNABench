# This script removes duplicate sequences from all .seq files in the specified directory.
import os
from hashlib import md5

raw_dir = "./processed_4096/"
# human dataset and multispecies dataset
human_out_file = raw_dir + "human.4k.seq"
ms_out_file = raw_dir + "multispecies.4k.seq"


print("Removing duplicates from the sequence file...")
seen_seqs = set()

with open(ms_out_file, "w") as out:
    for file in os.listdir(raw_dir):
        if file.endswith(".seq"):
            raw_seq_file = os.path.join(raw_dir, file)
            with open(raw_seq_file, "r") as f:
                for line in f:
                    h = md5(line.encode()).hexdigest()
                    if h not in seen_seqs:
                        out.write(line)
                        seen_seqs.add(h)

seen_seqs_human = set()
with open(human_out_file, "w") as out:
    raw_seq_file = os.path.join(raw_dir, "GCA_009914755.4_T2T-CHM13v2.0_genomic_4096.seq")
    with open(raw_seq_file, "r") as f:
        for line in f:
            h = md5(line.encode()).hexdigest()
            if h not in seen_seqs_human:
                out.write(line)
                seen_seqs_human.add(h)
print("Done!")