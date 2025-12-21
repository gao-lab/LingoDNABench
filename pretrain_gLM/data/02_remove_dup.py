# remove duplicates from the sequence file
import os
from hashlib import md5

raw_dir = "./processed_4096/"
out_file = raw_dir + "all.4k.seq"

print("Removing duplicates from the sequence file...")
seen_seqs = set()

with open(out_file, "w") as out:
    for file in os.listdir(raw_dir):
        if file.endswith(".seq"):
            raw_seq_file = os.path.join(raw_dir, file)
            with open(raw_seq_file, "r") as f:
                for line in f:
                    h = md5(line.encode()).hexdigest()
                    if h not in seen_seqs:
                        out.write(line)
                        seen_seqs.add(h)
print("Done!")