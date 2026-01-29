import os
import sys
import numpy as np
from Bio import SeqIO

feature_name = sys.argv[1]
data_dir = sys.argv[2]
split_dir = sys.argv[3]
num_targets = int(sys.argv[4])
split_size = int(sys.argv[5])

os.makedirs(split_dir, exist_ok=True)

def save_chunk(labels_chunk, n_rows, save_dir, elem, feature_name, chunk_idx):
    arr = labels_chunk[:n_rows]
    np.save(os.path.join(save_dir, f"{feature_name}_{elem}_labels_{chunk_idx}.npy"), arr)

def open_chunk_seq(save_dir, elem, feature_name, chunk_idx):
    path = os.path.join(save_dir, f"{feature_name}_{elem}_{chunk_idx}.seq")
    return open(path, "w")

for elem in ["train", "valid", "test"]:
    fasta_path = os.path.join(data_dir, f"{feature_name}_{elem}.fasta")
    full_seq_path = os.path.join(data_dir, f"{feature_name}_{elem}.seq")

    print(f"[{elem}] processing + streaming split...")

    with open(full_seq_path, "w") as full_seq_f:
        chunk_idx = 0
        in_chunk = 0

        labels_chunk = np.zeros((split_size, num_targets), dtype=np.uint8)
        chunk_seq_f = open_chunk_seq(split_dir, elem, feature_name, chunk_idx)

        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()


            full_seq_f.write(seq + "\n")

            chunk_seq_f.write(seq + "\n")

            left = str(record.id).split("::")[0]
            if left:
                for t in left.split(","):
                    t = t.strip()
                    if not t:
                        continue

                    idx = int(t)
                    if 0 <= idx < num_targets:
                        labels_chunk[in_chunk, idx] = 1
            in_chunk += 1
            if in_chunk == split_size:
                save_chunk(labels_chunk, in_chunk, split_dir, elem, feature_name, chunk_idx)
                chunk_seq_f.close()

                chunk_idx += 1
                in_chunk = 0
                labels_chunk.fill(0)
                chunk_seq_f = open_chunk_seq(split_dir, elem, feature_name, chunk_idx)

        if in_chunk > 0:
            save_chunk(labels_chunk, in_chunk, split_dir, elem, feature_name, chunk_idx)
        chunk_seq_f.close()

    print(f"[{elem}] done. chunks={chunk_idx + (1 if in_chunk > 0 else 0)}")
