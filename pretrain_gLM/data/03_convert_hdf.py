import h5py
import numpy as np


data_dir = "./processed_4096/"
seq_length = 4096
input_file = data_dir + "all.4k.seq"
output_file = data_dir + "multi-species." + str(seq_length) + ".h5"
chunk_size = 10000

def process_chunk(chunk):
    chunk = chunk.astype("S" + str(seq_length))
    return chunk

with h5py.File(output_file, "w") as f:
    with open(input_file) as file:
        file.seek(0)
        total_lines = sum(1 for line in file)
        file.seek(0)

        dset = f.create_dataset("seq", (total_lines, ), dtype = "S" + str(seq_length))

        chunk = []

        for i, line in enumerate(file):
            chunk.append(line.strip())
            if (i + 1) % chunk_size == 0 or (i + 1) == total_lines:
                dset[i + 1 - len(chunk):i + 1] = process_chunk(np.array(chunk))
                chunk = []

print("Done!")