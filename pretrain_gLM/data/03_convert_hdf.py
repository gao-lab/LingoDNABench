# This script converts the processed sequence files into HDF5 format for efficient storage and retrieval.
import h5py
import numpy as np


data_dir = "./processed_4096/"
dataset_dir = "./dataset/"
os.makedirs(dataset_dir, exist_ok=True)

seq_length = 4096
# human dataset
human_input_file = data_dir + "human.4k.seq"
human_output_file = dataset_dir + "human." + str(seq_length) + ".h5"

# multispecies dataset
multi_input_file = data_dir + "multi-species.4k.seq"
multi_output_file = dataset_dir + "ms." + str(seq_length) + ".h5"

# chunk size for processing
chunk_size = 10000

def process_chunk(chunk):
    chunk = chunk.astype("S" + str(seq_length))
    return chunk

def construct_dataset(input_file, output_file):
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

if __name__ == "__main__":
    print("Converting human dataset to HDF5 format...")
    construct_dataset(human_input_file, human_output_file)
    print("Human dataset conversion complete.")

    print("Converting multispecies dataset to HDF5 format...")
    construct_dataset(multi_input_file, multi_output_file)
    print("Multispecies dataset conversion complete.")
