# generate random sequence for training and testing
import random
import numpy as np
import tqdm
import h5py


# set parameters
seq_len = 4096 # length of each sequence
num_seq = 698_183 # number of sequences to generate
output_file = "./dataset/random." + str(seq_len) + ".h5"

# generate random sequence for training and testing
def generate_random_seq(seq_len, num_seq):
    seq_list = []
    print("Generating " + str(num_seq) + " random sequences.")
    for i in tqdm.tqdm(range(num_seq)):
        seq = ''.join(random.choices('ACGT', k=seq_len))
        seq_list.append(seq)
    seq_list = np.array(seq_list)
    #contruct hdf5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('seq', data=seq_list, dtype="S" + str(seq_len))
    print("Done!")

if __name__ == '__main__':
    generate_random_seq(seq_len, num_seq)