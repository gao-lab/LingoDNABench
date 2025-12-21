# generate random sequence for training and testing
import random
import numpy as np
import tqdm


# set parameters
seq_len = 4096 # length of each sequence
num_seq = 698_183 # number of sequences to generate
output_file = "random_seq.txt"

# generate random sequence for training and testing
def generate_random_seq(seq_len, num_seq):
    seq_list = []
    print("Generating " + str(num_seq) + " random sequences.")
    for i in tqdm.tqdm(range(num_seq)):
        seq = ''.join(random.choices('ACGT', k=seq_len))
        seq_list.append(seq)
    seq_list = np.array(seq_list)
    np.savetxt(output_file, seq_list, fmt='%s')
    print("Done!")

if __name__ == '__main__':
    generate_random_seq(seq_len, num_seq)