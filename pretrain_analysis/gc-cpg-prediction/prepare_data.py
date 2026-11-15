import argparse
import csv
from pathlib import Path

import numpy as np
import pysam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-fasta', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data/windows.tsv')
    parser.add_argument('--count', type=int, default=5000)
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.count < 1 or args.length < 3:
        parser.error('--count must be positive and --length must be at least 3')

    rng = np.random.default_rng(args.seed)
    chromosomes = [f'chr{index}' for index in range(1, 23)]
    occupied = {chrom: set() for chrom in chromosomes}
    seen = set()
    rows = []
    with pysam.FastaFile(str(args.reference_fasta)) as fasta:
        sizes = np.array([fasta.get_reference_length(chrom) for chrom in chromosomes])
        for _ in range(args.count * 30):
            chrom = str(rng.choice(chromosomes, p=sizes / sizes.sum()))
            cell = int(rng.integers(0, fasta.get_reference_length(chrom) // args.length))
            if any(index in occupied[chrom] for index in (cell - 1, cell, cell + 1)):
                continue
            start = cell * args.length
            sequence = fasta.fetch(chrom, start, start + args.length).upper()
            if len(sequence) != args.length or set(sequence) - set('ACGT') or sequence in seen:
                continue
            occupied[chrom].add(cell)
            seen.add(sequence)
            number = int(chrom[3:])
            split = 'train' if number <= 16 else 'valid' if number <= 19 else 'test'
            rows.append(dict(chrom=chrom, start=start, end=start + args.length, split=split,
                             gc=(sequence.count('G') + sequence.count('C')) / args.length,
                             cpg=sequence.count('CG') / (args.length - 1), sequence=sequence))
            if len(rows) == args.count:
                break
    if len(rows) != args.count:
        raise ValueError('Not enough distinct DNA windows')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['chrom', 'start', 'end', 'split', 'gc', 'cpg', 'sequence'],
                                delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} windows to {args.output}')


if __name__ == '__main__':
    main()
