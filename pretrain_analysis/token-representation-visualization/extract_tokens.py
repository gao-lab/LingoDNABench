import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmark'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gc-cpg-prediction'))
from matched_rw import load_model, tokenize_sequence, extract_sequence
from extract_features import load_pretrained


def load_pretrained_model(args):
    if args.model_type not in ('caduceus', 'deepgene'):
        return load_pretrained(args)

    if args.model_type == 'caduceus':
        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        sys.path.insert(0, str(args.checkpoint))
        from tokenization_caduceus import CaduceusTokenizer
        tokenizer = CaduceusTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    else:
        code = args.code_path or args.random_weights_dir / 'deepgene/code'
        sys.path.insert(0, str(code))
        from configuration_roformer import RoFormerConfig
        from modeling_roformer import RoFormerForMaskedLM
        from tokenizers import Tokenizer
        config = RoFormerConfig.from_pretrained(args.checkpoint, local_files_only=True)
        model = RoFormerForMaskedLM(config)
        tokenizer_path = args.tokenizer_path or args.random_weights_dir / 'deepgene/tokenizer.json'
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (args.checkpoint / 'model.safetensors').exists():
        from safetensors.torch import load_file
        source = load_file(str(args.checkpoint / 'model.safetensors'))
    else:
        source = torch.load(args.checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    own = model.state_dict(keep_vars=True)
    state = {}
    for key, value in source.items():
        target = key.removeprefix('caduceus.') if args.model_type == 'caduceus' else key
        if target in own:
            state[target] = value
        elif not (args.model_type == 'caduceus' and key.startswith('lm_head.')):
            raise ValueError(f'Unexpected pretrained tensor: {key}')
    aliases = defaultdict(list)
    for key, value in own.items():
        aliases[id(value)].append(key)
    for keys in aliases.values():
        present = [key for key in keys if key in state]
        if present:
            value = state[present[0]]
            if not all(torch.equal(state[key], value) for key in present):
                raise ValueError('Conflicting shared pretrained weights')
            for key in keys:
                state[key] = value
    if args.model_type == 'deepgene' and 'roformer.encoder.embed_positions.weight' not in state:
        state['roformer.encoder.embed_positions.weight'] = own['roformer.encoder.embed_positions.weight']
    model.load_state_dict(state, strict=True)
    return model, tokenizer


def select_tokens(kind, tokenizer, sequences):
    if kind == 'evo2':
        vocabulary = {ord(base): base for base in 'ACGT'}
    elif kind == 'lucaone':
        vocabulary = {tokenizer.get_idx(token): base for token, base in zip('1234', 'ATCG')}
    else:
        excluded = {1, 2, 3} if kind == 'deepgene' else set(tokenizer.all_special_ids)
        vocabulary = {index: token for token, index in tokenizer.get_vocab().items()
                      if token and set(token) <= set('ACGT') and index not in excluded}
        if kind in ('nt', 'GEN', 'dnabert'):
            width = 3 if kind == 'dnabert' else 6
            vocabulary = {index: token for index, token in vocabulary.items() if len(token) == width}

    records = []
    for sequence in sequences:
        dna = sequence['sequence']
        _, ids, _ = tokenize_sequence(kind, tokenizer, dna)
        if kind == 'dnabert':
            spans = [(index, index + 3) for index in range(len(dna) - 2)]
        else:
            if kind == 'evo2':
                tokens = list(dna)
            elif kind == 'lucaone':
                tokens = [tokenizer.get_tok(index).translate(str.maketrans('1234', 'ATCG')) for index in ids]
            elif kind == 'deepgene':
                tokens = [tokenizer.id_to_token(index) for index in ids]
            else:
                tokens = tokenizer.convert_ids_to_tokens(ids)
                if kind == 'omnina' and tokens[0].startswith('▁'):
                    tokens[0] = tokens[0][1:]
            if ''.join(tokens) != dna:
                raise ValueError('DNA tokens do not reconstruct the sequence')
            ends = np.cumsum([len(token) for token in tokens])
            spans = list(zip(np.r_[0, ends[:-1]], ends))
        if len(ids) != len(spans):
            raise ValueError('Token coordinates and IDs differ')
        positions = defaultdict(list)
        for position, index in enumerate(ids):
            if index in vocabulary and spans[position][1] > spans[position][0]:
                positions[index].append(position)
        for index in sorted(positions, key=lambda index: vocabulary[index]):
            token = vocabulary[index]
            key = f"20261105|{sequence['sequence_id']}|{token}".encode()
            position = positions[index][int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % len(positions[index])]
            start, end = spans[position]
            if dna[start:end] != token:
                raise ValueError('Token string differs from the genomic sequence')
            records.append(dict(sequence_id=sequence['sequence_id'], token_id=int(index), token=token,
                                token_index=position, base_start=int(start), base_end=int(end),
                                token_gc=(token.count('G') + token.count('C')) / len(token)))
    ordered = sorted(vocabulary, key=lambda index: vocabulary[index])
    selected = [ordered[index] for index in np.linspace(0, len(ordered) - 1, min(8, len(ordered)), dtype=int)]
    colors = [dict(token_id=int(index), token=vocabulary[index]) for index in selected]
    return records, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True, choices=[
        'dnabert', 'dnabert2', 'nt', 'caduceus', 'hyenadna', 'omnina', 'deepgene', 'lucaone', 'GEN', 'evo2'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--random-weights', action='store_true')
    parser.add_argument('--random-weights-dir', type=Path,
                        default=Path('matched-RW'))
    parser.add_argument('--code-path', type=Path)
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'data/sequences.tsv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not args.random_weights and args.checkpoint is None:
        parser.error('--checkpoint is required for pretrained models')
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.model_type == 'evo2' and not args.device.startswith('cuda'):
        parser.error('Evo2 requires CUDA')
    with args.input.open() as handle:
        sequences = list(csv.DictReader(handle, delimiter='\t'))
    if not sequences or len({row['sequence_id'] for row in sequences}) != len(sequences):
        raise ValueError('Input must contain distinct sequence IDs')
    if any(not row['sequence'] or set(row['sequence']) - set('ACGT') for row in sequences):
        raise ValueError('Input must contain A/C/G/T sequences')

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.random_weights:
        checkpoint = args.random_weights_dir / args.model_type
        model, tokenizer = load_model(args.model_type, checkpoint, args.device)
    else:
        checkpoint = args.checkpoint
        model, tokenizer = load_pretrained_model(args)
    if args.model_type != 'evo2':
        model.float()
    model.to(args.device).eval().requires_grad_(False)
    records, colors = select_tokens(args.model_type, tokenizer, sequences)
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record['sequence_id']].append(index)
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / 'representations.partial.npy'
    values = None
    try:
        for count, sequence in enumerate(sequences):
            hidden = extract_sequence(args.model_type, model, tokenizer, sequence['sequence'], -1, args.device)
            if values is None:
                values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32',
                                                  shape=(len(records), hidden.shape[1]))
            indices = groups[sequence['sequence_id']]
            positions = [records[index]['token_index'] for index in indices]
            values[indices] = hidden[positions]
            if (count + 1) % 50 == 0 or count + 1 == len(sequences):
                print(f'{count + 1}/{len(sequences)}', flush=True)
        values.flush()
        width = values.shape[1]
        del values
        temporary.replace(args.output / 'representations.npy')
    finally:
        if temporary.exists():
            temporary.unlink()
    with (args.output / 'tokens.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    layer = 28 if args.model_type == 'evo2' else (model.config.n_layer if args.model_type in (
        'caduceus', 'hyenadna') else model.config.num_hidden_layers)
    metadata = dict(model_type=args.model_type, state='random' if args.random_weights else 'pretrained',
                    checkpoint=str(checkpoint.resolve()), layer=layer, sequences=len(sequences),
                    input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    tokens_sha256=hashlib.sha256((args.output / 'tokens.tsv').read_bytes()).hexdigest(),
                    occurrences=len(records), hidden_width=width, identity_colors=colors,
                    seed=666 if args.random_weights else None)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
