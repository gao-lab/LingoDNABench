import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmark'))
from matched_rw import load_model, tokenize_sequence, extract_sequence


def load_pretrained(args):
    kind = args.model_type
    path = args.checkpoint
    code = args.code_path or args.random_weights_dir / kind / 'code'
    if kind == 'evo2':
        import evo2
        from evo2 import Evo2
        from vortex.model.tokenizer import CharLevelTokenizer

        previous = Path.cwd()
        try:
            os.chdir(Path(evo2.__file__).resolve().parent)
            model = Evo2('evo2_7b', local_path=str(path.resolve())).model
        finally:
            os.chdir(previous)
        tokenizer = CharLevelTokenizer(512)
    elif kind == 'deepgene':
        sys.path.insert(0, str(code))
        from modeling_roformer import RoFormerForMaskedLM
        from tokenizers import Tokenizer

        model = RoFormerForMaskedLM.from_pretrained(path, local_files_only=True)
        tokenizer_path = args.tokenizer_path or args.random_weights_dir / kind / 'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    elif kind == 'lucaone':
        sys.path.insert(0, str(code))
        from algorithms.llm.lucagplm.v2_0.lucaone_gplm import LucaGPLM
        from algorithms.llm.lucagplm.v2_0.lucaone_gplm_config import LucaGPLMConfig

        config = LucaGPLMConfig.from_pretrained(path, local_files_only=True)
        model = LucaGPLM(config, args=SimpleNamespace(pretrained_model_name=None, embedding_inference=True))
        state = torch.load(path / 'pytorch.pth', map_location='cpu', weights_only=True)
        own = model.state_dict()
        state = {key.removeprefix('module.'): value for key, value in state.items()
                 if key.removeprefix('module.') in own}
        model.load_state_dict(state, strict=True)
        tokenizer = model.alphabet
    else:
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        if kind == 'caduceus':
            model = AutoModelForMaskedLM.from_pretrained(path, trust_remote_code=True,
                                                       local_files_only=True).caduceus
        else:
            model = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        if kind in ('caduceus', 'hyenadna'):
            sys.path.insert(0, str(path))
            if kind == 'caduceus':
                from tokenization_caduceus import CaduceusTokenizer
                tokenizer = CaduceusTokenizer.from_pretrained(path, local_files_only=True)
            else:
                from tokenization_hyena import HyenaDNATokenizer
                tokenizer = HyenaDNATokenizer.from_pretrained(path, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    return model, tokenizer


def token_alignment(kind, tokenizer, sequence):
    _, ids, _ = tokenize_sequence(kind, tokenizer, sequence)
    if kind == 'dnabert':
        intervals = [(index, index + 3) for index in range(len(sequence) - 2)]
    else:
        if kind == 'lucaone':
            tokens = [tokenizer.get_tok(index).translate(str.maketrans('1234', 'ATCG')) for index in ids]
        elif kind == 'deepgene':
            tokens = [tokenizer.id_to_token(index) for index in ids]
        elif kind == 'evo2':
            tokens = list(sequence)
        else:
            tokens = tokenizer.convert_ids_to_tokens(ids)
            if kind == 'omnina' and tokens[0].startswith('▁'):
                tokens[0] = tokens[0][1:]
        if ''.join(tokens) != sequence:
            raise ValueError('DNA tokens do not reconstruct the input sequence')
        ends = np.cumsum([len(token) for token in tokens])
        intervals = list(zip(np.r_[0, ends[:-1]], ends))
    if len(ids) != len(intervals):
        raise ValueError('Token count differs from the coordinate count')
    coverage = np.zeros(len(sequence), dtype=np.float64)
    for start, end in intervals:
        coverage[start:end] += 1
    if not np.all(coverage > 0):
        raise ValueError('DNA tokens do not cover the full sequence')
    weights = np.array([np.sum(1 / coverage[start:end]) / len(sequence) for start, end in intervals])
    return ids, weights


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
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'data/windows.tsv')
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
        windows = list(csv.DictReader(handle, delimiter='\t'))
    if not windows:
        raise ValueError('Input contains no windows')
    for row in windows:
        sequence = row['sequence']
        if len(sequence) != int(row['end']) - int(row['start']) or set(sequence) - set('ACGT'):
            raise ValueError('Windows must contain reference-forward A/C/G/T sequences')

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.random_weights:
        path = args.random_weights_dir / args.model_type
        model, tokenizer = load_model(args.model_type, path, args.device)
    else:
        path = args.checkpoint
        model, tokenizer = load_pretrained(args)
    if args.model_type != 'evo2':
        model.float()
    model.to(args.device).eval().requires_grad_(False)

    if args.model_type == 'evo2':
        layer = 28
    elif args.model_type in ('caduceus', 'hyenadna'):
        layer = model.config.n_layer
    else:
        layer = model.config.num_hidden_layers

    features = []
    with torch.inference_mode():
        for index, row in enumerate(windows):
            sequence = row['sequence']
            ids, weights = token_alignment(args.model_type, tokenizer, sequence)
            hidden = extract_sequence(args.model_type, model, tokenizer, sequence, -1, args.device)
            if hidden.shape[0] != len(ids):
                raise ValueError('Token coordinates and feature rows differ')
            features.append(weights @ hidden)
            if (index + 1) % 100 == 0 or index + 1 == len(windows):
                print(f'{index + 1}/{len(windows)}', flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    values = np.asarray(features, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite features')
    np.save(args.output / 'base-mean.npy', values)
    metadata = dict(model_type=args.model_type, state='random' if args.random_weights else 'pretrained',
                    checkpoint=str(path.resolve()), layer=layer,
                    representation='benchmark-layer' if args.model_type == 'evo2' else 'last-layer',
                    windows=len(windows),
                    windows_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    hidden_width=values.shape[1], dtype='float32', tf32=False,
                    seed=666 if args.random_weights else None)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved features to {args.output}')


if __name__ == '__main__':
    main()
