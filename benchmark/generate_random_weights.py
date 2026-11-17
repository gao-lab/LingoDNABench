import argparse
import json
from pathlib import Path
import shutil

import torch

from matched_rw import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True, choices=[
        'dnabert', 'dnabert2', 'nt', 'caduceus', 'hyenadna',
        'omnina', 'deepgene', 'lucaone', 'GEN', 'evo2'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--code-path', type=Path)
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--output', type=Path, default=Path('matched-RW'))
    args = parser.parse_args()
    kind = args.model_type
    if kind != 'evo2' and args.checkpoint is None:
        parser.error('--checkpoint is required for this model')
    if kind == 'evo2' and not torch.cuda.is_available():
        parser.error('Evo2 requires CUDA')

    output = args.output / kind
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    if kind == 'evo2':
        import importlib.util
        import pkgutil
        from evo2.utils import CONFIG_MAP

        if importlib.util.find_spec('transformer_engine') is not None:
            raise RuntimeError('Use the Evo2 environment without Transformer Engine')
        output.mkdir(parents=True, exist_ok=True)
        config = pkgutil.get_data('evo2', CONFIG_MAP['evo2_7b'])
        (output / 'config.yml').write_bytes(config)
    else:
        source = args.checkpoint
        if not (source / 'config.json').is_file():
            raise FileNotFoundError(source / 'config.json')
        code = args.code_path or source / 'code'
        tokenizer = args.tokenizer_path or source / 'tokenizer.json'
        if kind == 'deepgene':
            for name in ('configuration_roformer.py', 'modeling_roformer.py'):
                if not (code / name).is_file():
                    raise FileNotFoundError(code / name)
            if not tokenizer.is_file():
                raise FileNotFoundError(tokenizer)
        if kind == 'lucaone' and not (code / 'algorithms').is_dir():
            raise FileNotFoundError(code / 'algorithms')
        output.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if not path.is_file() or path.name.endswith('.index.json'):
                continue
            if path.suffix in ('.py', '.json', '.txt', '.model', '.vocab', '.merges') or path.name in ('LICENSE', 'NOTICE'):
                if path.name != 'initialization.json':
                    shutil.copy2(path, output / path.name)
        if kind == 'deepgene':
            (output / 'code').mkdir()
            for name in ('configuration_roformer.py', 'modeling_roformer.py'):
                shutil.copy2(code / name, output / 'code' / name)
            shutil.copy2(tokenizer, output / 'tokenizer.json')
        elif kind == 'lucaone':
            shutil.copytree(code / 'algorithms', output / 'code/algorithms',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    device = 'cuda' if kind == 'evo2' else 'cpu'
    model, tokenizer = load_model(kind, output, device, initialize=True)
    if kind == 'dnabert':
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
    else:
        state = {name: value.cpu() for name, value in model.state_dict().items()}
        torch.save(state, output / 'random-weights.pt')
    metadata = dict(model_type=kind, seed=666, pretrained_weights_loaded=False,
                    parameters=sum(value.numel() for value in model.parameters()),
                    torch_version=torch.__version__)
    if kind != 'evo2':
        import transformers
        metadata['transformers_version'] = transformers.__version__
    else:
        metadata['initializer'] = 'normal(0,0.02); norm=1; bias=0; log_poles=uniform(-1,-0.1)'
    (output / 'initialization.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved seed 666 random weights: {output}')


if __name__ == '__main__':
    main()
