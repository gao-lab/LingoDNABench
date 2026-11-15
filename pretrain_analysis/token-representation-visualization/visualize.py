import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    panels = {}
    models = []
    input_digest = None
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in args.features:
        metadata = json.loads((directory / 'metadata.json').read_text())
        kind, state = metadata['model_type'], metadata['state']
        if state not in ('random', 'pretrained') or (kind, state) in panels:
            raise ValueError('Provide one random and one pretrained directory per model')
        if input_digest is None:
            input_digest = metadata['input_sha256']
        if metadata['input_sha256'] != input_digest:
            raise ValueError('All panels must use the same input sequences')
        if hashlib.sha256((directory / 'tokens.tsv').read_bytes()).hexdigest() != metadata['tokens_sha256']:
            raise ValueError('Token metadata has changed')
        with (directory / 'tokens.tsv').open() as handle:
            tokens = list(csv.DictReader(handle, delimiter='\t'))
        values = np.load(directory / 'representations.npy', mmap_mode='r')
        if values.shape != (len(tokens), metadata['hidden_width']) or not np.isfinite(values).all():
            raise ValueError('Invalid token representation matrix')
        coordinates_path = args.output / f'{kind}-{state}-umap.npy'
        cache_path = coordinates_path.with_suffix('.json')
        signature = dict(features=str(directory.resolve()), method='umap',
                         tokens_sha256=metadata['tokens_sha256'],
                         representations_size=(directory / 'representations.npy').stat().st_size,
                         representations_mtime_ns=(directory / 'representations.npy').stat().st_mtime_ns)
        if coordinates_path.exists() and cache_path.exists() and json.loads(cache_path.read_text()) == signature:
            coordinates = np.load(coordinates_path)
        else:
            from umap import UMAP
            coordinates = UMAP(metric='cosine', n_neighbors=15, min_dist=0.1,
                               random_state=42, n_jobs=1).fit_transform(values)
            np.save(coordinates_path, coordinates)
            cache_path.write_text(json.dumps(signature, indent=2) + '\n')
        panels[kind, state] = dict(metadata=metadata, tokens=tokens, coordinates=coordinates)
        if kind not in models:
            models.append(kind)
        print(kind, state, len(tokens), 'UMAP', flush=True)

    for kind in models:
        if any((kind, state) not in panels for state in ('random', 'pretrained')):
            raise ValueError(f'Provide both weight conditions for {kind}')
        random = panels[kind, 'random']['metadata']
        pretrained = panels[kind, 'pretrained']['metadata']
        if random['tokens_sha256'] != pretrained['tokens_sha256'] or random['identity_colors'] != pretrained['identity_colors']:
            raise ValueError('Random and pretrained panels must use the same token occurrences')

    models = [kind for kind in ('hyenadna', 'caduceus', 'deepgene', 'dnabert', 'dnabert2',
                                'omnina', 'lucaone', 'nt', 'GEN', 'evo2') if kind in models]
    display = {'nt': 'NT-2.5B-MS', 'GEN': 'GENERator-3B', 'dnabert': 'DNABERT-3mer',
               'dnabert2': 'DNABERT-2', 'caduceus': 'Caduceus-ps', 'hyenadna': 'HyenaDNA-1M',
               'omnina': 'OmniNA-220M', 'deepgene': 'DeepGene', 'lucaone': 'LucaOne', 'evo2': 'Evo2-7B'}
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 9})
    figure, axes = plt.subplots(4, len(models), figsize=(3.3 * len(models) + 1, 14), squeeze=False)
    for column, kind in enumerate(models):
        for row, state in enumerate(('random', 'pretrained')):
            panel = panels[kind, state]
            coordinates = panel['coordinates']
            tokens = panel['tokens']
            identities = np.array([int(token['token_id']) for token in tokens])
            shown = panel['metadata']['identity_colors']
            ax = axes[row, column]
            other = ~np.isin(identities, [token['token_id'] for token in shown])
            ax.scatter(*coordinates[other].T, s=3, c='#bcbcbc', alpha=0.22, edgecolors='none')
            for index, token in enumerate(shown):
                selected = identities == token['token_id']
                ax.scatter(*coordinates[selected].T, s=9, color=plt.get_cmap('tab10')(index),
                           alpha=0.8, edgecolors='none', label=token['token'])
            if row == 1:
                ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.015), ncol=2, frameon=False)
            values = np.array([float(token['token_gc']) for token in tokens])
            order = np.random.default_rng(42).permutation(len(tokens))
            axes[row + 2, column].scatter(*coordinates[order].T, s=5, c=values[order], cmap='viridis',
                                          norm=Normalize(0, 1), alpha=0.65, edgecolors='none')
        for row in range(4):
            ax = axes[row, column]
            if row in (0, 2):
                ax.set_title(display[kind])
            if column == 0:
                ax.set_ylabel('Random weights' if row % 2 == 0 else 'Pretrained')
            ax.set_xticks([])
            ax.set_yticks([])
    bar = figure.add_axes([0.37, 0.025, 0.26, 0.012])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='viridis'),
                              cax=bar, orientation='horizontal', label='Token GC fraction')
    colorbar.solids.set_rasterized(False)
    figure.subplots_adjust(left=0.12 if len(models) == 1 else 0.04, right=0.99,
                           top=0.96, bottom=0.08, hspace=0.4, wspace=0.15)
    for extension in ('png', 'pdf', 'svg'):
        figure.savefig(args.output / f'token-representations.{extension}', dpi=160, facecolor='white')
    plt.close(figure)


if __name__ == '__main__':
    main()
