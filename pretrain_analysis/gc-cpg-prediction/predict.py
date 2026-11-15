import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'data/windows.tsv')
    parser.add_argument('--features', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ridge-dtype', choices=['float64', 'float32'])
    args = parser.parse_args()

    with args.input.open() as handle:
        windows = list(csv.DictReader(handle, delimiter='\t'))
    split = np.array([row['split'] for row in windows])
    train, valid, test = [split == name for name in ('train', 'valid', 'test')]
    if any(mask.sum() < 2 for mask in (train, valid, test)):
        raise ValueError('Each split must have at least two windows')
    targets = np.array([[float(row['gc']), float(row['cpg'])] for row in windows])
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    test_indices = np.flatnonzero(test)
    metrics = []
    predictions = []
    sources = []
    for directory in args.features:
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['windows_sha256'] != digest or metadata['windows'] != len(windows):
            raise ValueError(f'Features were extracted from different windows: {directory}')
        expected = 'benchmark-layer' if metadata['model_type'] == 'evo2' else 'last-layer'
        if metadata.get('representation') != expected:
            raise ValueError(f'Unexpected representation: {directory}')
        if metadata['model_type'] == 'evo2' and metadata['layer'] != 28:
            raise ValueError(f'Expected Evo2 benchmark layer 28: {directory}')
        dtype = args.ridge_dtype or ('float32' if metadata['model_type'] == 'evo2' else 'float64')
        identity = dict(model_type=metadata['model_type'], state=metadata['state'], layer=metadata['layer'])
        sources.append(dict(path=str(directory.resolve()), ridge_dtype=dtype, **identity))
        values = np.load(directory / 'base-mean.npy').astype(dtype)
        if values.ndim != 2 or values.shape[0] != len(windows) or not np.isfinite(values).all():
            raise ValueError(f'Invalid feature matrix: {directory}')
        scaler = StandardScaler().fit(values[train])
        x_train, x_valid, x_test = [scaler.transform(values[mask]) for mask in (train, valid, test)]
        best_predictions = np.empty((len(test_indices), 2))
        best_alphas = np.zeros(2)
        if dtype == 'float64':
            best_errors = np.full(2, np.inf)
            for alpha in (0.01, 1., 100., 10000.):
                ridge = Ridge(alpha=alpha).fit(x_train, targets[train])
                errors = np.mean((ridge.predict(x_valid) - targets[valid]) ** 2, axis=0)
                improved = errors < best_errors
                best_predictions[:, improved] = ridge.predict(x_test)[:, improved]
                best_alphas[improved] = alpha
                best_errors[improved] = errors[improved]
        else:
            for column in range(2):
                best_error = np.inf
                for alpha in (0.01, 1., 100., 10000.):
                    ridge = Ridge(alpha=alpha).fit(x_train, targets[train, column])
                    error = np.mean((ridge.predict(x_valid) - targets[valid, column]) ** 2)
                    if error < best_error:
                        best_error = error
                        best_alphas[column] = alpha
                        best_predictions[:, column] = ridge.predict(x_test)
        for column, target in enumerate(('GC', 'CpG')):
            predicted = best_predictions[:, column]
            row = dict(**identity, target=target, alpha=float(best_alphas[column]),
                       spearman_r=float(spearmanr(targets[test, column], predicted)[0]),
                       r2=float(r2_score(targets[test, column], predicted)),
                       mae=float(mean_absolute_error(targets[test, column], predicted)),
                       test_windows=len(test_indices))
            metrics.append(row)
            for index, value in zip(test_indices, predicted):
                window = windows[index]
                predictions.append(dict(**identity, target=target, window_index=int(index),
                                        chrom=window['chrom'], start=window['start'], end=window['end'],
                                        observed=float(targets[index, column]), predicted=float(value)))
            print(metadata['model_type'], metadata['state'], target, row['spearman_r'], flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('metrics.tsv', metrics), ('predictions.tsv', predictions)]:
        with (args.output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    report = dict(windows_sha256=digest, windows=len(windows),
                  splits={name: int(np.sum(split == name)) for name in ('train', 'valid', 'test')},
                  alpha_grid=[0.01, 1., 100., 10000.], sources=sources)
    (args.output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
