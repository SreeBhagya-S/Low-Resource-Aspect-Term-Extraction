#!/usr/bin/env python3
"""Reproducible, post-hoc paired-review bootstrap for a frozen ATE component ablation.

Run from IDT_R3\\R3 on Windows:
py -3.12 .\\12_paired_ablation_bootstrap.py --manifest "..\\GitHub_R2\\data\\fixed_test_manifest_seed42.csv" --ablation-dir ".\\R3_component_ablation_fixed200" --outdir ".\\R3_component_ablation_fixed200\\bootstrap"

For each of 10,000 resamples of the same 200 reviews, evaluate full pipeline
minus ablated configuration via strict span + category IOB2 Micro-F1. This is a
retrospective sensitivity analysis, not a prospective confirmatory test. Does
NOT change models, labels, predictions, or component configurations.
"""
import argparse
import ast
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


def cells_list(value):
    x = ast.literal_eval(str(value))
    if not isinstance(x, list):
        raise ValueError('Cell does not contain a list')
    return x


def tags(cell):
    x = cells_list(cell)
    if x and isinstance(x[0], (list, tuple)):
        return [str(p[1]) for p in x]
    return [str(t) for t in x]


def strict_spans(seq):
    spans = set()
    start, cat = None, None
    for i, tag in enumerate([*seq, 'O']):
        if tag == 'O':
            if start is not None:
                spans.add((start, i, cat))
                start, cat = None, None
        elif tag.startswith('B-'):
            if start is not None:
                spans.add((start, i, cat))
            start, cat = i, tag[2:]
        elif tag.startswith('I-'):
            if start is None or cat != tag[2:]:
                raise ValueError(f'Invalid IOB2 continuation {tag} at position {i}')
        else:
            raise ValueError(f'Unknown tag: {tag}')
    return spans


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contributions(predfile, manifest):
    pred = pd.read_csv(predfile)
    if len(pred) != len(manifest):
        raise ValueError(f'Count mismatch: {predfile}')
    rows = np.zeros((len(pred), 3), dtype=np.int32)
    for i in range(len(pred)):
        row = pred.iloc[i]
        m = manifest.iloc[i]
        gold_tokens = cells_list(m['Word_Tokens_Evaluated'])
        predicted_tokens = cells_list(row['Word_Tokens'])
        pred_tags = tags(row['BIO_Tags'])
        true_tags = tags(m['Gold_BIO'])
        if gold_tokens != predicted_tokens[:len(gold_tokens)]:
            raise ValueError(f'Prediction token mismatch {predfile}, row {i}')
        if len(true_tags) != len(gold_tokens) or len(pred_tags) < len(gold_tokens):
            raise ValueError(f'Token/tag length mismatch {predfile}, row {i}')
        a = strict_spans(true_tags)
        b = strict_spans(pred_tags[:len(gold_tokens)])
        rows[i] = (len(a & b), len(b - a), len(a - b))
    return rows


def micro_f1(counts):
    # accepts counts shape (..., 3)
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--ablation-dir', required=True, type=Path)
    parser.add_argument('--outdir', required=True, type=Path)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 200:
        raise ValueError('Expected original 200-review test; stop and audit manifest')
    summary_path = args.ablation_dir / 'component_ablation.csv'
    summary = pd.read_csv(summary_path)
    if len(summary) != 7 or summary.iloc[0]['Configuration'] != 'Full pipeline':
        raise ValueError('Expected seven recorded configurations beginning with Full pipeline')
    counts = []
    file_hashes = {}
    for i in range(7):
        path = args.ablation_dir / f'prediction_{i:02d}.csv'
        counts.append(contributions(path, manifest))
        file_hashes[path.name] = sha(path)
    observed = []
    for i, counts_i in enumerate(counts):
        f1 = float(micro_f1(counts_i.sum(axis=0)))
        claimed = float(summary.iloc[i]['F1'])
        if abs(f1 - claimed) > 1e-10:
            raise ValueError(f'Recalculated F1 differs for configuration {i}: {f1} vs {claimed}')
        observed.append(f1)
    index = np.random.default_rng(args.seed).integers(0, 200, size=(args.resamples, 200))
    full = micro_f1(counts[0][index].sum(axis=1))
    rows = []
    for i in range(1, 7):
        ablated = micro_f1(counts[i][index].sum(axis=1))
        delta = full - ablated
        low, high = np.quantile(delta, [.025, .975], method='linear')
        rows.append({
            'Configuration': str(summary.iloc[i]['Configuration']),
            'Full_F1': observed[0], 'Ablated_F1': observed[i],
            'Full_minus_Ablated_F1': observed[0] - observed[i],
            'Paired_Bootstrap_CI_Lower': float(low),
            'Paired_Bootstrap_CI_Upper': float(high),
        })
    args.outdir.mkdir(parents=True, exist_ok=True)
    dest = args.outdir / 'paired_ablation_bootstrap_10000.csv'
    pd.DataFrame(rows).to_csv(dest, index=False)
    info = {
        'purpose': 'Exploratory post-hoc paired whole-review bootstrap for each full-vs-component-off difference',
        'test_reviews': 200, 'gold_spans': int(counts[0][:, 0].sum()+counts[0][:, 2].sum()),
        'resamples': args.resamples, 'seed': args.seed,
        'RNG': 'numpy.random.default_rng',
        'CI': '2.5th and 97.5th percentiles, numpy linear quantile',
        'metric': 'strict span-and-category Micro-F1, IOB2',
        'difference': 'F1_full - F1_component_off',
        'manifest_sha256': sha(args.manifest),
        'component_table_sha256': sha(summary_path),
        'prediction_sha256': file_hashes,
        'numpy': version('numpy'), 'pandas': version('pandas'),
        'warning': 'Post-hoc exploratory analysis on an already examined test set; no multiplicity adjustment. Do not use intervals to tune the pipeline.'
    }
    (args.outdir/'paired_ablation_bootstrap_manifest.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print('Saved:', dest)
    print('Saved:', args.outdir/'paired_ablation_bootstrap_manifest.json')

if __name__ == '__main__':
    main()
