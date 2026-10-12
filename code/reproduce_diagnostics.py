#!/usr/bin/env python3
"""Recompute appendix diagnostics from the distributed evaluator records.

Usage: python code/reproduce_diagnostics.py . --check
       python code/reproduce_diagnostics.py . --output-dir /tmp/appendix-results
"""
from __future__ import annotations
import argparse
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = ['idiom', 'metaphor', 'irony', 'sarcasm']
CONDITIONS = ['unsteered', 'monolingual_steering', 'main_crosslingual']


def prompt_key(frames):
    for key in ('simple_prompt', 'source_sentence', 'raw_prompt_text'):
        if all(key in frame for frame in frames):
            for frame in frames:
                if frame[key].dropna().duplicated().any():
                    raise ValueError(f'Duplicate prompt key: {key}')
            return key
    raise ValueError('No common prompt key')


def quality_tables(base):
    d = pd.read_csv(base / 'raw/quality_evaluated.csv.gz', low_memory=False)
    d = d[d.category.isin(CATEGORIES) & d.condition.isin(CONDITIONS)].copy()
    d['quality_score'] = pd.to_numeric(d.quality_score, errors='coerce')
    d['valid_quality'] = d.quality_score.isin(range(5))
    keys = ['condition', 'target_language', 'category']
    counts = d.groupby(keys).valid_quality.agg(evaluated_outputs='size', valid_scores='sum').reset_index()
    counts['invalid_or_missing_scores'] = counts.evaluated_outputs - counts.valid_scores
    cells = d[d.valid_quality].groupby(keys).quality_score.agg(n='size', mean_quality='mean').reset_index()
    means = cells.pivot(index=['target_language', 'category'], columns='condition', values='mean_quality').dropna(subset=CONDITIONS)
    cells['shared_cell'] = [(r.target_language, r.category) in means.index for r in cells.itertuples()]
    indices = np.random.default_rng(20260710).integers(0, len(means), (10000, len(means)))
    bootstrap, summary, differences = {}, [], []
    for condition in CONDITIONS:
        values = means[condition].to_numpy()
        bootstrap[condition] = values[indices].mean(axis=1)
        ci = np.percentile(bootstrap[condition], [2.5, 97.5])
        summary.append(dict(condition=condition, shared_cells=len(means), n=int(cells.loc[cells.shared_cell & cells.condition.eq(condition), 'n'].sum()), mean_quality=values.mean(), ci_lower=ci[0], ci_upper=ci[1]))
    for condition in CONDITIONS[1:]:
        ci = np.percentile(bootstrap[condition] - bootstrap['unsteered'], [2.5, 97.5])
        differences.append(dict(comparison=condition + '_minus_unsteered', shared_cells=len(means), difference=(means[condition] - means.unsteered).mean(), ci_lower=ci[0], ci_upper=ci[1]))
    return {'quality_summary.csv': pd.DataFrame(summary), 'quality_differences.csv': pd.DataFrame(differences), 'quality_by_language_category.csv': cells, 'quality_response_counts.csv': counts}


def reproduce(root):
    base = root / 'diagnostics'
    @lru_cache(maxsize=None)
    def read(relative):
        return pd.read_csv(base / relative, low_memory=False)

    out = quality_tables(base)
    # Restrict each three-condition pilot comparison to common valid prompts.
    pilot = pd.read_csv(base / 'configs/meaning_matched.csv')
    rows = []
    for route, g in pilot.groupby(['source_language', 'target_language', 'category'], sort=True):
        frames = {r.condition: read(r.evaluated_file) for r in g.itertuples()}
        key = prompt_key(list(frames.values())); pred = 'pred_' + route[2]
        sets = [set(f.loc[pd.to_numeric(f[pred], errors='coerce').notna(), key].dropna().astype(str)) for f in frames.values()]
        common = set.intersection(*sets)
        metrics = {}
        for condition, f in frames.items():
            sub = f[f[key].astype(str).isin(common)]
            labels = pd.to_numeric(sub[pred], errors='coerce')
            coherence = pd.to_numeric(sub.coherence_score, errors='coerce').dropna()
            metrics[condition] = dict(source_language=route[0], target_language=route[1], category=route[2], condition=condition, n=int(labels.notna().sum()), tbr=float(labels.mean()), valid_coherence_n=len(coherence), mean_coherence=float(coherence.mean()), low_coherence_rate=float((coherence<2).mean()))
        for record in metrics.values():
            record['gain_pp'] = 100 * (record['tbr'] - metrics['unsteered']['tbr'])
            rows.append(record)
    out['meaning_matched_results.csv'] = pd.DataFrame(rows)

    # At layer 14, every strength uses the intersection of generated prompt sets.
    sweep = pd.read_csv(base / 'configs/strength_sweep.csv')
    layer14 = {}
    for category, g in sweep[sweep.layer.eq(14)].groupby('category'):
        frames = [read(path) for row in g.itertuples() for path in (row.evaluated_file, row.baseline_file)]
        key = prompt_key(frames)
        layer14[category] = set.intersection(*(set(f[key].dropna().astype(str)) for f in frames))
    rows = []
    for row in sweep.itertuples():
        a, b = read(row.evaluated_file), read(row.baseline_file)
        key = prompt_key([a, b])
        common = layer14[row.category] if row.layer == 14 else set(a[key].dropna().astype(str)) & set(b[key].dropna().astype(str))
        a, b = (f[f[key].astype(str).isin(common)] for f in (a, b))
        x, y = (pd.to_numeric(f['pred_' + row.category], errors='coerce').dropna() for f in (a, b))
        coh = pd.to_numeric(a.coherence_score, errors='coerce').dropna()
        rows.append(dict(category=row.category, layer=row.layer, alpha=row.alpha, generated_n=len(a), valid_tbr_n=len(x), tbr_pct=100*x.mean(), baseline_valid_tbr_n=len(y), baseline_tbr_pct=100*y.mean(), valid_coherence_n=len(coh), mean_coherence=coh.mean(), low_coherence_pct=100*(coh<2).mean()))
    out['strength_sweep_results.csv'] = pd.DataFrame(rows)

    # Alternative-negative cross-lingual comparison: 50 prompt pairs per route.
    routes = pd.read_csv(base / 'configs/alternative_negative.csv')
    rows = []
    for row in routes.itertuples():
        a, b = read(row.evaluated_file), read(row.baseline_file)
        key = prompt_key([a, b]); pred = 'pred_' + row.category
        joined = a[[key, pred]].merge(b[[key, pred]], on=key, suffixes=('_steered', '_baseline'), validate='one_to_one')
        x, y = (pd.to_numeric(joined[pred + suffix], errors='coerce') for suffix in ('_steered', '_baseline'))
        valid = x.notna() & y.notna()
        rows.append(dict(source_language=row.source_language, target_language=row.target_language, category=row.category, matched_n=len(joined), valid_tbr_n=int(valid.sum()), tbr=x[valid].mean(), baseline_tbr=y[valid].mean(), difference=x[valid].mean()-y[valid].mean()))
    detail = pd.DataFrame(rows); rows = []
    for category in CATEGORIES + ['Overall']:
        g = detail if category == 'Overall' else detail[detail.category.eq(category)]
        rows.append(dict(category=category, comparisons=len(g), unsteered_tbr_pct=100*g.baseline_tbr.mean(), steered_tbr_pct=100*g.tbr.mean(), difference_pp=100*g.difference.mean(), positive=int((g.difference>0).sum())))
    out['alternative_negative_cells.csv'], out['alternative_negative_summary.csv'] = detail, pd.DataFrame(rows)
    return out


def check(root, frames):
    for name, actual in frames.items():
        expected = pd.read_csv(root / 'diagnostics/results' / name)
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check and args.output_dir is None:
        parser.error('Specify --check or --output-dir')
    frames = reproduce(args.root.resolve())
    if args.check:
        print(f'REPRODUCTION PASSED: {check(args.root.resolve(), frames)} result tables agree with the packaged references.')
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(args.output_dir / name, index=False)
        print(f'Wrote {len(frames)} tables to {args.output_dir}')


if __name__ == '__main__':
    main()
