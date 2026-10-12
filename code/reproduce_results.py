#!/usr/bin/env python3
"""Recompute paper statistics from saved generation-level evaluator judgments.

python code/reproduce_results.py . --check --output-dir _outputs
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided binomial test on discordant pairs, under p=1/2."""
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def bh(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1.0)
    return result


def load_outputs(root, manifest):
    columns = ['condition_id', 'sample_id', 'row_index', 'target_label', 'coherence_score']
    pieces = []
    for relative in manifest.public_output.drop_duplicates():
        pieces.append(pd.read_csv(root / 'evaluator_outputs' / relative, usecols=columns))
    data = pd.concat(pieces, ignore_index=True)
    if data.duplicated(['condition_id', 'sample_id']).any():
        raise ValueError('Duplicate sample ID within a condition')
    for name in ['target_label', 'coherence_score']:
        data[name] = pd.to_numeric(data[name], errors='coerce')
    if not data.target_label.dropna().isin([0, 1]).all():
        raise ValueError('Nonbinary target judgment')
    if not data.coherence_score.dropna().isin(range(5)).all():
        raise ValueError('Coherence score outside 0--4')
    return {key: frame.set_index('sample_id') for key, frame in data.groupby('condition_id', sort=False)}


def condition_metrics(manifest, outputs):
    rows = []
    for row in manifest.itertuples():
        d = outputs[row.condition_id]
        labels, coherence = d.target_label.dropna(), d.coherence_score.dropna()
        joint = d.dropna(subset=['target_label', 'coherence_score'])
        rows.append(dict(condition_id=row.condition_id, model=row.model, condition=row.condition,
                         source_lang=row.source_lang, target_lang=row.target_lang, category=row.category,
                         random_seed=row.random_seed, n=len(d), valid_tbr_n=len(labels),
                         tbr_pct=100*labels.mean(), valid_coherence_n=len(coherence),
                         mean_coherence=coherence.mean(), low_coherence_pct=100*coherence.lt(2).mean(),
                         valid_joint_n=len(joint), joint_success_pct=100*(joint.target_label.eq(1) & joint.coherence_score.ge(2)).mean()))
    return pd.DataFrame(rows)


def compare(a, b, *, geometry=False):
    """Apply the pairing and rate denominators used by each analysis."""
    x = a.sort_values('row_index').target_label.dropna()
    y = b.sort_values('row_index').target_label.dropna()
    if not geometry or set(x.index) == set(y.index):
        pairs = pd.concat([x.rename('a'), y.rename('b')], axis=1, join='inner')
    else:
        n = min(len(x), len(y))
        pairs = pd.DataFrame({'a': x.iloc[:n].to_numpy(), 'b': y.iloc[:n].to_numpy()})
    if not len(pairs):
        raise ValueError('No paired valid judgments')
    bonly = int((pairs.b.eq(1) & pairs.a.eq(0)).sum())
    aonly = int((pairs.a.eq(1) & pairs.b.eq(0)).sum())
    rate_a, rate_b = (x.mean(), y.mean()) if geometry else (pairs.a.mean(), pairs.b.mean())
    return dict(matched_n=len(pairs), tbr_a_pct=100*rate_a, tbr_b_pct=100*rate_b,
                delta_pp=100*(pairs.b.mean()-pairs.a.mean()), discordant_b_only=bonly,
                discordant_a_only=aonly, mcnemar_p_exact=exact_mcnemar(bonly, aonly))


def reproduce(root):
    manifest = pd.read_csv(root / 'evaluator_outputs/main_condition_manifest.csv')
    outputs = load_outputs(root, manifest)
    lookup = {}
    for row in manifest[manifest.condition.ne('random')].itertuples():
        key = (row.model, row.source_lang, row.target_lang, row.category, row.condition)
        if key in lookup:
            raise ValueError('Ambiguous condition metadata')
        lookup[key] = outputs[row.condition_id]

    # The reference supplies comparison identities and correction-family metadata;
    # every numeric statistic and significance decision is recomputed from raw labels.
    reference = pd.read_csv(root / 'results/comparison_level_results.csv')
    rows = []
    metadata = ['comparison_id', 'experiment', 'model', 'source_lang', 'target_lang', 'category', 'condition_a', 'condition_b', 'bh_family']
    for _, row in reference.iterrows():
        a = lookup[(row.model, row.target_lang, row.target_lang, row.category, row.condition_a)]
        b = lookup[(row.model, row.source_lang, row.target_lang, row.category, row.condition_b)]
        rows.append({**row[metadata].to_dict(), **compare(a, b, geometry=row.experiment == 'geometry')})
    comparisons = pd.DataFrame(rows)
    comparisons['mcnemar_q_bh'] = np.nan
    comparisons['result'] = 'descriptive_comparison'
    for family, group in comparisons.groupby('bh_family'):
        if family == 'descriptive_only':
            continue
        q = bh(group.mcnemar_p_exact)
        comparisons.loc[group.index, 'mcnemar_q_bh'] = q
        labels = np.where((q < .05) & (group.delta_pp > 0), 'significant gain',
                          np.where((q < .05) & (group.delta_pp < 0), 'significant loss', 'not significant'))
        if family.startswith('monolingual_'):
            labels = [value.replace(' ', '_') for value in labels]
        comparisons.loc[group.index, 'result'] = labels
    comparisons = comparisons[reference.columns]
    conditions = condition_metrics(manifest, outputs)
    # Equal category weights within a model/language/condition; repeated random
    # seeds and source routes are first averaged within their target/category cell.
    metric_columns = ['tbr_pct', 'mean_coherence', 'low_coherence_pct', 'joint_success_pct']
    cells = conditions.groupby(['model', 'condition', 'target_lang', 'category'], as_index=False)[metric_columns].mean()
    summaries = cells.groupby(['model', 'condition', 'target_lang'], as_index=False)[metric_columns].mean()

    mono = pd.read_csv(root / 'results/monolingual_all_88_cells.csv')
    cross = pd.read_csv(root / 'results/crosslingual_all_408_routes.csv')
    for table, experiment, positive_prefix, disc_prefix, baseline_col, result_col in [
        (mono, 'monolingual', 'monolingual', 'monolingual', 'unsteered_tbr_pct', 'adjusted_result'),
        (cross, 'primary_crosslingual', 'crosslingual', 'steered', 'unsteered_tcr_pct', 'adjusted_significance')]:
        for index, row in table.iterrows():
            language = row.language if experiment == 'monolingual' else row.target_lang
            source = language if experiment == 'monolingual' else row.source_lang
            match = comparisons[(comparisons.experiment == experiment) & (comparisons.model == row.model)
                                & (comparisons.category == row.category) & (comparisons.target_lang == language)
                                & (comparisons.source_lang == source)]
            if len(match) != 1:
                raise ValueError('Ambiguous result-table join')
            stats = match.iloc[0]
            values = dict(matched_n=stats.matched_n, unsteered_positive_n=round(stats.matched_n * stats.tbr_a_pct/100),
                          delta_pp=stats.delta_pp, discordant_unsteered_only=stats.discordant_a_only,
                          mcnemar_p_exact=stats.mcnemar_p_exact, mcnemar_q_bh=stats.mcnemar_q_bh)
            values.update({baseline_col: stats.tbr_a_pct, positive_prefix+'_positive_n': round(stats.matched_n*stats.tbr_b_pct/100),
                           positive_prefix+'_tbr_pct': stats.tbr_b_pct, 'discordant_'+disc_prefix+'_only': stats.discordant_b_only,
                           result_col: stats.result})
            for key, value in values.items():
                table.loc[index, key] = value

    explicit_manifest = pd.read_csv(root / 'evaluator_outputs/explicit_prompt/condition_manifest.csv')
    explicit_outputs = load_outputs(root, explicit_manifest)
    explicit = pd.read_csv(root / 'results/explicit_prompt_all_44_cells.csv')
    for index, row in explicit.iterrows():
        entry = explicit_manifest[(explicit_manifest.model == row.model) & (explicit_manifest.target_lang == row.language)
                                  & (explicit_manifest.category == row.category)].iloc[0]
        d = explicit_outputs[entry.condition_id]
        x, y = d.target_label.dropna(), d.coherence_score.dropna()
        values = dict(n=len(d), valid_tcr_n=len(x), TCR=x.mean(), valid_coherence_n=len(y), mean_coherence=y.mean(),
                      low_coherence_rate=y.lt(2).mean(), missing_tcr_n=len(d)-len(x), missing_coherence_n=len(d)-len(y),
                      raw_category_error_rate=1-len(x)/len(d), raw_coherence_error_rate=1-len(y)/len(d),
                      TCR_pct=100*x.mean(), low_coherence_rate_pct=100*y.lt(2).mean())
        for key, value in values.items():
            explicit.loc[index, key] = value

    matched_manifest = pd.read_csv(root / 'evaluator_outputs/matched_literal/condition_manifest.csv')
    matched_outputs = load_outputs(root, matched_manifest)
    matched = pd.read_csv(root / 'results/matched_literal_all_44_cells.csv')
    for index, row in matched.iterrows():
        selected = matched_manifest[(matched_manifest.model == row.model) & (matched_manifest.target_lang == row.language)
                                    & (matched_manifest.category == row.category)]
        rates = {}
        for entry in selected.itertuples():
            labels = matched_outputs[entry.condition_id].target_label.dropna()
            name = {'unsteered': 'unsteered', 'caption_built': 'caption_built', 'matched_literal': 'matched_literal'}[entry.condition]
            rates[name] = 100*labels.mean()
            matched.loc[index, name+'_tbr_pct'] = rates[name]
            matched.loc[index, name+'_valid_n'] = len(labels)
        a, b = rates['caption_built']-rates['unsteered'], rates['matched_literal']-rates['unsteered']
        matched.loc[index, 'caption_built_delta_pp'] = a
        matched.loc[index, 'matched_literal_delta_pp'] = b
        matched.loc[index, 'sign_retained'] = bool((a >= 0) == (b >= 0))

    return {'comparison_level_results.csv': comparisons, 'monolingual_all_88_cells.csv': mono,
            'crosslingual_all_408_routes.csv': cross, 'explicit_prompt_all_44_cells.csv': explicit,
            'matched_literal_all_44_cells.csv': matched, 'condition_metrics.csv': conditions,
            'model_language_summary.csv': summaries}


def check(root, frames):
    checked = 0
    for name, actual in frames.items():
        path = root / 'results' / name
        if not path.exists():
            continue
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), pd.read_csv(path), check_dtype=False,
                                      check_exact=False, atol=1e-9, rtol=1e-9)
        checked += 1
    if checked != 5:
        raise ValueError('Expected five distributed main-result tables')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not args.check and args.output_dir is None:
        parser.error('Specify --check or --output-dir')
    frames = reproduce(args.root.resolve())
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(args.output_dir / name, index=False)
        print(f'Wrote {len(frames)} tables to {args.output_dir}')
    if args.check:
        print(f'REPRODUCTION PASSED: {check(args.root.resolve(), frames)} main-result tables agree with the references.')


if __name__ == '__main__':
    main()
