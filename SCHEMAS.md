# Schemas and Units

- TBR columns ending in `_pct` are percentages on 0--100.
- Columns ending in `_pp` are percentage-point differences.
- Coherence scores use the paper's 0--4 rubric.
- In `comparison_level_results.csv`, condition A is the baseline and condition B
  is the intervention. Monolingual and primary cross-lingual TBR columns use
  paired valid judgments. Geometry TBR columns use each condition's valid
  judgments separately. `delta_pp` uses the aligned pairs in the paired test.
- `discordant_b_only` and `discordant_a_only` are the paired binary counts used
  by the exact two-sided McNemar test.
- Primary cross-lingual q-values use one 408-route Benjamini--Hochberg family.
- Monolingual q-values use a separate 22-cell family within each model.
- Rows labeled `descriptive_only` have empty q-values.
- `condition_id`, `sample_id`, and `evaluation_id` provide joins between
  condition manifests and row-level outputs.

Appendix diagnostic schemas, pairing rules, and quality-score units are described
in `diagnostics/README.md`. The diagnostic files retain their evaluator column names.
Quality-record IDs and original evaluator row indices provide explicit joins.

CrossCat uses a separate schema documented in
`diagnostics/crosscat/README.md`: model-scoped `job_id` joins exact evaluated text
to full response events; public conditions are `unsteered`, `omean`, and
`crosscat`. The 32 comparison rows each use 100 identical prompts with valid
judgments in all three conditions. Model and overall summaries weight these cells
equally. CrossCat values are descriptive; no significance values are inferred.

## Main reproduction code

`load_outputs` reads the judgment columns named by each condition manifest and
checks that sample IDs are unique within a condition. `compare` removes missing
category judgments and joins by `sample_id` for monolingual and primary
cross-lingual comparisons. For geometry comparisons, it joins by `sample_id`
when both valid identifier sets match; otherwise, it reproduces the original
geometry analysis's fallback by aligning valid rows in stored row order, up to
the shorter condition's length.
`exact_mcnemar` computes the two-sided binomial probability of the discordant
pairs; `bh` adjusts probabilities within the configured correction family.

Comparison identifiers, model/language/category labels, and correction-family
definitions come from the reference-table metadata. Numeric comparison values
and significance decisions are recalculated from the saved judgments.

`condition_metrics.csv` contains separate valid-judgment counts for TBR and
coherence. Joint success uses rows with both judgments available and requires a
positive category label and coherence of at least 2. These condition-level
summaries do not apply the paired mask used for significance tests.

`model_language_summary.csv` first averages repeated random seeds or source routes
within target-language/category cells, then averages the available categories
equally. The explicit-prompt table uses all valid judgments in each condition.
The matched-literal table uses the included valid judgments separately for each
condition; `sign_retained` compares whether both changes are nonnegative or both
are negative.
