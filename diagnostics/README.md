# Diagnostic code and data

`code/reproduce_diagnostics.py` reads the configuration and evaluator CSVs in this
directory and returns recomputed tables. All paths in the configuration CSVs are
relative to this directory.

```bash
python code/reproduce_diagnostics.py . --check --output-dir _outputs/diagnostics
```

Run this command from the repository root after installing `requirements.txt`.
`--check` compares the computed tables with `diagnostics/results/`.
`--output-dir` writes the computed tables to the specified directory.

## Functions

| Function | Operation |
|---|---|
| `prompt_key` | Selects a shared prompt column and rejects duplicate nonempty prompt keys. |
| `quality_tables` | Filters valid quality scores, constructs language/category means, identifies cells shared by all conditions, and bootstraps cell averages and condition differences. |
| `reproduce` | Reads configured evaluator files, applies each diagnostic's inclusion rules, and assembles output tables. |
| `check` | Compares recomputed DataFrames with the corresponding reference CSVs. |

## Input and output mapping

| Input | Outputs under `results/` |
|---|---|
| `configs/meaning_matched.csv` and linked evaluator files | `meaning_matched_results.csv` |
| `configs/strength_sweep.csv` and linked evaluator files | `strength_sweep_results.csv` |
| `configs/alternative_negative.csv` and linked evaluator files | `alternative_negative_cells.csv`, `alternative_negative_summary.csv` |
| `raw/quality_evaluated.csv.gz` | `quality_summary.csv`, `quality_differences.csv`, `quality_by_language_category.csv`, `quality_response_counts.csv` |

`configs/evaluation_files.csv` indexes portable evaluator files, row counts, and
omitted execution-specific columns. `original_row_index` is the zero-based row
index in an evaluator file. The raw files retain generation text, category labels,
coherence judgments, evaluator responses, and experimental settings.

Synthetic-pair category judgments use `deepseek-v4-flash` through the official
DeepSeek API with thinking disabled, temperature 0, and a 256-token output limit.
The category templates are in `prompts/tbr/` at the repository root. All three
conditions share 100 evaluation prompts per route. The category-evaluation
columns record the requested and returned model names, endpoint, settings,
response ID, and completion time. `coherence_evaluation_source_file` identifies
the evaluator file supplying each retained coherence score. Coherence means use
the available scores within the same 100-prompt sets.

`quality_record_id` is unique within the quality file. `category_evaluation_file`
and `row_index` link a quality record to its category-evaluation record. The
quality file's `sample_id` is a batch identifier, not a global join key.

## Inclusion and averaging rules

- **Synthetic-pair comparison:** intersect prompt sets with valid category
  judgments across the three configured conditions, separately for each route.
  Calculate coherence using available coherence scores within that prompt set.
- **Strength sweep:** intersect generated prompt sets across layer-14 strengths;
  at other layers, intersect each condition with its baseline. Use separate valid
  denominators for category and coherence judgments.
- **Alternative negatives:** join each route to its baseline by prompt and retain
  pairs with valid category judgments in both conditions. Average route rates
  equally within each category and for the overall output row.
- **Conditional quality:** retain quality scores from 0 through 4. Compute means
  within condition/target-language/category cells, pooling available source
  routes. Use cells with valid scores in all conditions for comparisons. Resample
  cells 10,000 times using NumPy seed 20260710 and shared draws across conditions.
  The resampling unit is a cell; outputs are not paired across conditions.

## Supplementary inputs

`construction/` stores synthetic pair candidates, screening responses, a screening
summary, and the selected construction pairs. `prompts/` stores the quality rubric,
quality-judging template, synthetic-pair generation and screening templates, and
category definitions. The repository-level `prompts/` directory contains the
category and coherence evaluator templates.

Columns ending in `_pct` use percentages; `_pp` uses percentage-point differences.
Other TBR and low-coherence columns use fractions. Quality and coherence use the
0--4 scale, and quality confidence intervals use score units.

## CrossCat comparison

The separate [CrossCat bundle](crosscat/README.md) contains 32 three-condition
comparisons across four models, English/Chinese, and four categories. It provides
exact evaluated text, all recorded response events, construction membership and
weights, and an independent reproduction command:

```bash
python code/reproduce_crosscat.py . --check --output-dir _outputs/crosscat
```

The eight tables documented above retain their existing inputs and calculations.
