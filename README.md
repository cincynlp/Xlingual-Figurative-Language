# Result reproduction and supplementary materials

This repository contains code for recomputing the paper's statistics from saved
generation and evaluator records, together with prompts, configuration files,
and construction pairs referenced in the appendix.

## Setup

Use Python 3.9 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The scripts use local CSV and compressed JSONL files. After dependency installation, execution requires
no network access, evaluator API credentials, model downloads, or GPU.
The repository supports recomputation from saved outputs; model generation and
new evaluator calls are outside the code's scope.

## Intervention scale

For each run, the selected target-language caption prompts pass through the
unsteered model with the same instruction and chat template as generation. The
L2 norm of the selected block's output activation at each prompt's last
non-padding token is computed in float32; averaging these norms gives the scale
`s`, which is held fixed during that generation invocation. This is a scalar for
the model, layer, and run. The paper's steering strength multiplies this scale;
the recorded norm standard deviation is not used.

## Commands

```bash
# Recompute main-result tables and compare them with the saved references.
python code/reproduce_results.py . --check --output-dir _outputs/main

# Recompute appendix diagnostic tables and compare them with the references.
python code/reproduce_diagnostics.py . --check --output-dir _outputs/diagnostics

# Recompute the CrossCat appendix comparison from complete saved judge responses.
python code/reproduce_crosscat.py . --check --output-dir _outputs/crosscat

# Check file integrity, schemas, record links, and diagnostic reproduction.
python code/validate_artifact.py .
```

`--check` compares recomputed tables with the reference CSVs and exits with an
error on a mismatch. `--output-dir` writes new CSVs without replacing the reference
files. Either option may be used on its own. Generated files under `_outputs/`,
virtual environments, Python caches, and Git metadata are excluded from the
distributed-file inventory.

## Code

| File | Function |
|---|---|
| `code/reproduce_results.py` | Loads saved judgments; pairs samples; calculates TBR, coherence, joint-success summaries, exact McNemar tests, and BH-adjusted values; reconstructs main result tables. |
| `code/reproduce_diagnostics.py` | Recomputes synthetic-pair, steering-strength, alternative-negative, and conditional-quality tables using their configured masks and averaging rules. |
| `code/reproduce_crosscat.py` | Checks saved evaluator requests and responses, category membership and language weights, and recomputes all 32 CrossCat comparison cells and model summaries. |
| `code/validate_artifact.py` | Checks the file inventory and checksums, required schemas and row counts, diagnostic record links, reference-table agreement, and configured anonymity patterns. |

See [ARTIFACT_MAP.md](ARTIFACT_MAP.md) for input and output locations,
[SCHEMAS.md](SCHEMAS.md) for fields and units, and
[diagnostics/README.md](diagnostics/README.md) for the diagnostic code.
