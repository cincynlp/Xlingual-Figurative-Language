# Code, inputs, and outputs

| Material | Location | Used by |
|---|---|---|
| Study settings and model configuration | `configs/paper_config.json` | Reference configuration |
| Main condition metadata | `evaluator_outputs/main_condition_manifest.csv` | `reproduce_results.py` |
| Main saved generations and judgments | `evaluator_outputs/main/*.csv.gz` | `reproduce_results.py` |
| Explicit-prompt records and condition metadata | `evaluator_outputs/explicit_prompt/` | `reproduce_results.py` |
| Matched-literal records and condition metadata | `evaluator_outputs/matched_literal/` | `reproduce_results.py` |
| Main reference tables | `results/*.csv` | `reproduce_results.py --check` |
| Diagnostic input-file configuration | `diagnostics/configs/` | `reproduce_diagnostics.py` |
| Diagnostic generations and judgments | `diagnostics/raw/` | `reproduce_diagnostics.py` |
| Diagnostic reference tables | `diagnostics/results/` | `reproduce_diagnostics.py --check` |
| CrossCat construction, generation, and evaluator protocol | `diagnostics/crosscat/protocol.json` | `reproduce_crosscat.py` |
| CrossCat exact evaluated continuations and complete response events | `diagnostics/crosscat/raw/*.jsonl.gz` | `reproduce_crosscat.py` |
| CrossCat 32 comparison cells and model summaries | `diagnostics/crosscat/results/` | `reproduce_crosscat.py --check` |
| Synthetic pair candidates, screening responses, and selected pairs | `diagnostics/construction/` | Supplementary construction records |
| Category-evaluation prompts | `prompts/tbr/`, `prompts/prompt_registry.json` | Supplementary evaluator templates |
| Coherence-evaluation prompt and rubric | `prompts/coherence/coherence_prompt.txt` | Supplementary evaluator template |
| Generation templates | `prompts/generation_prompt_templates.json` | Supplementary generation templates |
| Quality and synthetic-pair generation/screening prompts | `diagnostics/prompts/` | Supplementary evaluator and generation templates |
| File sizes, row counts, and integrity hashes | `MANIFEST.csv`, `CHECKSUMS.sha256` | `validate_artifact.py` |

`reproduce_results.py` writes the five table names already present in `results/`,
plus `condition_metrics.csv` and `model_language_summary.csv`.
`reproduce_diagnostics.py` writes the eight table names present in
`diagnostics/results/`. Both scripts write only to the requested output directory.
`reproduce_crosscat.py` independently writes `cell_results.csv`,
`model_summary.csv`, and an `audit.json` report to its requested output directory.
