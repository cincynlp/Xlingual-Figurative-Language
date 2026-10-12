# CrossCat category-matching diagnostic

This diagnostic compares unsteered generation, OMean, and CrossCat on the same
100 test prompts per model, target language, and category. The scope is four
models, English and Chinese targets, and idiom, metaphor, irony, and sarcasm:
32 comparison cells and 9,600 valid category/text judgments. Simile is excluded.

## Construction and generation

OMean averages the target-category directions from other source languages.
CrossCat uses exactly that source-language pool, averages the available other
study categories within each source language, and then gives each language equal
weight. The final direction is L2-normalized. Thus CrossCat excludes both the
target category and target language; it is not a mean within the target language.
Both constructions use caption-negative directions.

For target category c and target language t, let S be the other languages with an
available direction for c, and let A(s,c) be the available study categories in
source language s excluding c. CrossCat is:

```text
normalize(sum over s in S of [sum over k in A(s,c) of v(s,k)] / |A(s,c)|)
```

English, Chinese, Bengali, and Italian provide all four categories; Spanish and
German provide idiom, metaphor, and irony. For idiom, metaphor, and irony targets,
S contains five languages and CrossCat has 13 terms. For sarcasm, S contains
three languages and CrossCat has nine terms. The integer weights in `protocol.json`
are proportional to the reciprocal category counts and give equal total weight
to each source language. Stored normalized vectors were converted to float32
without undoing their small bfloat16 normalization rounding errors, consistent
with cached OMean composition.

The first 100 prompts were selected before new judgments. Unsteered and OMean
continuations were reused from cached runs; CrossCat required 3,200 new
continuations. All three conditions were freshly judged for this diagnostic.
The same 100 unsteered continuations per model and language are evaluated against
each of the four category prompts, so 9,600 judgments do not imply 9,600 distinct
continuations. These are the cached raw generations recorded by the protocol;
they are not asserted to match every earlier published evaluator copy.

Generation uses the fixed manuscript layers (14, 35, 15, and 19 for Qwen3-8B,
Qwen3-32B, Llama-3.1-8B, and Ministral-3-8B), ratio 1, bfloat16, normal continuation
prompts, all-prompt-token steering, and at most 1,024 new tokens. Each new cell
reuses the hidden-norm mean/std measured on the corresponding cached 500-prompt
OMean run and retains its generation batch size. Per-model sampling settings,
per-cell seeds, batch sizes, and calibration statistics appear in `protocol.json`.
Cached generation seeds were not recorded, so the conditions share prompts but
do not claim identical RNG states. No parameters were selected using these test
judgments.

## Judging and reproduction

Requests used the published language/category TBR templates, the official
DeepSeek endpoint, requested model `deepseek-v4-flash`, temperature 0, a
256-token output limit, and thinking disabled. The returned model identifier is
`deepseek-flash`. Each saved response must contain exactly one explicit
`Label: YES` or `Label: NO` line, one completion choice with finish reason `stop`,
and no reasoning content. Missing, invalid, or truncated responses are never
negative labels. All 9,600 recorded attempts have valid completed responses.

From the repository root:

```bash
python code/reproduce_crosscat.py . --check --output-dir _outputs/crosscat
```

This reconstructs and hashes each full evaluator request, reparses labels from
the raw responses, verifies attempt joins and 100 common prompts in every cell,
checks the source-language pool and category weights, and compares both resulting
tables with their saved references. It requires no network or GPU. The package
supports recomputation from saved outputs; source vector tensors and the model
generation backend are not distributed here, so it does not regenerate vectors
or continuations.

`cell_results.csv` contains all 32 cells, common-valid denominators, each
condition's positive counts and TBR, gains over unsteered generation, and
OMean-minus-CrossCat differences. `model_summary.csv` averages the eight cells
equally within each model and all 32 equally for the overall row. Rates are
percentages and differences are percentage points. Values remain unrounded in
CSV; the paper displays two decimals with decimal round-half-up. No p-values or
claims of statistical significance are added.

## Record schema and provenance

- `protocol.json` contains the model list, condition definitions, source-file
  registry, source-language/category memberships and weights, generation
  settings, norm calibration, and evaluator settings and template hashes.
- `raw/{model}_inputs.jsonl.gz` contains exact `text` and `simple_prompt`, their
  model-scoped `job_id`, target language/category, public `condition`, original
  zero-based row index, `text_sha256`, and a stable `source_file_id`. That ID joins
  the protocol registry and deliberately does not identify an execution path.
- `raw/{model}_responses.jsonl.gz` retains the complete original event stream
  byte-for-byte after decompression. A `(job_id, run_id, attempt)` tuple links
  start and finish events; `response` holds the complete API payload, including
  evaluator text, response ID, returned model, and usage.
- Legacy job IDs contain `matched_omean` or `mismatched_omean`; these identifiers
  are preserved for exact joins. The input `legacy_condition` maps them to the
  public names `omean` and `crosscat` without changing the recorded evidence.
- `original_input_sha256` and `original_judge_manifest_sha256` retain the opaque
  hashes referenced by original events. Original metadata included execution
  paths, so these two hashes cannot be recomputed from rewritten public metadata.
  Generated text, template, evaluation prompt, and full request hashes can be
  recomputed exactly and are checked. Distributed-file checksums cover the
  anonymized public records. No credentials or machine logs are distributed.

The contrast measures the additional TBR effect of category matching within this
construction. Other-category means also increase TBR, and their comparison with
OMean does not uniquely separate category identity from shared corpus/domain
features. Its behavioral scope is English and Chinese; use of other source
languages in the means does not extend the target-language evaluation scope.
