# Data provenance and record identifiers

The CSVs contain saved generations, evaluator responses, configuration metadata,
and reference statistics used by the reproduction scripts. Generated text and
evaluator responses are retained as recorded. Missing judgments remain missing.

`condition_id`, `sample_id`, and `evaluation_id` join the main condition manifests
to row-level records. `source_file_id` is a stable artifact identifier. Original
execution paths, file names, and source-file hashes are omitted from manifests.
The distributed-file hashes in `MANIFEST.csv` and `CHECKSUMS.sha256` apply to the
files included here.

`diagnostics/configs/evaluation_files.csv` indexes diagnostic evaluator files and
their row counts. `original_row_index` identifies a row within its evaluator file.
Quality records use `category_evaluation_file` and `row_index` to link to category
judgments; their `sample_id` field is not a global join key.

The matched-literal files contain the first 100 valid category judgments in each
condition. The reproduction script calculates each condition's rate using those
included judgments. Additional pairing rules are documented in `SCHEMAS.md` and
`diagnostics/README.md`.

CrossCat retains exact evaluated continuations and the complete original JSONL
response-event stream after decompression. Each API response payload, response ID,
request settings, token usage, and attempt join is retained. Public input records
replace execution paths with stable model-scoped source IDs and map legacy
condition names to `omean` and `crosscat`. The raw event `input_sha256` and
`judge_manifest_sha256` refer to original private metadata; their matching opaque
identifiers are provided for provenance, rather than presented as hashes of the
rewritten public metadata. Text, template, full evaluation prompt, and request
hashes can all be independently verified from distributed files. Source-file
paths, source-file hashes, machine logs, and credentials are omitted. See
`diagnostics/crosscat/README.md` for the exact reproduction scope.
