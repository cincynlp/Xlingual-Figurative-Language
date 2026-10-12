#!/usr/bin/env python3
"""Validate the anonymous results repository."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import pandas as pd


CATEGORIES = {"idiom", "metaphor", "irony", "sarcasm"}
CONDITIONS = {"unsteered", "monolingual", "crosslingual", "random", "omean", "residual"}


def distributed_files(root: Path):
    """Visit artifact files while excluding local execution and Git directories."""
    ignored = {".git", ".venv", "venv", "__pycache__", "_outputs"}
    for parent, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in ignored)
        for name in sorted(names):
            if name != ".DS_Store" and not name.endswith((".pyc", ".pyo")):
                yield Path(parent) / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(value: bool, message: str, errors: list[str]) -> None:
    if not value:
        errors.append(message)


def frame(root: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(root / relative, low_memory=False)


def validate_results(root: Path, errors: list[str]) -> None:
    mono = frame(root, "results/monolingual_all_88_cells.csv")
    require(len(mono) == 88, "monolingual result count is not 88", errors)
    require(mono.groupby("model").size().eq(22).all(), "monolingual results are not 22 cells per model", errors)
    require(set(mono.category) == CATEGORIES, "monolingual category set is incorrect", errors)
    require((mono.adjusted_result == "significant_gain").sum() == 61, "monolingual significant-gain count is not 61", errors)
    require((mono.adjusted_result == "significant_loss").sum() == 0, "monolingual significant-loss count is not 0", errors)
    for prefix in ("unsteered", "monolingual"):
        rates = 100 * mono[prefix + "_positive_n"] / mono.matched_n
        require((rates - mono[prefix + "_tbr_pct"]).abs().lt(1e-8).all(),
                prefix + " rates disagree with paired positive counts", errors)
    require((mono.monolingual_tbr_pct - mono.unsteered_tbr_pct - mono.delta_pp).abs().lt(1e-8).all(),
            "monolingual gains disagree with paired rates", errors)

    cross = frame(root, "results/crosslingual_all_408_routes.csv")
    require(len(cross) == 408, "cross-lingual route count is not 408", errors)
    require(set(cross.category) == CATEGORIES, "cross-lingual category set is incorrect", errors)
    require((cross.adjusted_significance == "significant gain").sum() == 233, "cross-lingual significant-gain count is not 233", errors)
    require((cross.adjusted_significance == "significant loss").sum() == 6, "cross-lingual significant-loss count is not 6", errors)
    require(round(cross.unsteered_tcr_pct.mean(), 1) == 12.1, "cross-lingual unsteered mean is not 12.1", errors)
    require(round(cross.crosslingual_tbr_pct.mean(), 1) == 20.4, "cross-lingual steered mean is not 20.4", errors)
    require(round(cross.delta_pp.mean(), 1) == 8.3, "cross-lingual gain is not 8.3", errors)

    master = frame(root, "results/comparison_level_results.csv")
    require(master.comparison_id.is_unique, "comparison IDs are not unique", errors)
    require(len(master[master.experiment == "monolingual"]) == 88, "comparison master lacks 88 monolingual rows", errors)
    require(len(master[master.experiment == "primary_crosslingual"]) == 408, "comparison master lacks 408 primary cross-lingual rows", errors)
    descriptive = master[master.bh_family == "descriptive_only"]
    require(descriptive.mcnemar_q_bh.isna().all(), "descriptive comparisons contain q-values", errors)

    explicit = frame(root, "results/explicit_prompt_all_44_cells.csv")
    require(len(explicit) == 44, "explicit-prompt result count is not 44", errors)
    require(explicit.groupby("model").size().eq(22).all(), "explicit-prompt results are not 22 cells per model", errors)
    bn = explicit[explicit.language == "bn"].groupby("model").TCR.mean()
    require(round(100 * bn.get("qwen", float("nan")), 1) == 20.8, "Qwen Bengali explicit-prompt mean is incorrect", errors)
    require(round(100 * bn.get("mistral", float("nan")), 1) == 61.1, "Ministral Bengali explicit-prompt mean is incorrect", errors)

    matched = frame(root, "results/matched_literal_all_44_cells.csv")
    require(len(matched) == 44, "matched-literal result count is not 44", errors)
    require(matched.groupby("model").size().eq(22).all(), "matched-literal results are not 22 cells per model", errors)
    qwen_metaphor = matched[(matched.model == "qwen") & (matched.category == "metaphor")]
    require(round(qwen_metaphor.caption_built_delta_pp.mean(), 1) == 5.8, "Qwen caption-built metaphor mean is incorrect", errors)
    require(round(qwen_metaphor.matched_literal_delta_pp.mean(), 1) == -1.8, "Qwen matched-literal metaphor mean is incorrect", errors)
    qwen = matched[matched.model == "qwen"]
    require(round(qwen.caption_built_delta_pp.mean(), 1) == 3.5, "Qwen caption-built overall mean is incorrect", errors)
    require(round(qwen.matched_literal_delta_pp.mean(), 1) == 5.4, "Qwen matched-literal overall mean is incorrect", errors)


def validate_outputs(root: Path, errors: list[str]) -> None:
    manifest = frame(root, "evaluator_outputs/main_condition_manifest.csv")
    require(len(manifest) == 1024, "main condition count is not 1024", errors)
    require(set(manifest.condition) == CONDITIONS, "main condition set is incorrect", errors)
    require(set(manifest.category) == CATEGORIES, "main evaluator category set is incorrect", errors)
    expected = {"unsteered": 88, "monolingual": 88, "crosslingual": 408, "random": 264, "omean": 88, "residual": 88}
    require(manifest.condition.value_counts().to_dict() == expected, "main condition counts are incorrect", errors)
    require(manifest.condition_id.is_unique, "main condition IDs are not unique", errors)
    condition_ids = set(manifest.condition_id)

    required_columns = {
        "condition_id", "evaluation_id", "sample_id", "source_sentence",
        "generation_prompt", "completion_only", "evaluated_text", "target_label",
        "target_reason", "target_raw", "coherence_score", "coherence_reason", "coherence_raw",
    }
    expected_rows = {
        "evaluator_outputs/main/baseline_and_monolingual.csv.gz": 88_000,
        "evaluator_outputs/main/crosslingual.csv.gz": 204_000,
        "evaluator_outputs/main/random.csv.gz": 26_400,
        "evaluator_outputs/main/geometry.csv.gz": 88_000,
    }
    seen: set[str] = set()
    for relative, expected_count in expected_rows.items():
        count = 0
        for chunk in pd.read_csv(root / relative, chunksize=10_000, low_memory=False):
            count += len(chunk)
            require(required_columns.issubset(chunk.columns), f"required columns missing in {relative}", errors)
            require(set(chunk.condition_id).issubset(condition_ids), f"unknown condition ID in {relative}", errors)
            require(chunk.evaluation_id.is_unique, f"duplicate evaluation ID within {relative}", errors)
            overlap = seen.intersection(chunk.evaluation_id)
            require(not overlap, f"evaluation ID repeated across main files at {relative}", errors)
            seen.update(chunk.evaluation_id)
        require(count == expected_count, f"row count for {relative} is {count}, expected {expected_count}", errors)

    explicit_manifest = frame(root, "evaluator_outputs/explicit_prompt/condition_manifest.csv")
    require(len(explicit_manifest) == 44, "explicit-prompt condition count is not 44", errors)
    require(set(explicit_manifest.category) == CATEGORIES, "explicit-prompt category set is incorrect", errors)
    explicit_rows = sum(len(chunk) for chunk in pd.read_csv(root / "evaluator_outputs/explicit_prompt/explicit_prompt.csv.gz", chunksize=10_000))
    require(explicit_rows == 4_400, "explicit-prompt evaluator row count is not 4400", errors)

    matched_manifest = frame(root, "evaluator_outputs/matched_literal/condition_manifest.csv")
    require(len(matched_manifest) == 132, "matched-literal condition count is not 132", errors)
    require(set(matched_manifest.category) == CATEGORIES, "matched-literal category set is incorrect", errors)
    require((matched_manifest.public_rows == 100).all(), "matched-literal conditions do not have 100 rows", errors)
    require((matched_manifest.valid_tbr_n == 100).all(), "matched-literal conditions do not have 100 valid judgments", errors)
    matched_rows = sum(len(chunk) for chunk in pd.read_csv(root / "evaluator_outputs/matched_literal/matched_literal.csv.gz", chunksize=10_000))
    require(matched_rows == 13_200, "matched-literal evaluator row count is not 13200", errors)


def validate_diagnostics(root: Path, errors: list[str]) -> None:
    from reproduce_diagnostics import reproduce, check
    try:
        count = check(root, reproduce(root))
        require(count == 8, "expected eight reproduced appendix tables", errors)
        base = root / "diagnostics"
        registry = frame(root, "diagnostics/configs/evaluation_files.csv")
        require(registry.file.is_unique, "duplicate diagnostic evaluator file IDs", errors)
        quality = frame(root, "diagnostics/raw/quality_evaluated.csv.gz")
        require(quality.quality_record_id.is_unique, "duplicate quality record IDs", errors)
        require(set(quality.category) == CATEGORIES, "quality category scope is incorrect", errors)
        for relative, group in quality.groupby("category_evaluation_file"):
            source = pd.read_csv(base / relative, low_memory=False).set_index("original_row_index")
            require(set(group.row_index).issubset(source.index), "quality record has missing category-evaluation row", errors)
            for category, selected in group.groupby("category"):
                labels = pd.to_numeric(source.loc[selected.row_index, "pred_" + category], errors="coerce")
                require(labels.eq(1).all(), "quality record does not link to a positive category judgment", errors)
        for row in registry.itertuples():
            actual = pd.read_csv(base / row.file, low_memory=False)
            require(len(actual) == row.rows, "diagnostic raw file row-count mismatch: " + row.file, errors)
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        errors.append("appendix reproduction failed: " + str(exc))

    from reproduce_crosscat import reproduce as reproduce_crosscat, check as check_crosscat
    try:
        tables, audits = reproduce_crosscat(root)
        require(check_crosscat(root, tables) == 2, "expected two CrossCat tables", errors)
        require(sum(row["valid_n"] for row in audits) == 9600, "expected 9600 valid CrossCat judgments", errors)
        require(len(tables["cell_results.csv"]) == 32, "expected 32 CrossCat cells", errors)
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        errors.append("CrossCat reproduction failed: " + str(exc))


def validate_manifest(root: Path, errors: list[str]) -> None:
    manifest = frame(root, "MANIFEST.csv")
    files = {
        path.relative_to(root).as_posix()
        for path in distributed_files(root)
        if path.is_file() and path.name not in {"MANIFEST.csv", "CHECKSUMS.sha256"}
    }
    require(manifest.path.is_unique, "manifest paths are not unique", errors)
    require(set(manifest.path) == files, "manifest does not cover exactly the distributed files", errors)
    for _, row in manifest.iterrows():
        path = root / row.path
        require(int(row.bytes) == path.stat().st_size, f"manifest size mismatch for {row.path}", errors)
        require(str(row.sha256) == sha256_file(path), f"manifest checksum mismatch for {row.path}", errors)

    checksum_path = root / "CHECKSUMS.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        require(path.exists(), f"checksummed file missing: {relative}", errors)
        if path.exists():
            require(sha256_file(path) == expected, f"checksum mismatch: {relative}", errors)


def validate_anonymity(root: Path, errors: list[str]) -> None:
    tokens = {
        b"/" + b"home" + b"/": "absolute home path",
        b"cincy" + b"nlp.services.ai.azure.com": "execution-specific endpoint",
        b"user" + b"_3D0": "provider account identifier",
    }
    secrets = (
        re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"(?i)(api[_-]?key|token)\s*[:=]\s*[\"'][A-Za-z0-9_-]{20,}[\"']"),
    )
    for path in distributed_files(root):
        if not path.is_file():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rb") as handle:
                carry = b""
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    data = carry + chunk
                    for token, label in tokens.items():
                        if token in data:
                            errors.append(f"{label} found in {path.relative_to(root)}")
                    for pattern in secrets:
                        if pattern.search(data):
                            errors.append(f"possible credential found in {path.relative_to(root)}")
                    carry = data[-256:]
        except OSError:
            continue


def validate_layout(root: Path, errors: list[str]) -> None:
    required = {
        "README.md", "ARTIFACT_MAP.md", "SCHEMAS.md", "PROVENANCE.md", ".gitignore",
        "MANIFEST.csv", "CHECKSUMS.sha256", "requirements.txt",
        "configs/paper_config.json", "code/validate_artifact.py", "code/reproduce_results.py",
        "prompts/coherence/coherence_prompt.txt", "prompts/generation_prompt_templates.json",
        "prompts/prompt_registry.json", "results/monolingual_all_88_cells.csv",
        "results/crosslingual_all_408_routes.csv", "results/comparison_level_results.csv",
        "results/explicit_prompt_all_44_cells.csv", "results/matched_literal_all_44_cells.csv",
        "evaluator_outputs/main_condition_manifest.csv",
        "evaluator_outputs/main/baseline_and_monolingual.csv.gz",
        "evaluator_outputs/main/crosslingual.csv.gz", "evaluator_outputs/main/random.csv.gz",
        "evaluator_outputs/main/geometry.csv.gz",
        "evaluator_outputs/explicit_prompt/condition_manifest.csv",
        "evaluator_outputs/explicit_prompt/explicit_prompt.csv.gz",
        "evaluator_outputs/matched_literal/condition_manifest.csv",
        "evaluator_outputs/matched_literal/matched_literal.csv.gz",
    }
    required.update({
        "code/reproduce_diagnostics.py", "diagnostics/README.md",
        "diagnostics/configs/evaluation_files.csv",
        "diagnostics/configs/meaning_matched.csv", "diagnostics/configs/strength_sweep.csv",
        "diagnostics/configs/alternative_negative.csv", "diagnostics/raw/quality_evaluated.csv.gz",
        "diagnostics/prompts/quality_judge.txt", "diagnostics/prompts/meaning_matched_generation.txt",
        "diagnostics/prompts/meaning_matched_screening.txt", "diagnostics/prompts/definitions.json",
        "diagnostics/construction/meaning_matched_selected_pairs.csv",
        "diagnostics/construction/meaning_matched_candidates.csv",
        "diagnostics/construction/meaning_matched_screening_summary.csv",
    })
    required.update({"code/reproduce_crosscat.py", "diagnostics/crosscat/README.md",
                     "diagnostics/crosscat/protocol.json", "diagnostics/crosscat/results/cell_results.csv",
                     "diagnostics/crosscat/results/model_summary.csv"})
    for model in ("qwen3_8b", "qwen3_32b", "llama3_1_8b", "ministral3_8b"):
        for kind in ("inputs", "responses"):
            required.add("diagnostics/crosscat/raw/" + model + "_" + kind + ".jsonl.gz")
    for name in ["quality_summary", "quality_differences", "quality_by_language_category",
                 "quality_response_counts",
                 "meaning_matched_results", "strength_sweep_results", "alternative_negative_cells",
                 "alternative_negative_summary"]:
        required.add("diagnostics/results/" + name + ".csv")
    registry_path = root / "diagnostics/configs/evaluation_files.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        required.update("diagnostics/" + name for name in registry.file)
    tbr = {path.relative_to(root).as_posix() for path in (root / "prompts/tbr").glob("*.txt")}
    require(len(tbr) == 22, "target-behavior prompt count is not 22", errors)
    required.update(tbr)
    actual = {
        path.relative_to(root).as_posix()
        for path in distributed_files(root)
        if path.is_file()
    }
    require(actual == required, f"package layout differs from the expected layout: extra={sorted(actual-required)}, missing={sorted(required-actual)}", errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    validate_layout(root, errors)
    if not errors:
        validate_results(root, errors)
        validate_outputs(root, errors)
        validate_diagnostics(root, errors)
        validate_manifest(root, errors)
        validate_anonymity(root, errors)
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("VALIDATION PASSED")
    files = list(distributed_files(root))
    print(json.dumps({"files": len(files), "bytes": sum(path.stat().st_size for path in files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
