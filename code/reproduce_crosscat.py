#!/usr/bin/env python3
"""Recompute the CrossCat diagnostic from saved text and full judge responses."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

CATEGORIES = ("idiom", "metaphor", "irony", "sarcasm")
LANGUAGES = ("en", "zh")
CONDITIONS = ("unsteered", "omean", "crosscat")
LEGACY = {"unsteered": "unsteered", "matched_omean": "omean", "mismatched_omean": "crosscat"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_response(payload):
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    if choice.get("finish_reason") != "stop" or message.get("reasoning_content") not in (None, ""):
        return None
    lines = re.findall(r"(?im)^[ \t]*Label[ \t]*:[^\r\n]*", content)
    labels = re.findall(r"(?m)^[ \t]*Label[ \t]*:[ \t]*(YES|NO)[ \t]*\r?$", content)
    return int(labels[0] == "YES") if len(lines) == len(labels) == 1 else None


def validate_construction(model):
    availability = {lang: set(CATEGORIES) - ({"sarcasm"} if lang in ("es", "de") else set())
                    for lang in ("en", "zh", "bn", "es", "it", "de")}
    seen = set()
    for cell in model["cells"]:
        language, category = cell["target_language"], cell["target_category"]
        require((language, category) not in seen, "duplicate construction cell")
        seen.add((language, category))
        pool = {lang for lang, cats in availability.items() if lang != language and category in cats}
        require(set(cell["source_languages"]) == pool, "CrossCat source pool differs from OMean availability")
        terms = {(term["lang"], term["category"]): term["weight"] for term in cell["terms"]}
        expected = {(lang, cat) for lang in pool for cat in availability[lang] - {category}}
        require(len(terms) == len(cell["terms"]) and set(terms) == expected, "incorrect CrossCat category membership")
        totals = []
        for lang in pool:
            weights = [weight for (source, _), weight in terms.items() if source == lang]
            require(all(weight > 0 for weight in weights) and len(set(weights)) == 1, "unequal categories within language")
            totals.append(sum(weights))
        require(len(set(totals)) == 1, "unequal source-language weights")
        require(cell["norm_calibration"]["num_texts"] == 500, "incorrect norm-calibration sample count")
        require(cell["generation_settings"]["ratio"] == 1, "incorrect intervention ratio")
        require(abs(cell["control_vector_norm"] - 1) < 1e-5, "CrossCat vector was not normalized")
    require(seen == {(lang, cat) for lang in LANGUAGES for cat in CATEGORIES}, "incorrect construction cells")


def reproduce(root):
    base = root / "diagnostics/crosscat"
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    require(len(protocol["models"]) == 4, "expected four models")
    cells, summaries, audits = [], [], []
    for model in protocol["models"]:
        validate_construction(model)
        model_id = model["model_id"]
        jobs = read_jsonl(base / "raw" / (model_id + "_inputs.jsonl.gz"))
        events = read_jsonl(base / "raw" / (model_id + "_responses.jsonl.gz"))
        require(len(jobs) == 2400, model_id + ": expected 2400 inputs")
        by_id = {job["job_id"]: job for job in jobs}
        require(len(by_id) == len(jobs), "duplicate job IDs")
        identities = {}
        grouped = collections.defaultdict(dict)
        cell_protocol = {(cell["target_language"], cell["target_category"]): cell for cell in model["cells"]}
        baseline_rows = {}
        for job in jobs:
            require(LEGACY.get(job["legacy_condition"]) == job["condition"], "condition mapping mismatch")
            lang, cat = job["target_language"], job["target_category"]
            require(lang in LANGUAGES and cat in CATEGORIES and job["condition"] in CONDITIONS, "unexpected cell")
            require(job["text_sha256"] == sha(job["text"].encode("utf-8")), "changed generated text")
            require(job["source_file_id"] in model["source_files"], "unknown source file ID")
            require(0 <= job["original_row_index"] < 100, "unexpected original row index")
            require(job["source_file_id"] == cell_protocol[(lang, cat)]["source_file_ids"][job["condition"]], "source file differs from cell protocol")
            if job["condition"] == "unsteered":
                baseline_key = (lang, job["original_row_index"])
                value = (job["source_file_id"], job["simple_prompt"], job["text"])
                require(baseline_rows.get(baseline_key, value) == value, "unsteered output differs across category evaluations")
                baseline_rows[baseline_key] = value
            template_info = model["judge"]["prompt_metadata"][lang + "_" + cat]
            template = (root / template_info["public_path"]).read_text(encoding="utf-8")
            require(sha(template.encode("utf-8")) == template_info["prompt_sha256"], "changed judge template")
            prompt = template.replace("{input}", job["text"])
            request = {**model["judge"]["request_settings"], "messages": [
                {"role": "system", "content": model["judge"]["system_message"]},
                {"role": "user", "content": prompt}]}
            identities[job["job_id"]] = {
                "input_sha256": job["original_input_sha256"], "text_sha256": job["text_sha256"],
                "prompt_sha256": sha(prompt.encode("utf-8")), "request_sha256": sha(canonical(request)),
                "template_sha256": template_info["prompt_sha256"],
                "judge_manifest_sha256": model["original_judge_manifest_sha256"],
            }
            group = grouped[(lang, cat, job["condition"])]
            require(job["simple_prompt"] not in group, "duplicate prompt in condition")
            group[job["simple_prompt"]] = job
        started, finished, labels, returned_models = {}, {}, {}, set()
        usage = collections.Counter()
        for event in events:
            job_id = event["job_id"]
            require(job_id in by_id, "response has unknown job ID")
            for field, expected in identities[job_id].items():
                require(event.get(field) == expected, "response identity mismatch: " + field)
            key = (job_id, event["run_id"], event["attempt"])
            if event["event"] == "attempt_started":
                require(key not in started, "duplicate attempt start")
                require(event["request_settings"] == model["judge"]["request_settings"], "request settings changed")
                started[key] = event
            elif event["event"] == "attempt_finished":
                require(key in started and key not in finished, "missing start or duplicate finish")
                finished[key] = event
                parsed = parse_response(event.get("response", {}))
                require((event["status"] == "valid") == (parsed is not None), "stored validity differs from raw response")
                if parsed is not None:
                    require(parsed == event["label"], "stored label differs from raw response")
                    require(job_id not in labels, "multiple valid responses for one job")
                    labels[job_id] = parsed
                response = event.get("response", {})
                require(event.get("response_id") == response.get("id"), "response ID mismatch")
                require(event.get("usage") == response.get("usage"), "usage mismatch")
                returned_models.add(response.get("model"))
                usage.update({key: value for key, value in (response.get("usage") or {}).items()
                              if isinstance(value, (int, float))})
            else:
                raise ValueError("unknown response event")
        require(set(started) == set(finished), "interrupted attempts remain")
        require(len(labels) == 2400, model_id + ": missing or invalid judgments remain")
        model_cells = []
        for lang in LANGUAGES:
            for cat in CATEGORIES:
                groups = [grouped[(lang, cat, condition)] for condition in CONDITIONS]
                common = set.intersection(*(set(group) for group in groups))
                require(len(common) == 100 and all(len(group) == 100 for group in groups), "not 100 identical prompts")
                for prompt in common:
                    require(len({group[prompt]["original_row_index"] for group in groups}) == 1, "prompt row alignment differs")
                row = {"model": model_id, "target_language": lang, "target_category": cat, "common_valid_n": 100}
                for condition, group in zip(CONDITIONS, groups):
                    positive = sum(labels[group[prompt]["job_id"]] for prompt in common)
                    row[condition + "_positive_n"] = positive
                    row[condition + "_tbr_pct"] = 100 * positive / len(common)
                row["omean_gain_pp"] = row["omean_tbr_pct"] - row["unsteered_tbr_pct"]
                row["crosscat_gain_pp"] = row["crosscat_tbr_pct"] - row["unsteered_tbr_pct"]
                row["omean_minus_crosscat_pp"] = row["omean_tbr_pct"] - row["crosscat_tbr_pct"]
                model_cells.append(row)
        cells.extend(model_cells)
        summaries.append(macro(model_id, model_cells))
        audits.append({"model": model_id, "input_n": len(jobs), "valid_n": len(labels),
                       "attempt_n": len(started), "returned_models": sorted(returned_models), "usage": dict(usage)})
    summaries.append(macro("all_models", cells))
    return {"cell_results.csv": cells, "model_summary.csv": summaries}, audits


def macro(model_id, rows):
    result = {"model": model_id, "cell_n": len(rows)}
    for key in ("unsteered_tbr_pct", "omean_tbr_pct", "crosscat_tbr_pct", "omean_gain_pp", "crosscat_gain_pp", "omean_minus_crosscat_pp"):
        result[key] = sum(row[key] for row in rows) / len(rows)
    return result


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def check(root, tables):
    for name, rows in tables.items():
        require((root / "diagnostics/crosscat/results" / name).read_bytes() == csv_bytes(rows), "reference mismatch: " + name)
    return len(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    tables, audits = reproduce(root)
    if args.check:
        check(root, tables)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            (args.output_dir / name).write_bytes(csv_bytes(rows))
        (args.output_dir / "audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"valid_judgments": sum(row["valid_n"] for row in audits), "cells": len(tables["cell_results.csv"]),
                      "reference_tables_checked": len(tables) if args.check else 0, "summary": tables["model_summary.csv"]}, indent=2))


if __name__ == "__main__":
    main()
