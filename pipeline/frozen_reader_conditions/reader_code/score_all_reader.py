"""Score the frozen full-cohort reader run after every prediction is fixed.

Answer metrics call the pinned original benchmark functions unchanged.  The
input map expands strict identical-input reuse into the complete condition grid.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import string
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np


DATASETS = ("2wiki", "hotpotqa")
CAPS = (10, 20, 30, 50)
OFFICIAL_FILES = {
    "2wiki": "2wikimultihop_evaluate.py",
    "hotpotqa": "hotpot_evaluate_v1.py",
}
ANSWER_FUNCTIONS = ("normalize_answer", "f1_score", "exact_match_score")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def json_lines(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank record at {path}:{number}")
            yield number, json.loads(line)


def inside(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"File escapes the run directory: {relative}")
    return path


def load_official_metrics(path, expected_sha256):
    """Extract only the original three answer functions, without editing them."""
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError(f"Official evaluator hash mismatch: {path}")
    tree = ast.parse(path.read_text(), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ANSWER_FUNCTIONS]
    if {node.name for node in functions} != set(ANSWER_FUNCTIONS):
        raise ValueError(f"Official answer functions are incomplete: {path}")
    namespace = {"re": re, "string": string, "Counter": Counter}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_stats(path):
    spec = importlib.util.spec_from_file_location("frozen_reader_paired_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_gate(root, plan_path):
    """Complete output and frozen-file checks precede loading answer labels."""
    plan = read_json(plan_path)
    if plan.get("status") != "frozen_before_new_reader_generation":
        raise ValueError("Analysis plan was not frozen before new generation")
    if (plan.get("datasets") != list(DATASETS) or plan.get("K_values") != list(CAPS)
            or plan.get("n_per_dataset") != 300):
        raise ValueError("Unexpected cohort or K configuration")
    policies = plan.get("policies", [])
    if len(policies) != 19 or len(set(policies)) != 19:
        raise ValueError("Expected exactly 19 unique frozen policies")
    if len(plan.get("comparisons", [])) != 7:
        raise ValueError("Expected the seven frozen paired comparison definitions")
    comparison_names = set()
    for family, name, left, right in plan["comparisons"]:
        if left not in policies or right not in policies or left == right:
            raise ValueError(f"Invalid paired condition definition: {name}")
        comparison_names.add((family, name))
    if len(comparison_names) != 7:
        raise ValueError("Repeated paired comparison definitions")
    if plan.get("seed") != 2026110602 or plan.get("bootstrap_resamples") != 5000:
        raise ValueError("Unexpected frozen bootstrap configuration")

    marker = read_json(root / "generation_complete.json")
    if marker.get("status") != "complete":
        raise ValueError("Generation has not completed successfully")
    if marker.get("analysis_plan_sha256") != sha256(plan_path):
        raise ValueError("Generation marker belongs to another analysis plan")
    predictions = inside(root, marker.get("predictions_path", ""))
    if sha256(predictions) != marker.get("predictions_sha256"):
        raise ValueError("Generated predictions changed after completion")

    required = {
        "inputs/group_input_map.jsonl.gz",
        "inputs/unique_inputs.jsonl.gz",
        "inputs/history_selected_predictions.jsonl.gz",
        "labels/2wiki.json", "labels/hotpotqa.json",
        "official_answer_scoring/source_receipt.json",
        "official_answer_scoring/2wikimultihop_evaluate.py",
        "official_answer_scoring/hotpot_evaluate_v1.py",
        "code/paired_stats.py", "code/score_all_reader.py",
    }
    frozen = plan.get("files", {})
    if not required.issubset(frozen):
        raise ValueError(f"Required files missing from freeze: {sorted(required - set(frozen))}")
    for relative, expected in frozen.items():
        if sha256(inside(root, relative)) != expected:
            raise ValueError(f"Frozen file changed: {relative}")
    if sha256(__file__) != frozen["code/score_all_reader.py"]:
        raise ValueError("Executing scorer differs from the frozen scorer")
    return plan, marker, predictions


def load_input_records(root):
    unique = {}
    for number, row in json_lines(root / "inputs/unique_inputs.jsonl.gz"):
        key = row["input_key"]
        if key in unique:
            raise ValueError(f"Duplicate unique input at row {number}: {key}")
        messages_hash = hashlib.sha256(json.dumps(
            row["messages"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if messages_hash != row["prompt_sha256"]:
            raise ValueError(f"Actual message hash mismatch: {key}")
        meta = row["meta"]
        documents = meta["documents"]
        selected = row["selected_doc_ids"]
        if len(selected) != len(set(selected)):
            raise ValueError(f"Repeated selected document ID: {key}")
        if meta["retained_doc_count"] != len(selected):
            raise ValueError(f"Raw context count mismatch: {key}")
        if meta["reader_doc_count"] != len(documents):
            raise ValueError(f"Reader context count mismatch: {key}")
        removed = meta["budget_removed_doc_count"]
        if len(documents) + removed != len(selected):
            raise ValueError(f"Reader budget removal mismatch: {key}")
        if [document["doc_id"] for document in documents] != selected[:len(documents)]:
            raise ValueError(f"Reader document ordering mismatch: {key}")
        tokens = meta["tokenizer_input_tokens"]
        if not isinstance(tokens, int) or not 50 < tokens <= 24512:
            raise ValueError(f"Invalid input token count: {key}")
        if meta.get("prompt_sha256") != messages_hash:
            raise ValueError(f"Metadata message hash mismatch: {key}")
        unique[key] = row
    return unique


def validate_groups(groups, unique, plan):
    seen = set()
    roster = defaultdict(set)
    condition_ids = defaultdict(set)
    referenced = set()
    for group in groups:
        dataset, question_id = group["dataset"], group["id"]
        policy, cap = group["policy"], group["K"]
        key = (dataset, question_id, policy, cap)
        if key in seen:
            raise ValueError(f"Duplicate condition group: {key}")
        if (dataset not in plan["datasets"] or policy not in plan["policies"]
                or cap not in plan["K_values"]):
            raise ValueError(f"Unplanned condition group: {key}")
        input_key = group["input_key"]
        if input_key not in unique:
            raise ValueError(f"Missing reader input: {input_key}")
        row = unique[input_key]
        if row["dataset"] != dataset or row["id"] != question_id:
            raise ValueError(f"Reader input belongs to a different question: {key}")
        if (group["prompt_sha256"] != row["prompt_sha256"]
                or group["raw_context_sha256"] != row["raw_context_sha256"]
                or group["selected_doc_ids"] != row["selected_doc_ids"]):
            raise ValueError(f"Reader context does not match condition: {key}")
        if len(group["selected_doc_ids"]) != cap:
            raise ValueError(f"Condition does not contain its requested K: {key}")
        seen.add(key)
        referenced.add(input_key)
        roster[dataset].add(question_id)
        condition_ids[(dataset, policy, cap)].add(question_id)
    if referenced != set(unique):
        raise ValueError("Unique input file contains unreferenced inputs")
    expected_conditions = {(dataset, policy, cap) for dataset in plan["datasets"]
                           for policy in plan["policies"] for cap in plan["K_values"]}
    if set(condition_ids) != expected_conditions:
        raise ValueError("Condition grid is incomplete")
    for dataset, ids in roster.items():
        if len(ids) != plan["n_per_dataset"]:
            raise ValueError(f"Question roster size mismatch: {dataset}")
    for (dataset, policy, cap), ids in condition_ids.items():
        if ids != roster[dataset]:
            raise ValueError(f"Condition has a different question cohort: {dataset}/{policy}/{cap}")
    expected_groups = len(expected_conditions) * plan["n_per_dataset"]
    if len(groups) != expected_groups:
        raise ValueError("Total group count mismatch")
    return roster


def validate_prediction(row, unique_row, source):
    key = unique_row["input_key"]
    if row.get("failed") or not isinstance(row.get("prediction"), str):
        raise ValueError(f"Failed or missing prediction: {key}")
    if row.get("prompt_sha256") != unique_row["prompt_sha256"]:
        raise ValueError(f"Prediction prompt mismatch: {key}")
    if row.get("finish_reason") not in ("stop", "length"):
        raise ValueError(f"Unexpected finish reason for {key}: {row.get('finish_reason')}")
    usage = row.get("usage")
    if not isinstance(usage, dict):
        raise ValueError(f"Missing original token usage: {key}")
    if usage.get("prompt_tokens") != unique_row["meta"]["tokenizer_input_tokens"]:
        raise ValueError(f"Actual generation input token count differs: {key}")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 0:
        raise ValueError(f"Invalid completion token count: {key}")
    if source == "historical" and (not row.get("source_file")
                                    or not isinstance(row.get("source_line"), int)
                                    or row["source_line"] < 1):
        raise ValueError(f"Historical prediction lacks its fixed source: {key}")


def load_predictions(root, predictions_path, unique, groups, marker):
    predictions = {}
    counts = Counter()
    sources = [("historical", root / "inputs/history_selected_predictions.jsonl.gz"),
               ("new_generation", predictions_path)]
    for source, path in sources:
        for number, row in json_lines(path):
            key = row["input_key"]
            if key not in unique:
                raise ValueError(f"Prediction refers to an unplanned input: {key}")
            if key in predictions:
                raise ValueError(f"Multiple predictions selected for the same input: {key}")
            validate_prediction(row, unique[key], source)
            predictions[key] = {**row, "prediction_source": source,
                                "prediction_record_file": str(path),
                                "prediction_record_line": number}
            counts[source] += 1
    if set(predictions) != set(unique):
        raise ValueError(f"Missing fixed predictions: {len(set(unique) - set(predictions))}")
    for group in groups:
        source = predictions[group["input_key"]]["prediction_source"]
        if bool(group["historical_match"]) != (source == "historical"):
            raise ValueError("Historical reuse differs from the pre-generation input map")
    if (marker.get("unique_inputs") != len(unique)
            or marker.get("generated_unique") != counts["new_generation"]
            or marker.get("history_unique") != counts["historical"]):
        raise ValueError("Completion counts differ from fixed prediction records")
    return predictions, dict(counts)


def load_answer_labels(root, roster):
    labels = {}
    for dataset in DATASETS:
        document = read_json(root / "labels" / f"{dataset}.json")
        rows = document["labels"]
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or set(by_id) != roster[dataset]:
            raise ValueError(f"Answer labels do not match the paired question cohort: {dataset}")
        if any(not isinstance(row.get("answer"), str) for row in rows):
            raise ValueError(f"Expected one canonical answer string per question: {dataset}")
        labels[dataset] = by_id
    return labels


def score_groups(groups, unique, predictions, labels, official):
    rows = []
    for group in groups:
        dataset, question_id = group["dataset"], group["id"]
        input_row = unique[group["input_key"]]
        prediction = predictions[group["input_key"]]
        answer = labels[dataset][question_id]["answer"]
        functions = official[dataset]
        em = float(functions["exact_match_score"](prediction["prediction"], answer))
        f1, precision, recall = functions["f1_score"](prediction["prediction"], answer)
        meta = input_row["meta"]
        rows.append({
            "dataset": dataset, "id": question_id, "policy": group["policy"],
            "K": group["K"], "input_key": group["input_key"],
            "question": input_row["question"], "prompt_sha256": input_row["prompt_sha256"],
            "raw_context_sha256": input_row["raw_context_sha256"],
            "prediction": prediction["prediction"], "canonical_answer": answer,
            "EM": em, "F1": float(f1), "precision": float(precision), "recall": float(recall),
            "prediction_source": prediction["prediction_source"],
            "history_source": prediction.get("history_source"),
            "historical_source_file": prediction.get("source_file"),
            "historical_source_line": prediction.get("source_line"),
            "prediction_record_file": prediction["prediction_record_file"],
            "prediction_record_line": prediction["prediction_record_line"],
            "input_tokens": prediction["usage"]["prompt_tokens"],
            "output_tokens": prediction["usage"]["completion_tokens"],
            "retained_doc_count": meta["retained_doc_count"],
            "reader_doc_count": meta["reader_doc_count"],
            "budget_removed_doc_count": meta["budget_removed_doc_count"],
            "body_truncated_doc_count": meta["body_truncated_doc_count"],
            "finish_reason": prediction["finish_reason"],
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["policy"], row["K"], row["id"]))


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row["dataset"], row["policy"], row["K"])].append(row)
    table = []
    for (dataset, policy, cap), values in sorted(cells.items()):
        n = len(values)
        em_count = int(sum(row["EM"] for row in values))
        table.append({
            "dataset": dataset, "policy": policy, "K": cap, "n": n,
            "EM_count": em_count, "EM_percent": 100 * em_count / n,
            "F1_points": 100 * sum(row["F1"] for row in values) / n,
            "mean_input_tokens": sum(row["input_tokens"] for row in values) / n,
            "mean_output_tokens": sum(row["output_tokens"] for row in values) / n,
            "mean_retained_doc_count": sum(row["retained_doc_count"] for row in values) / n,
            "mean_reader_doc_count": sum(row["reader_doc_count"] for row in values) / n,
            "mean_body_truncated_doc_count": sum(row["body_truncated_doc_count"] for row in values) / n,
            "budget_removed_question_count": sum(row["budget_removed_doc_count"] > 0 for row in values),
            "body_truncated_question_count": sum(row["body_truncated_doc_count"] > 0 for row in values),
            "length_finish_count": sum(row["finish_reason"] == "length" for row in values),
            "empty_answer_count": sum(not row["prediction"].strip() for row in values),
            "historical_answer_groups": sum(row["prediction_source"] == "historical" for row in values),
            "new_answer_groups": sum(row["prediction_source"] == "new_generation" for row in values),
            "unique_inputs_in_condition": len({row["input_key"] for row in values}),
        })
    return table


def paired_contrasts(rows, labels, roster, plan, stats):
    lookup = {(row["dataset"], row["id"], row["policy"], row["K"]): row for row in rows}
    results = []
    for dataset in plan["datasets"]:
        ids = sorted(roster[dataset])
        components = stats.support_components(labels[dataset], ids)
        for cap in plan["K_values"]:
            for family, name, left, right in plan["comparisons"]:
                result = {"dataset": dataset, "K": cap, "family": family,
                          "contrast": name, "left": left, "right": right, "metrics": {}}
                for metric in ("EM", "F1"):
                    first = [lookup[(dataset, question_id, left, cap)][metric] for question_id in ids]
                    second = [lookup[(dataset, question_id, right, cap)][metric] for question_id in ids]
                    interval = stats.paired_interval(
                        first, second, components, seed=plan["seed"],
                        resamples=plan["bootstrap_resamples"])
                    interval["difference_points"] = 100 * interval["difference"]
                    interval["question_ci95_points"] = [100 * value for value in interval["question_ci95"]]
                    interval["cluster_ci95_points"] = [100 * value for value in interval["cluster_ci95"]]
                    result["metrics"][metric] = interval
                em = result["metrics"]["EM"]
                result["EM_exact_mcnemar_p"] = stats.exact_mcnemar(em["wins"], em["losses"])
                results.append(result)
    adjusted = stats.holm_adjust([row["EM_exact_mcnemar_p"] for row in results])
    for result, pvalue in zip(results, adjusted):
        result["EM_holm_adjusted_p"] = pvalue
        result["EM_holm_family_size"] = len(results)
        result["F1_inference"] = "Secondary pointwise paired and support-component bootstrap intervals; no F1 p-value claimed."
    return results


def write_outputs(output, rows, result):
    if output.exists():
        raise ValueError(f"Refusing to overwrite previous scoring output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".preparing_", dir=output.parent))
    try:
        with gzip.open(temporary / "per_question_answers.jsonl.gz", "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (temporary / "all_conditions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["table"][0]))
            writer.writeheader()
            writer.writerows(result["table"])
        (temporary / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (temporary / "paired_contrasts.json").write_text(json.dumps(result["contrasts"], ensure_ascii=False, indent=2) + "\n")
        manifest = {file.name: sha256(file) for file in sorted(temporary.iterdir())}
        (temporary / "score_receipt.json").write_text(json.dumps({
            "status": "complete", "rows": len(rows), "cells": len(result["table"]),
            "files": manifest, "scorer_sha256": sha256(__file__),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--stats-module", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = args.analysis_plan or root / "analysis_plan.json"
    plan, marker, predictions_path = verify_gate(root, plan_path)
    unique = load_input_records(root)
    groups = [row for _, row in json_lines(root / "inputs/group_input_map.jsonl.gz")]
    roster = validate_groups(groups, unique, plan)
    predictions, source_counts = load_predictions(root, predictions_path, unique, groups, marker)
    if any(row.get("model") != plan["request_params"]["model"] for row in predictions.values()):
        raise ValueError("A prediction was returned by a different served model")
    gate_passed_at = datetime.now(timezone.utc).isoformat()

    # Answer annotations are opened only after all input/output checks above pass.
    labels = load_answer_labels(root, roster)
    official_root = root / "official_answer_scoring"
    receipt = read_json(official_root / "source_receipt.json")
    official_hashes = {row["local_file"]: row["sha256"] for row in receipt["sources"]}
    official = {dataset: load_official_metrics(official_root / name, official_hashes[name])
                for dataset, name in OFFICIAL_FILES.items()}
    if args.stats_module and sha256(args.stats_module) != plan["files"]["code/paired_stats.py"]:
        raise ValueError("Alternate statistics module differs from the frozen module")
    stats = load_stats(args.stats_module or root / "code/paired_stats.py")
    rows = score_groups(groups, unique, predictions, labels, official)
    result = {
        "status": "complete", "scope": "Paired original historical 20261105 cohorts, 300 questions per dataset; no independent confirmation claim.",
        "answer_metric": "Canonical-reference Answer EM and token F1 from the pinned original official 2Wiki/Hotpot scripts; no answer-ID alias expansion; uniform question macro averages.",
        "annotation_gate_passed_at_utc": gate_passed_at,
        "analysis_plan_sha256": sha256(plan_path),
        "generation_complete_sha256": sha256(root / "generation_complete.json"),
        "groups": len(groups), "unique_inputs": len(unique),
        "unique_prediction_sources": source_counts,
        "table": summarize(rows),
        "contrasts": paired_contrasts(rows, labels, roster, plan, stats),
        "reused_prediction_note": "Historical rows require exact question, ordered full raw context, actual messages, token counts, model/tokenizer/template and decoding configuration. Shared input keys share one frozen prediction across current conditions.",
        "latency_note": "Generation service wall time under shared batching is not an end-to-end latency experiment.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output or root / "analysis/scored_answers"
    write_outputs(output, rows, result)
    print(json.dumps({"status": "complete", "groups": len(rows), "cells": len(result["table"]),
                      "contrasts": len(result["contrasts"]), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
