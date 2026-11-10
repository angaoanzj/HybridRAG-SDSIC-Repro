#!/usr/bin/env python3
"""Recompute cutoff-specific claims from this small saved-record bundle.

Standard library only. Paths resolve relative to this file. No model, network,
retrieval or write operation is performed. The exported Boolean fields retain
the original saved measurements; this script does not re-annotate documents.
"""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / "review_claim_check.json").read_text())
    for name, expected in report["portable_files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    with (root / "small_minimal_evidence.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 300
    assert len({row["id"] for row in rows}) == 300
    assert all(row["dataset"] == "2wiki" and row["cue"] == "fewrel_exact" for row in rows)
    reference = [int(row["reference_FT_K50"]) for row in rows]
    variant = [int(row["variant_FT_K50"]) for row in rows]
    for row, left, right in zip(rows, reference, variant):
        assert left == int(int(row["reference_FT_first_complete_K"]) <= 50)
        assert right == int(int(row["variant_FT_first_complete_K"]) <= 50)
    discordant = sum(left != right for left, right in zip(reference, variant))
    assert sum(reference) == sum(variant) == 252
    assert discordant == 0
    assert sum(not int(row["ordered_document_ids_equal_K50"]) for row in rows) == 252
    assert sum(not int(row["document_id_sets_equal_K50"]) for row in rows) == 181
    deltas = []
    ever_changed = set()
    for cutoff in range(1, 51):
        pairs = [
            (int(row["reference_FT_first_complete_K"]) <= cutoff,
             int(row["variant_FT_first_complete_K"]) <= cutoff)
            for row in rows
        ]
        deltas.append(sum(right for left, right in pairs) - sum(left for left, right in pairs))
        ever_changed.update(row["id"] for row, pair in zip(rows, pairs) if pair[0] != pair[1])
    assert len(ever_changed) == 43
    assert min(deltas) == -5 and max(deltas) == 4
    with (root / "synonym_affected_pairs.csv").open(newline="") as handle:
        affected = list(csv.DictReader(handle))
    assert len(affected) == 4
    assert len({(row["dataset"], row["id"]) for row in affected}) == 1
    assert {int(row["K"]) for row in affected} == {10, 20, 30, 50}
    assert all(row["morph_input_key"] != row["syn_input_key"] for row in affected)
    assert all(row["morph_ordered_ids_sha256"] != row["syn_ordered_ids_sha256"] for row in affected)
    assert all(row["morph_prediction"] == row["syn_prediction"] for row in affected)
    assert all(float(row["morph_EM"]) == float(row["syn_EM"]) == 0 for row in affected)
    assert all(float(row["morph_F1"]) == float(row["syn_F1"]) == 0 for row in affected)
    synonym = report["synonyms"]
    assert synonym["accepted_synonym_keys"] == [["boy", "n"]]
    assert len(synonym["accepted_record"]) == 1
    assert synonym["accepted_record"][0]["seed"] == "sons"
    assert synonym["accepted_record"][0]["synonyms"] == ["boy"]
    print(json.dumps({
        "status": "PASS_SAVED_RECORD_CUTOFF_CHECK",
        "K50": {"N": 300, "reference_complete": sum(reference),
                "variant_complete": sum(variant), "discordant_indicators": discordant,
                "different_document_orders": 252, "different_document_sets": 181},
        "all_K1_50": {"questions_changed_at_some_K": len(ever_changed),
                      "minimum_delta_pp": min(deltas) / 3, "maximum_delta_pp": max(deltas) / 3},
        "synonym": {"seed": "sons", "accepted_term": "boy", "affected_question_cutoff_pairs": 4,
                    "single_arm_outputs": 8, "within_pair_predictions_identical": True},
        "scope": "Saved measurements and exported comparisons; no model or retrieval execution"
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
