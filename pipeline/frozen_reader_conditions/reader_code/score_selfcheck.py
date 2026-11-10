"""Small deterministic checks for official metrics and fail-closed joins."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile

import score_all_reader as scorer


def expect_error(function, *args):
    try:
        function(*args)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"Invalid fixture unexpectedly passed: {function.__name__}")


def save_lines(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def make_fixture():
    plan = {"datasets": ["2wiki"], "policies": ["left", "right"],
            "K_values": [10], "n_per_dataset": 2}
    unique = {}
    groups = []
    for index in range(2):
        question_id = f"q{index}"
        input_key = f"input{index}"
        prompt_hash = f"prompt{index}"
        document_ids = [f"doc{i}" for i in range(10)]
        unique[input_key] = {
            "input_key": input_key, "dataset": "2wiki", "id": question_id,
            "question": "Synthetic fixture", "prompt_sha256": prompt_hash,
            "raw_context_sha256": f"raw{index}", "selected_doc_ids": document_ids,
            "meta": {"tokenizer_input_tokens": 100, "retained_doc_count": 10,
                     "reader_doc_count": 10, "budget_removed_doc_count": 0,
                     "body_truncated_doc_count": 0},
        }
        for policy in plan["policies"]:
            groups.append({"dataset": "2wiki", "id": question_id,
                           "policy": policy, "K": 10, "input_key": input_key,
                           "prompt_sha256": prompt_hash, "raw_context_sha256": f"raw{index}",
                           "selected_doc_ids": document_ids, "historical_match": index == 0})
    historical = {"input_key": "input0", "prompt_sha256": "prompt0",
                  "prediction": "The Alice", "finish_reason": "stop",
                  "usage": {"prompt_tokens": 100, "completion_tokens": 3},
                  "source_file": "fixed_history.jsonl", "source_line": 1}
    generated = {"input_key": "input1", "prompt_sha256": "prompt1",
                 "prediction": "wrong", "finish_reason": "stop",
                 "usage": {"prompt_tokens": 100, "completion_tokens": 2}}
    marker = {"unique_inputs": 2, "generated_unique": 1, "history_unique": 1}
    return plan, unique, groups, historical, generated, marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--stats-module", type=Path, required=True)
    args = parser.parse_args()
    receipt = scorer.read_json(args.official_root / "source_receipt.json")
    hashes = {row["local_file"]: row["sha256"] for row in receipt["sources"]}
    official = {dataset: scorer.load_official_metrics(args.official_root / filename, hashes[filename])
                for dataset, filename in scorer.OFFICIAL_FILES.items()}
    cases = [
        ("The, Cat!", "cat", 1.0, 1.0),
        ("YES", "yes", 1.0, 1.0),
        ("yes", "no", 0.0, 0.0),
        ("yes indeed", "yes", 0.0, 0.0),
        ("unknown", "Alice", 0.0, 0.0),
        ("", "", 1.0, 0.0),
        ("", "Alice", 0.0, 0.0),
        ("red red blue", "red blue", 0.0, 0.8),
        ("Jean-Rodolphe Perronet", "Jean Rodolphe Perronet", 0.0, 0.4),
        ("  A\tB  ", "b", 1.0, 1.0),
        ("Renáta", "renata", 0.0, 0.0),
        ("noanswer", "noanswer", 1.0, 1.0),
    ]
    assertions = 0
    for functions in official.values():
        for prediction, answer, expected_em, expected_f1 in cases:
            assert float(functions["exact_match_score"](prediction, answer)) == expected_em
            assert abs(functions["f1_score"](prediction, answer)[0] - expected_f1) < 1e-12
            assertions += 2

    plan, unique, groups, historical, generated, marker = make_fixture()
    roster = scorer.validate_groups(groups, unique, plan)
    assert roster == {"2wiki": {"q0", "q1"}}
    expect_error(scorer.validate_groups, groups + [groups[0]], unique, plan)
    expect_error(scorer.validate_groups, groups[:-1], unique, plan)
    wrong = copy.deepcopy(groups)
    wrong[0]["selected_doc_ids"] = list(reversed(wrong[0]["selected_doc_ids"]))
    expect_error(scorer.validate_groups, wrong, unique, plan)
    wrong = copy.deepcopy(groups)
    wrong[0]["id"] = "different_question"
    expect_error(scorer.validate_groups, wrong, unique, plan)
    wrong_prediction = copy.deepcopy(generated)
    wrong_prediction["usage"]["prompt_tokens"] = 99
    expect_error(scorer.validate_prediction, wrong_prediction, unique["input1"], "new_generation")
    wrong_prediction = {**generated, "failed": True}
    expect_error(scorer.validate_prediction, wrong_prediction, unique["input1"], "new_generation")
    assertions += 7

    with tempfile.TemporaryDirectory(prefix="reader_score_selfcheck_") as temporary:
        root = Path(temporary)
        (root / "inputs").mkdir()
        history_path = root / "inputs/history_selected_predictions.jsonl.gz"
        predictions_path = root / "predictions.jsonl.gz"
        save_lines(history_path, [historical])
        save_lines(predictions_path, [generated])
        predictions, counts = scorer.load_predictions(root, predictions_path, unique, groups, marker)
        assert counts == {"historical": 1, "new_generation": 1}
        labels = {"2wiki": {"q0": {"answer": "Alice", "gold_titles": ["T0"]},
                             "q1": {"answer": "Bob", "gold_titles": ["T1"]}}}
        rows = scorer.score_groups(groups, unique, predictions, labels, official)
        cells = scorer.summarize(rows)
        assert len(rows) == 4 and len(cells) == 2
        assert all(cell["EM_count"] == 1 and cell["F1_points"] == 50.0 for cell in cells)
        assert all(cell["historical_answer_groups"] == 1 and cell["new_answer_groups"] == 1 for cell in cells)
        save_lines(predictions_path, [generated, generated])
        expect_error(scorer.load_predictions, root, predictions_path, unique, groups, marker)
        save_lines(predictions_path, [])
        expect_error(scorer.load_predictions, root, predictions_path, unique, groups, marker)
        save_lines(predictions_path, [generated])
        wrong = copy.deepcopy(groups)
        wrong[0]["historical_match"] = False
        expect_error(scorer.load_predictions, root, predictions_path, unique, wrong, marker)
        assertions += 7

    stats = scorer.load_stats(args.stats_module)
    assert stats.exact_mcnemar(0, 0) == 1.0
    assert stats.exact_mcnemar(4, 0) == 0.125
    assert stats.holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    interval = stats.paired_interval([1, 1, 1], [0, 0, 0], [[0], [1], [2]], resamples=20)
    assert interval["difference"] == 1.0 and interval["question_ci95"] == [1.0, 1.0]
    contrast_plan = {**plan, "comparisons": [["fixture", "left_minus_right", "left", "right"]],
                     "seed": 2026110602, "bootstrap_resamples": 20}
    contrasts = scorer.paired_contrasts(rows, labels, roster, contrast_plan, stats)
    assert len(contrasts) == 1 and contrasts[0]["EM_holm_adjusted_p"] == 1.0
    assert contrasts[0]["metrics"]["F1"]["difference_points"] == 0.0
    assert "F1_p" not in contrasts[0] and contrasts[0]["EM_holm_family_size"] == 1
    assertions += 7
    print(json.dumps({"status": "passed", "assertions": assertions,
                      "official_metric_fixtures_per_dataset": len(cases),
                      "synthetic_groups": 4, "answer_labels_from_real_cohort_read": False,
                      "scorer_sha256": scorer.sha256(Path(scorer.__file__))}, indent=2))


if __name__ == "__main__":
    main()
