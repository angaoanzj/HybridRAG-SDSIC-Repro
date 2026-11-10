"""Prepared AutoDL experiment. No results exist until this script actually runs.

Keeps the original 2,417 MuSiQue questions, corpus, model, score and fusion.
No label is passed to the retriever or query selector. No network downloads.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
K_VALUES = (10, 20, 30, 50)
ARMS = {
    "RRF200": {"depths": [200], "modes": []},
    "single100": {"depths": [100, 100], "modes": ["new"]},
    "stop100_50": {"depths": [100, 50], "modes": ["new"]},
    "iter2_100_50_50": {"depths": [100, 50, 50], "modes": ["new", "new"]},
    "repeat2_100_50_50": {"depths": [100, 50, 50], "modes": ["new", "repeat"]},
    "stop50_50": {"depths": [50, 50], "modes": ["new"]},
    "stop50_50_50": {"depths": [50, 50, 50], "modes": ["new", "new"]},
    "iter3_50_50_50_50": {"depths": [50, 50, 50, 50], "modes": ["new", "new", "new"]},
    "repeat3_50_50_50_50": {"depths": [50, 50, 50, 50], "modes": ["new", "new", "repeat"]},
}
COMPARISONS = {
    "iter2_100_50_50-minus-single100": "Same D cap; combines iteration with depth reallocation.",
    "iter2_100_50_50-minus-stop100_50": "Same executed prefix; incremental new q2 including additional calls/returns.",
    "repeat2_100_50_50-minus-stop100_50": "Same prefix; repeated q1 votes without new candidates.",
    "iter2_100_50_50-minus-repeat2_100_50_50": "New versus repeated q2; realized budget may differ when regeneration stops.",
    "stop50_50_50-minus-stop50_50": "Same first two depths; incremental q2 including its retrieval budget.",
    "iter3_50_50_50_50-minus-stop50_50_50": "Same first three depths; incremental q3 including its retrieval budget.",
    "repeat3_50_50_50_50-minus-stop50_50_50": "Same first three depths; repeats last executed query without new candidates.",
    "iter3_50_50_50_50-minus-repeat3_50_50_50_50": "New versus repeated q3; realized budget may differ when regeneration stops.",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def normalized(text):
    return " ".join(text.casefold().split())


def inspect_inputs(study_root, model):
    expected = json.loads((HERE / "original_data_manifest.json").read_text())
    public_path = study_root / "data/musique/public.json"
    labels_path = study_root / "data/musique/labels.json"
    corpus_path = study_root / "data/musique/corpus.jsonl"
    for path, key in [(public_path, "public_sha256"), (labels_path, "labels_sha256"), (corpus_path, "corpus_sha256")]:
        if sha(path) != expected[key]:
            raise ValueError(f"Original input mismatch: {path}")
    public = json.loads(public_path.read_text())
    ids = json.loads((HERE / "question_ids.json").read_text())
    if [q["id"] for q in public] != ids or len(ids) != 2417 or len(set(ids)) != 2417:
        raise ValueError("The fixed original question IDs/order must not change")
    if any(set(q) != {"id", "question", "split"} for q in public):
        raise ValueError("Retrieval inputs must be public-only records")
    labels = {q["id"]: q for q in json.loads(labels_path.read_text())}
    if set(labels) != set(ids):
        raise ValueError("Label/public IDs differ")
    index = study_root / "indexes/musique"
    im = json.loads((index / "index_manifest.json").read_text())
    if im.get("status") != "complete" or im["corpus_sha256"] != expected["corpus_sha256"]:
        raise ValueError("Original index manifest is incomplete or mismatched")
    if sha(index / "embeddings.npy") != im["embeddings_sha256"]:
        raise ValueError("Original corpus embeddings changed")
    for relative, record in im["model_files"].items():
        if sha(model / relative) != record["sha256"]:
            raise ValueError(f"Indexed model mismatch: {relative}")
    sources = json.loads((HERE / "frozen_sources.json").read_text())
    for name, record in sources.items():
        if sha(HERE / "frozen" / name) != record["sha256"]:
            raise ValueError(f"Frozen source changed: {name}")
    return public, labels, corpus_path, index


def score(ids, label, lookup):
    gold = set(label["gold_doc_ids"])
    gold_titles = set(label["gold_titles"])
    positions = {doc_id: rank for rank, doc_id in enumerate(ids, 1)}
    metrics = {}
    for k in K_VALUES:
        selected = set(ids[:k])
        titles = {lookup[doc_id]["title"] for doc_id in selected}
        metrics[str(k)] = {
            "FP": int(gold <= selected),
            "paragraph_recall": len(gold & selected) / len(gold),
            "FT": int(gold_titles <= titles),
        }
    return {"at_k": metrics, "gold_ranks": {d: positions.get(d) for d in sorted(gold)}}


def retrieve(backend, expander, question, arm, measured_call, fused_rank):
    config = ARMS[arm]
    calls, rounds = [], []
    used = {normalized(question["question"])}
    all_targets = set()
    stop_reason = "scheduled_depths_completed"
    for round_no, depth in enumerate(config["depths"]):
        proposal = None
        if round_no == 0:
            query = question["question"]
            mode = "original"
        else:
            mode = config["modes"][round_no - 1]
            if mode == "repeat":
                query = rounds[-1]["query"]
                proposal = rounds[-1]["proposal"]
            else:
                proposal = expander.bridge_query(question["question"], fused_rank(calls, .5), backend.doc_lookup)
                query = proposal.get("query")
                # q1 exactly follows the archived single-expansion trigger.
                if not query or (round_no == 1 and query.strip() == question["question"].strip()):
                    stop_reason = f"q{round_no}_empty_or_unchanged"
                    break
                if round_no > 1 and normalized(query) in used:
                    stop_reason = f"q{round_no}_previously_executed"
                    break
        entities = [] if proposal is None else proposal.get("provenance", {}).get("entities", [])
        targets = [item["mentioned_title"] for item in entities]
        new_targets = sorted(set(targets) - all_targets)
        for route in ("bm25", "dense"):
            call = measured_call(backend, query, route, depth, f"q{round_no}", f"{question['id']}:{arm}:q{round_no}:{route}")
            call["round"] = round_no
            if call["cache_hit"]:
                raise AssertionError("Query caching must be disabled")
            calls.append(call)
        used.add(normalized(query))
        all_targets.update(targets)
        rounds.append({"round": round_no, "mode": mode, "depth_per_route": depth, "query": query,
                       "selected_titles": targets, "new_titles": new_targets, "proposal": proposal,
                       "ranking_after_round": fused_rank(calls, .5)})
    requested = sum(call["depth"] for call in calls)
    returned = sum(len(call["returned"]) for call in calls)
    if requested > 400 or returned > requested:
        raise AssertionError("D <= 400 violated")
    return {"ranking": fused_rank(calls, .5), "calls": calls, "rounds": rounds,
            "physical_route_calls": len(calls), "executed_query_instances": len(rounds),
            "supplementary_queries": len(rounds) - 1, "requested_records": requested,
            "returned_records": returned, "stop_reason": stop_reason}


def summarize(rows, ids):
    by = {(row["id"], row["arm"]): row for row in rows}
    if len(by) != len(ids) * len(ARMS):
        raise ValueError("Incomplete or duplicate query-arm rows")
    prefix_pairs = [("iter2_100_50_50", "stop100_50"), ("repeat2_100_50_50", "stop100_50"),
                    ("stop50_50_50", "stop50_50"), ("iter3_50_50_50_50", "stop50_50_50"),
                    ("repeat3_50_50_50_50", "stop50_50_50")]
    for longer, shorter in prefix_pairs:
        for qid in ids:
            prefix = by[qid, shorter]["trajectory_digests"]
            if by[qid, longer]["trajectory_digests"][:len(prefix)] != prefix:
                raise ValueError(f"Prefix mismatch for {qid}: {longer} vs {shorter}")
    arms = {}
    for arm in ARMS:
        selected = [by[qid, arm] for qid in ids]
        arms[arm] = {"n": len(selected),
            "supplementary_distribution": dict(collections.Counter(r["supplementary_queries"] for r in selected)),
            "mean_physical_route_calls": sum(r["physical_route_calls"] for r in selected) / len(selected),
            "mean_returned_records": sum(r["returned_records"] for r in selected) / len(selected),
            "at_k": {str(k): {metric: sum(r["at_k"][str(k)][metric] for r in selected) / len(selected)
                              for metric in ("FP", "FT", "paragraph_recall")} for k in K_VALUES}}
    comparisons = {}
    for name, interpretation in COMPARISONS.items():
        a, b = name.split("-minus-")
        comparisons[name] = {"interpretation": interpretation,
            "same_realized_D_n": sum(by[i, a]["returned_records"] == by[i, b]["returned_records"] for i in ids),
            "at_k": {}}
        for k in K_VALUES:
            differences = [by[i, a]["at_k"][str(k)]["FP"] - by[i, b]["at_k"][str(k)]["FP"] for i in ids]
            comparisons[name]["at_k"][str(k)] = {"wins": sum(d > 0 for d in differences),
                "losses": sum(d < 0 for d in differences), "delta_pp": 100 * sum(differences) / len(ids)}
    # Main-paper anchors are validation gates, not substituted experimental results.
    anchor_counts = {a: sum(by[i, a]["at_k"]["50"]["FP"] for i in ids) for a in ("RRF200", "single100")}
    if anchor_counts != {"RRF200": 1130, "single100": 1358}:
        raise ValueError(f"Single-round anchors do not reproduce the archived experiment: {anchor_counts}")
    return {"status": "COMPLETED", "arms": arms, "paired": comparisons, "anchor_counts": anchor_counts,
            "inference": "Descriptive fixed public-development comparison; no new confirmatory significance claims."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute:
        parser.error("Choose exactly one of --preflight-only or --execute")
    if platform.system() != "Linux":
        raise RuntimeError("This experiment must run on the authorized AutoDL Linux host, not the local Mac")
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1"})
    public, labels, corpus, index = inspect_inputs(args.study_root, args.model)
    if args.preflight_only:
        print(json.dumps({"status": "INPUTS_VERIFIED_NO_RETRIEVAL", "questions": len(public)}))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {"status": "RUNNING", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question_ids": [q["id"] for q in public], "question_ids_sha256": sha(HERE / "question_ids.json"),
        "arms": ARMS, "comparisons": COMPARISONS, "K": K_VALUES, "D_cap": 400, "alpha": .5,
        "threads": 8, "device": "cpu", "query_cache": False, "model_path": str(args.model),
        "corpus_sha256": sha(corpus), "index_manifest_sha256": sha(index / "index_manifest.json"),
        "script_sha256": sha(__file__), "original_manifest": json.loads((HERE / "original_data_manifest.json").read_text()),
        "scope": "Quality experiment only. Per-call stopwatch fields are diagnostics, not a matched latency study."}
    atomic(args.out / "protocol.json", protocol)
    sys.path.insert(0, str(HERE / "frozen"))
    from retrieval_backends import RetrievalBackend
    from expansion_variants import VariantExpander
    from run_retrieval import measured_call, fused_rank
    try:
        backend = RetrievalBackend(corpus, index, args.model, threads=8, device="cpu", cache_queries=False)
        expander = VariantExpander(backend.docs, policy="no_second_branch")
        backend.warmup()
        rows = []
        with gzip.open(args.out / "raw_rankings_and_calls.jsonl.gz", "wt", encoding="utf-8") as raw, \
             gzip.open(args.out / "quality_rows.jsonl.gz", "wt", encoding="utf-8") as quality:
            for qi, question in enumerate(public):
                order = list(ARMS)
                random.Random(2026110907 + qi).shuffle(order)
                for arm in order:
                    result = retrieve(backend, expander, question, arm, measured_call, fused_rank)
                    raw.write(json.dumps({"id": question["id"], "arm": arm, **result}, ensure_ascii=False) + "\n")
                    row = {"id": question["id"], "arm": arm, "hops": labels[question["id"]]["hops"],
                        **{key: result[key] for key in ("physical_route_calls", "executed_query_instances", "supplementary_queries",
                                                      "requested_records", "returned_records", "stop_reason")},
                        "round_titles": [r["selected_titles"] for r in result["rounds"]],
                        "new_title_counts": [len(r["new_titles"]) for r in result["rounds"]],
                        "trajectory_digests": [hashlib.sha256(json.dumps({"query": r["query"],
                            "ranking": r["ranking_after_round"]}, sort_keys=True).encode()).hexdigest()
                            for r in result["rounds"]],
                        "metrics_after_round": [score(r["ranking_after_round"], labels[question["id"]], backend.doc_lookup)
                                                for r in result["rounds"]],
                        **score(result["ranking"], labels[question["id"]], backend.doc_lookup)}
                    quality.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                if (qi + 1) % 25 == 0:
                    raw.flush()
                    quality.flush()
                    atomic(args.out / "progress.json", {"completed_questions": qi + 1, "total_questions": len(public), "updated_unix": time.time()})
        summary = summarize(rows, [q["id"] for q in public])
        atomic(args.out / "summary.json", summary)
        atomic(args.out / "complete.json", {"status": "COMPLETED", "questions": len(public), "arms": len(ARMS),
            "rows": len(rows), "protocol_sha256": sha(args.out / "protocol.json"),
            "raw_sha256": sha(args.out / "raw_rankings_and_calls.jsonl.gz"),
            "quality_sha256": sha(args.out / "quality_rows.jsonl.gz"), "summary_sha256": sha(args.out / "summary.json")})
    except BaseException:
        atomic(args.out / "failed.json", {"status": "FAILED_NOT_COMPLETE", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
