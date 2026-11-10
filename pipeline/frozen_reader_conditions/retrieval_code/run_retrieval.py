"""Execute frozen depth and query-expansion retrieval pilots without a reader."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from query_expansion import QueryExpander
from retrieval_backends import RetrievalBackend, file_sha256, write_json


POLICIES = ("bm25", "dense", "hybrid_rrf", "hybrid_quota",
            "query_expand", "bridge_expand")
K_VALUES = (20, 50, 100, 200)
D_VALUES = (200, 400, 800)
ALPHA_VALUES = (0.3, 0.5, 0.7, 0.9)


def public_question(record):
    return {key: record[key] for key in ("id", "split", "question")}


def fused_rank(calls, alpha):
    """Normalize route votes within each query and query votes across queries."""
    query_ids = {call["query_id"] for call in calls}
    scores = collections.defaultdict(float)
    for call in calls:
        weight = alpha if call["route"] == "dense" else 1 - alpha
        weight /= max(len(query_ids), 1)
        seen = set()
        for record in call["returned"]:
            doc_id = record["doc_id"]
            if doc_id not in seen:
                scores[doc_id] += weight / (60 + record["rank"])
                seen.add(doc_id)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def quota_selection(calls, ranked, k):
    """Reserve one quarter of the requested slots for each retrieval route."""
    chosen = []
    for route in ("bm25", "dense"):
        route_ids = [row["doc_id"] for call in calls if call["route"] == route
                     for row in call["returned"]]
        remaining_quota = k // 4
        for doc_id in route_ids:
            if doc_id not in chosen:
                chosen.append(doc_id)
                remaining_quota -= 1
            if remaining_quota <= 0:
                break
    chosen.extend(doc_id for doc_id in ranked if doc_id not in chosen)
    return chosen[:k]


def measured_call(backend, query, route, depth, query_id, call_id):
    result = backend.search(query, route, int(depth))
    timing = result["timing"]
    return {
        "call_id": call_id, "query_id": query_id, "query": query,
        "round": 0 if query_id == "q0" else 1,
        "route": route, "depth": int(depth), "returned": result["results"],
        "query_encode_seconds": timing["query_encoding_seconds"],
        "search_seconds": timing["search_seconds"],
        "total_seconds": timing["total_seconds"],
        "cache_hit": timing["query_cache_hit"],
        "query_cache_enabled": timing.get("query_cache_enabled", False),
    }


def retrieve_group(backend, expander, question, policy, d_budget, alpha):
    started = time.perf_counter()
    prefix = hashlib.sha256(f"{question['id']}:{policy}:{d_budget}:{time.time_ns()}".encode()).hexdigest()[:24]
    calls = []

    def run(query, route, depth, query_id):
        call = measured_call(backend, query, route, depth, query_id,
                             f"{prefix}_{len(calls)}")
        calls.append(call)

    original = str(question["question"])
    expansion = {"query": None, "provenance": [], "reason": "single_query"}
    query_generation_seconds = 0.0
    fusion_seconds = 0.0
    if policy in {"bm25", "dense"}:
        run(original, policy, d_budget, "q0")
        ranking = [record["doc_id"] for record in calls[0]["returned"]]
        initial = ranking[:len(ranking) // 2]
    elif policy in {"hybrid_rrf", "hybrid_quota"}:
        for route in ("bm25", "dense"):
            run(original, route, d_budget // 2, "q0")
        fusion_started = time.perf_counter()
        ranking = fused_rank(calls, alpha)
        prefix_calls = [{**call, "returned": call["returned"][:d_budget // 4]}
                        for call in calls]
        initial = fused_rank(prefix_calls, alpha)
        fusion_seconds += time.perf_counter() - fusion_started
    else:
        for route in ("bm25", "dense"):
            run(original, route, d_budget // 4, "q0")
        fusion_started = time.perf_counter()
        initial = fused_rank(calls, alpha)
        fusion_seconds += time.perf_counter() - fusion_started
        expansion_started = time.perf_counter()
        if policy == "query_expand":
            expansion = expander.question_query(original)
        else:
            expansion = expander.bridge_query(original, initial, backend.doc_lookup)
        query_generation_seconds = time.perf_counter() - expansion_started
        expanded = expansion.get("query")
        if expanded and expanded.strip() != original.strip():
            for route in ("bm25", "dense"):
                run(expanded, route, d_budget // 4, "q1")
        else:
            expansion = {**expansion, "query": None,
                         "reason": expansion.get("reason", "no_distinct_expansion")}
        fusion_started = time.perf_counter()
        ranking = fused_rank(calls, alpha)
        fusion_seconds += time.perf_counter() - fusion_started
    return {
        "calls": calls, "ranking": ranking, "initial": initial,
        "expanded_query": expansion.get("query"),
        "expansion_provenance": expansion.get("provenance", []),
        "reason": expansion.get("reason", "completed"),
        "query_generation_seconds": query_generation_seconds,
        "fusion_seconds": fusion_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def title_scores(ids, gold_titles, lookup):
    selected = {lookup[doc_id]["title"] for doc_id in ids}
    gold = set(gold_titles)
    return len(selected & gold) / len(gold), float(gold <= selected)


def tune_alpha(backend, dev_questions, out, signature):
    """Only this offline calibration function reads training labels."""
    records = []
    for question in dev_questions:
        calls = [measured_call(backend, question["question"], route, 200, "q0",
                               f"tuning_{question['id']}_{route}")
                 for route in ("bm25", "dense")]
        for alpha in ALPHA_VALUES:
            ranked = fused_rank(calls, alpha)[:50]
            recall, complete = title_scores(ranked, question["gold_titles"],
                                             backend.doc_lookup)
            records.append({"id": question["id"], "alpha": alpha,
                            "recall": recall, "complete_support": complete})
    scores = []
    for alpha in ALPHA_VALUES:
        subset = [record for record in records if record["alpha"] == alpha]
        scores.append({"alpha": alpha,
                       "recall": sum(row["recall"] for row in subset) / len(subset),
                       "complete_support": sum(row["complete_support"] for row in subset) / len(subset)})
    best = max(scores, key=lambda row: (row["recall"], row["complete_support"],
                                        -abs(row["alpha"] - 0.5), -row["alpha"]))
    result = {"run_signature": signature,
              "dev_ids": [question["id"] for question in dev_questions],
              "selected_alpha": best["alpha"], "split": "dev", "k": 50,
              "d_budget": 400, "trials": scores, "rows": records,
              "test_labels_used": False}
    write_json(out, result)
    return best["alpha"]


def read_recoverable_jsonl(path):
    """Keep complete committed rows and remove a torn final append on retry."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end != len(data):
        path.write_bytes(data[:end])
    return [json.loads(line) for line in data[:end].splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hotpotqa", "2wiki"), required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_lock = (args.out / ".run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another process owns this run directory") from exc
    corpus_path = args.data_root / args.dataset / "corpus.jsonl"
    questions_path = args.data_root / args.dataset / "queries.json"
    questions = json.loads(questions_path.read_text())["queries"]
    configurations = [
        {"policy": policy, "k": k, "d_budget": d_budget,
         "j_budget": 4, "datasets": [args.dataset], "splits": ["dev", "test"]}
        for policy in POLICIES for d_budget in D_VALUES for k in K_VALUES]
    code_paths = [Path(__file__), Path(__file__).with_name("query_expansion.py"),
                  Path(__file__).with_name("retrieval_backends.py")]
    identity = {
        "dataset": args.dataset, "corpus_sha256": file_sha256(corpus_path),
        "questions_sha256": file_sha256(questions_path),
        "index_manifest_sha256": file_sha256(args.index / "index_manifest.json"),
        "code_sha256": {path.name: file_sha256(path) for path in code_paths},
        "configurations": configurations, "seed": args.seed,
        "alpha_grid": list(ALPHA_VALUES), "threads": args.threads,
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = args.out / "manifest.json"
    old = None
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["signature"] != signature:
            raise ValueError("run identity changed; choose a fresh run directory")
    manifest = {
        "status": "running", "signature": signature, "identity": identity,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "status": "historical_sample_matched_100_200_retrieval",
            "datasets": [args.dataset], "splits": ["dev", "test"],
            "configurations": configurations, "latency_samples_shared_across_K": True,
            "latency_cache": "model warmed, query embedding cache disabled",
            "initial_reference": "first half of D; depth baselines use rank prefixes",
            "primary_pair": "bridge_expand minus hybrid_rrf at the same K and D cap",
            "resource_matching": "shared ceilings and final K; actual J/D/latency reported",
            "reader_calls": 0, "corpus": "fixed validation context union",
            "fusion": "RRF k=60, route alpha calibrated on dev, equal query weights",
        },
    }
    tuning_path = args.out / "tuning.json"
    if old is not None and old.get("tuning_sha256"):
        if not tuning_path.exists() or file_sha256(tuning_path) != old["tuning_sha256"]:
            raise ValueError("frozen dev calibration changed on resume")
    write_json(manifest_path, manifest)
    backend = RetrievalBackend(corpus_path, args.index, args.model,
                               threads=args.threads, device="cpu", cache_queries=False)
    warmup_seconds = backend.warmup()
    expander_started = time.perf_counter()
    expander = QueryExpander(backend.docs)
    expander_build_seconds = time.perf_counter() - expander_started
    dev_questions = [question for question in questions if question["split"] == "dev"]
    if tuning_path.exists():
        calibration = json.loads(tuning_path.read_text())
        if (calibration.get("run_signature") != signature
                or calibration.get("dev_ids") != [question["id"] for question in dev_questions]):
            raise ValueError("dev calibration belongs to a different run")
        alpha = calibration["selected_alpha"]
        if alpha not in ALPHA_VALUES:
            raise ValueError("calibrated alpha is outside the frozen grid")
    else:
        alpha = tune_alpha(backend, dev_questions, tuning_path, signature)
    manifest.update({"selected_alpha": alpha, "warmup_seconds": warmup_seconds,
                     "expander_build_seconds": expander_build_seconds,
                     "tuning_sha256": file_sha256(tuning_path)})
    write_json(manifest_path, manifest)
    results_path, calls_path = args.out / "results.jsonl", args.out / "calls.jsonl"
    previous_calls = read_recoverable_jsonl(calls_path)
    known_call_ids = {call["call_id"] for call in previous_calls}
    old_rows = read_recoverable_jsonl(results_path)
    done = set()
    for row in old_rows:
        if (row["signature"] != signature or row.get("alpha") != alpha
                or not set(row["calls"]) <= known_call_ids):
            raise ValueError("incompatible or uncommitted result row")
        key = (row["id"], row["policy"], row["d_budget"], row["k"])
        if key in done:
            raise ValueError("duplicate result row")
        done.add(key)
    start = time.perf_counter()
    expected = len(questions) * len(configurations)
    with calls_path.open("a", encoding="utf-8") as calls_output, results_path.open("a", encoding="utf-8") as results_output:
        for number, private_question in enumerate(questions):
            question = public_question(private_question)
            policies = list(POLICIES)
            random.Random(args.seed ^ int(hashlib.sha256(question["id"].encode()).hexdigest()[:8], 16)).shuffle(policies)
            for policy in policies:
                for d_budget in D_VALUES:
                    missing = [k for k in K_VALUES if (question["id"], policy, d_budget, k) not in done]
                    if not missing:
                        continue
                    group = retrieve_group(backend, expander, question, policy, d_budget, alpha)
                    calls = group["calls"]
                    for call in calls:
                        calls_output.write(json.dumps(call, ensure_ascii=False) + "\n")
                    calls_output.flush()
                    os.fsync(calls_output.fileno())
                    cumulative = list(dict.fromkeys(row["doc_id"] for call in calls for row in call["returned"]))
                    for k in missing:
                        fusion_started = time.perf_counter()
                        selected = (quota_selection(calls, group["ranking"], k)
                                    if policy == "hybrid_quota" else group["ranking"][:k])
                        selection_seconds = time.perf_counter() - fusion_started
                        row = {
                            "signature": signature, "id": question["id"],
                            "dataset": args.dataset, "split": question["split"],
                            "policy": policy, "k": k, "d_budget": d_budget, "j_budget": 4,
                            "alpha": alpha,
                            "selected_ids": selected, "cumulative_ids": cumulative,
                            "initial_ids": group["initial"][:k],
                            "calls": [call["call_id"] for call in calls],
                            "q_count": len({call["query_id"] for call in calls}),
                            "j_count": len(calls),
                            "d_count": sum(len(call["returned"]) for call in calls),
                            "total_seconds": group["total_seconds"] + selection_seconds,
                            "fusion_seconds": group["fusion_seconds"] + selection_seconds,
                            "query_generation_seconds": group["query_generation_seconds"],
                            "expanded_query": group["expanded_query"],
                            "expansion_provenance": group["expansion_provenance"],
                            "reason": group["reason"],
                        }
                        results_output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done.add((question["id"], policy, d_budget, k))
                    results_output.flush()
                    os.fsync(results_output.fileno())
            progress = {"status": "running", "dataset": args.dataset,
                        "completed_questions": number + 1, "questions": len(questions),
                        "rows": len(done), "expected_rows": expected,
                        "seconds": time.perf_counter() - start}
            write_json(args.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    manifest.update({"status": "complete", "rows": len(done), "expected_rows": expected,
                     "invocation_seconds": time.perf_counter() - start,
                     "results_sha256": file_sha256(results_path),
                     "calls_sha256": file_sha256(calls_path),
                     "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    if len(done) != expected:
        raise ValueError("incomplete run")
    write_json(manifest_path, manifest)
    write_json(args.out / "progress.json", {"status": "complete", "rows": len(done), "questions": len(questions)})


if __name__ == "__main__":
    main()
