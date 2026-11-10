"""CPU-only, label-blind repair-study retrieval runner; scoring is separate.

Each committed group is one atomic gzip JSONL shard. A complete or failed pass
also publishes results.jsonl.gz containing one record per planned group. Failed
attempts remain under errors/ and can be retried on resume without discarding
their evidence. An identity mismatch always requires a new output directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import sys
import time
import traceback

from expansion_variants import POLICIES, QueryExpander, VariantExpander, variant_parameters

PROTOCOL_VERSION = "hybridrag_external_plaintext_fresh_questions_cpu_20261108_v1"
ALPHAS = {"2wiki": 0.3, "hotpotqa": 0.7, "strategyqa": 0.5}
HISTORICAL_ALTERNATES = {"2wiki": 0.7, "hotpotqa": 0.5}
TIMING_POLICIES = ("hybrid_rrf", "score2_u24", "family_u24")
PUBLIC_FIELDS = frozenset(("id", "question", "split"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _reject_label_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if ("gold" in normalized or "label" in normalized or
                    normalized in {"answer", "answers", "supporting_facts", "evidences", "support_titles"}):
                raise ValueError(f"labels are forbidden in public input: {key}")
            _reject_label_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_label_keys(child)


def load_public_questions(path, default_split="test"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    _reject_label_keys(data)
    rows = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("public JSON must contain a nonempty queries list")
    ids, questions, output = set(), set(), []
    for row in rows:
        if (not isinstance(row, dict) or not {"id", "question"} <= set(row)
                or not set(row) <= PUBLIC_FIELDS):
            raise ValueError("public query must contain id/question and optional split only")
        row = {"split": default_split, **row}
        if any(not isinstance(row[key], str) or not row[key].strip() for key in PUBLIC_FIELDS):
            raise ValueError("id, question and split must be nonempty strings")
        normalized = " ".join(row["question"].lower().split())
        if row["id"] in ids or normalized in questions:
            raise ValueError("duplicate public ID or normalized question")
        ids.add(row["id"])
        questions.add(normalized)
        output.append({key: row[key] for key in ("id", "question", "split")})
    return output


def plan_conditions(mode, dataset, question_number=0, seed=20261105):
    alpha = ALPHAS[dataset]
    if mode in {"confirmation", "holdout"}:
        return [{"policy": p, "alpha": alpha, "repeat": 0, "source_top_n": 5}
                for p in POLICIES]
    if mode == "historical":
        return [{"policy": p, "alpha": alpha, "repeat": 0, "source_top_n": 5}
                for p in POLICIES]
    if mode == "sensitivity":
        return [{"policy": "bridge_expand", "alpha": alpha, "repeat": 0,
                 "source_top_n": n} for n in (3, 5, 10)]
    if mode == "timing":
        orders = list(itertools.permutations(TIMING_POLICIES))
        random.Random(f"{seed}:{dataset}:balanced_timing_orders").shuffle(orders)
        return [{"policy": p, "alpha": alpha, "repeat": repeat + 1, "source_top_n": 5}
                for repeat in range(5)
                for p in orders[(question_number * 5 + repeat) % len(orders)]]
    raise ValueError("unknown mode")


def group_identity(dataset, question, mode, condition):
    return {"dataset": dataset, "id": question["id"], "split": question["split"],
            "mode": mode, "d_budget": 400, **condition}


def tree_fingerprints(directory):
    directory = Path(directory)
    files = {str(path.relative_to(directory)): {"sha256": sha256_file(path),
             "bytes": path.stat().st_size} for path in sorted(directory.rglob("*"))
             if path.is_file() and not path.name.startswith(".")}
    if not files:
        raise ValueError(f"empty artifact directory: {directory}")
    return {"tree_sha256": digest_json(files), "files": files}


def machine_info():
    def read_optional(path):
        try:
            return Path(path).read_text().strip()
        except (OSError, UnicodeDecodeError):
            return None
    cpuinfo = read_optional("/proc/cpuinfo")
    cpu_model = next((line.split(":", 1)[1].strip() for line in (cpuinfo or "").splitlines()
                      if line.startswith("model name")), platform.processor())
    packages = {}
    for name in ("numpy", "scipy", "torch", "transformers", "sentence-transformers", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"platform": platform.platform(), "python": sys.version,
            "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "memory_info": read_optional("/proc/meminfo"), "versions": packages,
            "cgroup": {p: read_optional(p) for p in (
                "/proc/self/cgroup", "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max",
                "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us")},
            "gpu_disabled": True, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "shared_node": True, "shared_node_note": "co-tenant activity is not controlled"}


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def disk_guard(out, storage_root, min_free_bytes, max_output_bytes):
    """Stop before another group, preserving committed shards and resume state."""
    reserve = 64 * 1024 ** 2
    free = shutil.disk_usage(out).free
    used = sum(path.stat().st_size for path in storage_root.rglob("*") if path.is_file())
    if free < min_free_bytes + reserve or used >= max_output_bytes - reserve:
        raise RuntimeError(f"storage pre-stop: free={free}, output_bytes={used}; "
                           f"reserve={reserve}, min_free={min_free_bytes}, cap={max_output_bytes}")
    return {"free_bytes": free, "output_bytes": used}


def write_group(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"committed group already exists: {path}")
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(gzip.compress(payload, mtime=0))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_group(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one record per group shard: {path}")
    return rows[0]


class RecordingBackend:
    """Retain finished physical-call evidence even if a later call fails."""
    def __init__(self, backend):
        self.backend = backend
        self.finished_calls = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def search(self, query, route, depth):
        result = self.backend.search(query, route, depth)
        self.finished_calls.append(result)
        return result


def execute_group(backend, expander, question, identity, retrieve, quota):
    recorded = RecordingBackend(backend)
    header = {**identity, "question": question["question"], "public_question": question,
              "variant_params": expander.params, "group_key": digest_json(identity)}
    try:
        group = retrieve(recorded, expander, question, identity["policy"], 400, identity["alpha"])
        sizes = (50,)
        selection_started = time.perf_counter()
        selections = {str(k): (quota(group["calls"], group["ranking"], k)
                              if identity["policy"] == "hybrid_quota" else group["ranking"][:k])
                      for k in sizes}
        selection_seconds = time.perf_counter() - selection_started
        # Smaller K caps reuse the same calls/ranking outside the K50 timer.
        selections.update({str(k): group["ranking"][:k] for k in (10, 20, 30)})
        from reviewer_extensions import group_diagnostics
        diagnostics = group_diagnostics(backend, expander, question, group, identity["mode"])
        row = {**header, **group, "status": "complete", "selections": selections,
               "q1_provenance": group["expansion_provenance"],
               "reviewer_diagnostics": diagnostics,
               "initial_selections": diagnostics["initial_selections"],
               "selection_seconds": selection_seconds,
               "pipeline_plus_selection_seconds": group["total_seconds"] + selection_seconds,
               "j_count": len(group["calls"]),
               "q_count": len({call["query_id"] for call in group["calls"]}),
               "distinct_query_text_count": len({call["query"] for call in group["calls"]}),
               "d_count": sum(len(call["returned"]) for call in group["calls"]),
               "warmup_excluded": True, "gpu_disabled": True, "shared_node": True,
               "cache_queries": False}
        if row["j_count"] > 4 or row["d_count"] > 400:
            raise ValueError("executed group exceeded frozen call/return caps")
        for k, ids in selections.items():
            if len(ids) > int(k) or len(ids) != len(set(ids)):
                raise ValueError("selection exceeded cap or contained duplicate IDs")
        if any(call["cache_hit"] or call["query_cache_enabled"] for call in group["calls"]):
            raise ValueError("query-vector caching violates this protocol")
        return row
    except Exception as exc:
        return {**header, "status": "error", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "partial_backend_calls": recorded.finished_calls,
                "calls": [], "ranking": [], "initial": [], "selections": {},
                "expanded_query": None, "expansion_provenance": None,
                "q1_provenance": None, "reason": "execution_error"}


def publish_aggregate(out, planned, signature):
    destination = out / "results.jsonl.gz"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            for key in planned:
                success = out / "groups" / f"{key}.jsonl.gz"
                errors = sorted((out / "errors" / key).glob("*.jsonl.gz"))
                if not success.exists() and not errors:
                    continue
                row = read_group(success if success.exists() else errors[-1])
                if row["signature"] != signature:
                    raise ValueError("aggregate encountered a foreign group")
                compressed.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "corpus", "index", "model", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(ALPHAS))
    parser.add_argument("--mode", required=True,
                        choices=("confirmation", "historical", "holdout", "timing"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--expected-questions", type=int,
                        help="confirmation N fixed in the frozen protocol; default 300")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--technical-preflight", action="store_true",
                        help="technical run of 5-10 public historical questions; never new effectiveness evidence")
    parser.add_argument("--storage-root", type=Path,
                        help="shared study-output root only, excluding corpora/models/indexes")
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--max-output-gib", type=float, default=2.0)
    args = parser.parse_args()
    if args.threads != 8:
        parser.error("the reviewer protocol fixes --threads 8")
    from reviewer_extensions import verify_protocol
    frozen_protocol_sha256 = verify_protocol(args.protocol, args.protocol_sha256)
    # Bind the selected vocabulary to both the dataset and frozen protocol.
    frozen = json.loads(args.protocol.read_text())
    cue_path = Path(os.environ.get("HYBRIDRAG_CUE_FILE", ""))
    cue_digest = os.environ.get("HYBRIDRAG_CUE_SHA256", "")
    if not cue_path.is_file() or sha256_file(cue_path) != cue_digest:
        raise ValueError("Missing or modified cue file")
    cue_payload = json.loads(cue_path.read_text())
    if cue_payload.get("status") != "mined_without_task_labels" or frozen["cue_config_sha256"] != cue_digest:
        raise ValueError("Shared cue configuration/protocol mismatch")

    if args.expected_questions is not None and args.expected_questions <= 0:
        parser.error("--expected-questions must be positive")
    default_split = {"confirmation": "confirmation", "timing": "confirmation",
                     "historical": "historical", "holdout": "project_holdout", "sensitivity": "dev"}[args.mode]
    questions = load_public_questions(args.questions, default_split)
    required = {"confirmation": args.expected_questions or 300,
                "historical": args.expected_questions or 300,
                "holdout": args.expected_questions or 300,
                "sensitivity": 100, "timing": 100}.get(args.mode)
    if args.technical_preflight:
        if args.mode != "confirmation" or "preflight" not in args.out.parts:
            raise ValueError("technical preflight requires mode confirmation and a preflight output path")
        if not 5 <= len(questions) <= 10:
            raise ValueError("technical preflight requires 5-10 historical questions")
        required = len(questions)
        for question in questions:
            question["split"] = "technical_preflight_historical"
    if required is not None and len(questions) != required:
        raise ValueError(f"{args.mode} requires exactly {required} public questions")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    from run_retrieval import quota_selection
    from reviewer_extensions import retrieve_group
    from retrieval_backends import RetrievalBackend

    args.out.mkdir(parents=True, exist_ok=True)
    storage_root = args.storage_root or args.out
    if not storage_root.is_dir():
        raise ValueError("--storage-root must be an existing output directory")
    lock = (args.out / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    code_dir = Path(__file__).parent
    code_names = tuple(path.name for path in sorted(code_dir.glob("*.py")))
    environment = machine_info()
    identity = {"protocol_version": PROTOCOL_VERSION, "dataset": args.dataset,
        "mode": args.mode, "threads": args.threads, "seed": args.seed,
        "frozen_protocol_sha256": frozen_protocol_sha256,
        "K_values_from_same_calls": [10, 20, 30, 50],
        "timing_scope": "post-warmup encoding, retrieval, expansion, fusion; K50 selection separately recorded; excludes auxiliary candidate audit, loading, indexing, logging",
        "technical_preflight": args.technical_preflight,
        "runtime_versions": environment["versions"], "python": platform.python_version(),
        "cpu_model": environment["cpu_model"], "affinity": environment["affinity"],
        "questions_sha256": sha256_file(args.questions), "corpus_sha256": sha256_file(args.corpus),
        "model": tree_fingerprints(args.model), "index": tree_fingerprints(args.index),
        "code_sha256": {name: sha256_file(code_dir / name) for name in code_names},
        "base_alpha": ALPHAS[args.dataset], "d_budget": 400,
        "query_cache_enabled": False, "device": "cpu", "labels_accepted": False,
        "cue_config_sha256": cue_digest,
        "conditions": plan_conditions(args.mode, args.dataset, seed=args.seed),
        "timing_order": "seeded_permutation_cycle_of_all_six_method_orders",
        "disk_guard": {"storage_root": str(storage_root.resolve()),
                       "min_free_gib": args.min_free_gib, "max_output_gib": args.max_output_gib,
                       "prestop_reserve_mib": 64}}
    signature = digest_json(identity)
    manifest_path = args.out / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old is not None and old.get("signature") != signature:
        raise ValueError("run identity changed; use a fresh output directory")
    if old is None and any(path.name != ".run.lock" for path in args.out.iterdir()):
        raise ValueError("unidentified nonempty output directory; refusing to mix outputs")
    planned, question_plans = {}, []
    for number, question in enumerate(questions):
        conditions = plan_conditions(args.mode, args.dataset, number, args.seed)
        if args.mode != "timing":
            random.Random(f"{args.seed}:{args.dataset}:{args.mode}:{question['id']}").shuffle(conditions)
        local = []
        for condition in conditions:
            item = group_identity(args.dataset, question, args.mode, condition)
            key = digest_json(item)
            if key in planned:
                raise ValueError("duplicate planned group")
            planned[key] = item
            local.append((key, item))
        question_plans.append(local)
    completed, shard_hashes = set(), {}
    for path in sorted((args.out / "groups").glob("*.jsonl.gz")):
        row = read_group(path)
        key = row.get("group_key")
        if (key not in planned or path.name != f"{key}.jsonl.gz" or
                row.get("signature") != signature or row.get("status") != "complete"):
            raise ValueError(f"invalid or foreign committed group: {path}")
        digest = sha256_file(path)
        if old and key in old.get("group_file_sha256", {}) and old["group_file_sha256"][key] != digest:
            raise ValueError("committed group bytes changed after a previous invocation")
        completed.add(key)
        shard_hashes[key] = digest
    manifest = {"signature": signature, "identity": identity, "status": "running",
        "created_at_utc": old.get("created_at_utc") if old else utc_now(),
        "invocations": old.get("invocations", []) if old else [],
        "expected_groups": len(planned), "completed_groups": len(completed),
        "group_file_sha256": shard_hashes, "planned_group_keys": list(planned),
        "machine": environment, "resume_compatible": True,
        "technical_preflight": args.technical_preflight}
    invocation = {"started_at_utc": utc_now(), "starting_completed_groups": len(completed),
                  "machine": manifest["machine"]}
    manifest["invocations"].append(invocation)
    write_json(manifest_path, manifest)
    backend = RetrievalBackend(args.corpus, args.index, args.model,
                               threads=args.threads, device="cpu", cache_queries=False)
    warmup_seconds = backend.warmup()
    actual_device = str(backend._model.device)
    if actual_device != "cpu":
        raise ValueError(f"model is not on CPU: {actual_device}")
    base = QueryExpander(backend.docs)
    expanders = {}
    for item in planned.values():
        condition = (item["policy"], item["source_top_n"])
        if condition not in expanders:
            expanders[condition] = VariantExpander(policy=condition[0], source_top_n=condition[1],
                                                 title_lookup=base.title_lookup)
    invocation.update({"warmup_seconds": warmup_seconds, "warmup_excluded": True,
                       "actual_model_device": actual_device})
    write_json(manifest_path, manifest)
    started, errors_this_pass = time.perf_counter(), []
    for number, (question, local) in enumerate(zip(questions, question_plans), 1):
        disk_guard(args.out, storage_root, int(args.min_free_gib * 1024 ** 3),
                   int(args.max_output_gib * 1024 ** 3))
        for key, item in local:
            if key in completed:
                continue
            expander = expanders[(item["policy"], item["source_top_n"])]
            row = execute_group(backend, expander, question, item, retrieve_group, quota_selection)
            row.update({"signature": signature, "recorded_at_utc": utc_now()})
            if row["status"] == "complete":
                path = args.out / "groups" / f"{key}.jsonl.gz"
                write_group(path, row)
                completed.add(key)
                shard_hashes[key] = sha256_file(path)
            else:
                path = args.out / "errors" / key / f"{time.time_ns()}.jsonl.gz"
                write_group(path, row)
                errors_this_pass.append(key)
        progress = {"status": "running", "dataset": args.dataset, "mode": args.mode,
            "visited_questions": number, "expected_questions": len(questions),
            "completed_questions": sum(all(k in completed for k, _ in p) for p in question_plans),
            "expected_groups": len(planned), "completed_groups": len(completed),
            "errors_this_pass": len(errors_this_pass), "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes()}
        write_json(args.out / "progress.json", progress)
        manifest.update({"completed_groups": len(completed), "group_file_sha256": shard_hashes,
                         "peak_rss_bytes": peak_rss_bytes()})
        write_json(manifest_path, manifest)
        print(json.dumps(progress), flush=True)
    publish_aggregate(args.out, planned, signature)
    status = "complete" if len(completed) == len(planned) else "failed"
    invocation.update({"finished_at_utc": utc_now(), "status": status,
        "elapsed_seconds": time.perf_counter() - started, "errors_this_pass": errors_this_pass,
        "peak_rss_bytes": peak_rss_bytes()})
    manifest.update({"status": status, "completed_groups": len(completed),
        "failed_group_keys": [key for key in planned if key not in completed],
        "results_sha256": sha256_file(args.out / "results.jsonl.gz"),
        "completed_at_utc": utc_now(), "peak_rss_bytes": peak_rss_bytes()})
    write_json(manifest_path, manifest)
    write_json(args.out / "progress.json", {"status": status, "expected_groups": len(planned),
        "completed_groups": len(completed), "expected_questions": len(questions),
        "results_sha256": manifest["results_sha256"], "peak_rss_bytes": peak_rss_bytes()})
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
