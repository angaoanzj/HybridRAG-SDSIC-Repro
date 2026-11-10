"""Build resumable BGE vectors and a sparse BM25 index for a frozen corpus."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from retrieval_backends import (BM25_SETTINGS, DOCUMENT_RENDER, build_sparse_index,
                                file_sha256, read_corpus, render_document, write_json)


def model_fingerprints(model_path):
    files = {}
    for path in sorted(model_path.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin"}:
            files[str(path.relative_to(model_path))] = {
                "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if not files or not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("model snapshot has no local weight file")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.chunk_size) <= 0:
        raise ValueError("thread, batch and chunk sizes must be positive")
    if args.device not in {"cpu", "cuda"}:
        raise ValueError("index device must be cpu or cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.out / ".build.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another process is building this index") from error
    started = time.perf_counter()
    docs = read_corpus(args.corpus)
    chunks = args.out / "embedding_chunks"
    chunks.mkdir(exist_ok=True)
    code_paths = [Path(__file__), Path(__file__).with_name("retrieval_backends.py")]
    identity = {
        "corpus_sha256": file_sha256(args.corpus), "documents": len(docs),
        "model_files": model_fingerprints(args.model),
        "document_render": DOCUMENT_RENDER, "max_sequence_length": 512,
        "normalized_embeddings": True, "dtype": "float32", "device": args.device,
        "threads": args.threads, "batch_size": args.batch_size,
        "chunk_size": args.chunk_size, "bm25": BM25_SETTINGS,
        "code_sha256": {path.name: file_sha256(path) for path in code_paths},
        "library_versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "sentence-transformers", "numpy", "scipy")},
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")},
    }
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = args.out / "index_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("signature") != signature:
            raise ValueError("existing index has a different signature; choose a fresh directory")
        if old.get("status") == "complete":
            for relative, fingerprint in old["outputs"].items():
                if file_sha256(args.out / relative) != fingerprint["sha256"]:
                    raise ValueError(f"completed index output corrupted: {relative}")
            print(json.dumps({"status": "complete_reused", "signature": signature}), flush=True)
            return
    manifest = {**identity, "signature": signature, "status": "running",
                "corpus_path": str(args.corpus), "model_path": str(args.model),
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(manifest_path, manifest)
    write_json(args.out / "docids.json", [doc["doc_id"] for doc in docs])
    sparse_files = [args.out / name for name in
                    ("sparse_bm25.npz", "vocabulary.json", "sparse_metadata.json")]
    if all(path.exists() for path in sparse_files):
        sparse_metadata = json.loads(sparse_files[-1].read_text())
        if sparse_metadata.get("signature") != signature:
            raise ValueError("incomplete sparse index has an incompatible signature")
    else:
        sparse_metadata = build_sparse_index(docs, args.out)
        sparse_metadata["signature"] = signature
        write_json(args.out / "sparse_metadata.json", sparse_metadata)
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = SentenceTransformer(str(args.model), device=args.device, local_files_only=True)
    model.max_seq_length = 512
    model.float()
    model.eval()
    total_chunks = (len(docs) + args.chunk_size - 1) // args.chunk_size
    chunk_records = []
    for number, start in enumerate(range(0, len(docs), args.chunk_size)):
        end = min(start + args.chunk_size, len(docs))
        vector_path = chunks / f"{number:05d}.npy"
        record_path = chunks / f"{number:05d}.json"
        if vector_path.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("signature") != signature or record.get("start") != start:
                raise ValueError("embedding chunk signature mismatch")
            vectors = np.load(vector_path, mmap_mode="r")
            if vectors.shape != tuple(record["shape"]) or vectors.shape[0] != end - start:
                raise ValueError("embedding chunk shape mismatch")
            if file_sha256(vector_path) != record["sha256"]:
                raise ValueError("embedding chunk file corrupted")
        else:
            chunk_started = time.perf_counter()
            texts = [render_document(doc) for doc in docs[start:end]]
            vectors = model.encode(texts, batch_size=args.batch_size,
                                   normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True).astype(np.float32)
            if len(vectors) != end - start or not np.all(np.isfinite(vectors)):
                raise ValueError("invalid model embeddings")
            temporary = vector_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, vectors)
            temporary.replace(vector_path)
            record = {"signature": signature, "start": start, "end": end,
                      "shape": list(vectors.shape), "sha256": file_sha256(vector_path),
                      "seconds": time.perf_counter() - chunk_started}
            write_json(record_path, record)
        chunk_records.append(record)
        progress = {"status": "encoding", "chunk": number + 1,
                    "total_chunks": total_chunks, "completed_documents": end,
                    "total_documents": len(docs),
                    "compute_seconds": sum(row["seconds"] for row in chunk_records),
                    "invocation_seconds": time.perf_counter() - started}
        write_json(args.out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    dimension = chunk_records[0]["shape"][1]
    final_temporary = args.out / "embeddings.npy.tmp"
    combined = np.lib.format.open_memmap(final_temporary, mode="w+", dtype=np.float32,
                                        shape=(len(docs), dimension))
    for number, record in enumerate(chunk_records):
        combined[record["start"]:record["end"]] = np.load(chunks / f"{number:05d}.npy")
    combined.flush()
    del combined
    final_temporary.replace(args.out / "embeddings.npy")
    output_names = ["embeddings.npy", "docids.json", "sparse_bm25.npz",
                    "vocabulary.json", "sparse_metadata.json"]
    manifest.update({"status": "complete", "embedding_dimensions": dimension,
                     "truncated_document_count": None,
                     "truncation_audit": "max length is enforced; no extra full tokenization scan",
                     "embedding_compute_seconds": sum(row["seconds"] for row in chunk_records),
                     "sparse_build_seconds": sparse_metadata["seconds"],
                     "invocation_seconds": time.perf_counter() - started,
                     "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                     "outputs": {name: {"bytes": (args.out / name).stat().st_size,
                                         "sha256": file_sha256(args.out / name)}
                                 for name in output_names}})
    write_json(manifest_path, manifest)
    write_json(args.out / "progress.json", {"status": "complete", "documents": len(docs)})
    print(json.dumps({"status": "complete", "documents": len(docs),
                      "seconds": manifest["invocation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
