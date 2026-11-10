"""Frozen public-corpus BM25 and BGE retrieval backends.

The corpus contains document text only. Query labels are never accepted here.
BM25 uses nonnegative Robertson IDF and unique query terms. Dense retrieval is
an exact cosine search over normalized, float32 BGE document vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from scipy import sparse


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOCUMENT_RENDER = "title + '\\n' + text"
BM25_SETTINGS = {
    "k1": 1.5,
    "b": 0.75,
    "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
    "tokenization": "lowercase ASCII [a-z0-9]+; no stopwords",
    "query_term_frequency": "unique query terms",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_corpus(path):
    with Path(path).open(encoding="utf-8") as source:
        docs = [json.loads(line) for line in source if line.strip()]
    ids = [doc["doc_id"] for doc in docs]
    if not docs or len(ids) != len(set(ids)):
        raise ValueError("corpus must contain nonempty, unique document IDs")
    for doc in docs:
        if not isinstance(doc["title"], str) or not isinstance(doc["text"], str):
            raise ValueError("document title and text must be strings")
    return docs


def render_document(doc):
    return doc["title"] + "\n" + doc["text"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def build_sparse_index(docs, out):
    """Write document-term BM25 weights and vocabulary without dense expansion."""
    from sklearn.feature_extraction.text import CountVectorizer

    out = Path(out)
    started = time.perf_counter()
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None,
                                 lowercase=False, dtype=np.float32)
    matrix = vectorizer.fit_transform([render_document(doc) for doc in docs]).tocsr()
    lengths = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    average_length = float(lengths.mean())
    if average_length <= 0:
        raise ValueError("corpus has no indexable tokens")
    frequencies = np.diff(matrix.tocsc().indptr)
    idf = np.log1p((len(docs) - frequencies + 0.5) / (frequencies + 0.5))
    row_ids = np.repeat(np.arange(len(docs)), np.diff(matrix.indptr))
    k1, b = BM25_SETTINGS["k1"], BM25_SETTINGS["b"]
    length_norm = k1 * (1 - b + b * lengths / average_length)
    matrix.data *= ((k1 + 1) * idf[matrix.indices]
                    / (matrix.data + length_norm[row_ids])).astype(np.float32)
    temporary = out / "sparse_bm25.npz.tmp"
    with temporary.open("wb") as stream:
        sparse.save_npz(stream, matrix, compressed=True)
    temporary.replace(out / "sparse_bm25.npz")
    write_json(out / "vocabulary.json", vectorizer.vocabulary_)
    metadata = {**BM25_SETTINGS, "documents": len(docs),
                "vocabulary_size": matrix.shape[1], "nonzeros": matrix.nnz,
                "average_document_tokens": average_length,
                "seconds": time.perf_counter() - started}
    write_json(out / "sparse_metadata.json", metadata)
    return metadata


class RetrievalBackend:
    def __init__(self, corpus_path, index_dir, model_path, threads=8, device="cpu",
                 cache_queries=False):
        self.index_dir = Path(index_dir)
        self.docs = read_corpus(corpus_path)
        self.doc_lookup = {doc["doc_id"]: doc for doc in self.docs}
        self.doc_ids = [doc["doc_id"] for doc in self.docs]
        manifest = json.loads((self.index_dir / "index_manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError("index is incomplete")
        if manifest["corpus_sha256"] != file_sha256(corpus_path):
            raise ValueError("index corpus signature mismatch")
        recorded_ids = json.loads((self.index_dir / "docids.json").read_text())
        if recorded_ids != self.doc_ids:
            raise ValueError("index and corpus document order differ")
        self.embeddings = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        if self.embeddings.shape[0] != len(self.docs):
            raise ValueError("embedding and corpus sizes differ")
        self.bm25 = sparse.load_npz(self.index_dir / "sparse_bm25.npz").tocsc()
        self.vocabulary = json.loads((self.index_dir / "vocabulary.json").read_text())
        self.sparse_metadata = json.loads((self.index_dir / "sparse_metadata.json").read_text())
        if self.bm25.shape != (len(self.docs), len(self.vocabulary)):
            raise ValueError("BM25 vocabulary and corpus sizes differ")
        self._tie_order = np.argsort(np.asarray(self.doc_ids), kind="stable")
        self.device = device
        self.threads = threads
        self.cache_queries = bool(cache_queries)
        self._query_vectors = {}
        self._model = None
        self._model_path = Path(model_path)
        self._manifest = manifest

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            expected = self._manifest["model_files"]
            for relative, fingerprint in expected.items():
                if file_sha256(self._model_path / relative) != fingerprint["sha256"]:
                    raise ValueError(f"model file differs from indexed model: {relative}")
            torch.set_num_threads(self.threads)
            self._model = SentenceTransformer(str(self._model_path), device=self.device,
                                               local_files_only=True)
            self._model.max_seq_length = self._manifest["max_sequence_length"]
            self._model.float()
            self._model.eval()

    def warmup(self):
        """Load the dense model before measured calls; report separately."""
        started = time.perf_counter()
        self._load_model()
        self._model.encode([QUERY_PREFIX + "retrieval warmup"], batch_size=1,
                           normalize_embeddings=True, show_progress_bar=False,
                           convert_to_numpy=True)
        return time.perf_counter() - started

    def search(self, query, route, depth):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if route not in {"bm25", "dense"}:
            raise ValueError(f"unknown retrieval route: {route}")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError("depth must be a positive integer")
        started = time.perf_counter()
        encoding_seconds, cache_hit = 0.0, False
        if route == "bm25":
            terms = sorted({self.vocabulary[token] for token in tokenize(query)
                            if token in self.vocabulary})
            search_start = time.perf_counter()
            scores = (np.asarray(self.bm25[:, terms].sum(axis=1)).ravel()
                      if terms else np.zeros(len(self.docs), dtype=np.float32))
        else:
            self._load_model()
            encoding_start = time.perf_counter()
            cache_hit = self.cache_queries and query in self._query_vectors
            if cache_hit:
                vector = self._query_vectors[query]
            else:
                vector = self._model.encode([QUERY_PREFIX + query], batch_size=1,
                                            normalize_embeddings=True,
                                            show_progress_bar=False,
                                            convert_to_numpy=True)[0]
                vector = np.asarray(vector, dtype=np.float32)
                if self.cache_queries:
                    self._query_vectors[query] = vector
            encoding_seconds = time.perf_counter() - encoding_start
            search_start = time.perf_counter()
            scores = self.embeddings @ vector
        # Sorting the precomputed doc-ID order keeps exact-score ties stable.
        ranked = self._tie_order[np.argsort(-scores[self._tie_order], kind="stable")]
        selected = ranked[:min(depth, len(ranked))]
        search_seconds = time.perf_counter() - search_start
        results = [{"doc_id": self.doc_ids[index], "rank": rank,
                    "raw_score": float(scores[index])}
                   for rank, index in enumerate(selected, 1)]
        return {"query": query, "route": route, "requested_depth": depth,
                "returned": len(results), "results": results,
                "timing": {"query_encoding_seconds": encoding_seconds,
                           "search_seconds": search_seconds,
                           "total_seconds": time.perf_counter() - started,
                           "query_cache_hit": cache_hit,
                           "query_cache_enabled": self.cache_queries,
                           "device": self.device, "threads": self.threads}}
