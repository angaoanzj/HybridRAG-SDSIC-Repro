"""Synthetic fixtures only; no dataset questions, model, or effectiveness test."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import types

from expansion_variants import (QueryExpander, VariantExpander, POLICIES,
                                BASE_POLICIES)
from frozen_entity_variants import VariantExpander as EntityExpander
from frozen_order_variants import VariantExpander as OrderExpander
from frozen_core_variants import VariantExpander as CoreExpander
from reviewer_runner import execute_group, group_identity, plan_conditions
from reviewer_extensions import retrieve_group, verify_protocol


def main():
    # The frozen retrieval module imports the backend class at module load.
    # Synthetic checks exercise its unchanged pure call/fusion functions while
    # making real backend/model construction impossible, with no SciPy required.
    stub = types.ModuleType("retrieval_backends")

    class ForbiddenRealBackend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("synthetic checks cannot construct a real backend")

    stub.RetrievalBackend = ForbiddenRealBackend
    stub.file_sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    stub.write_json = lambda p, value: Path(p).write_text(json.dumps(value))
    sys.modules["retrieval_backends"] = stub
    from run_retrieval import quota_selection
    docs = [
        {"doc_id": "d0", "title": "Silver River",
         "text": "Silver River mentions Roman Writer. It was written by Jane Author. Roman Writer returned.",
         "sentences": ["Silver River mentions Roman Writer.", "It was written by Jane Author.", "Roman Writer returned."]},
        {"doc_id": "d1", "title": "Golden Book",
         "text": "Golden Book was written by Helen Poet.",
         "sentences": ["Golden Book was written by Helen Poet."]},
        {"doc_id": "d2", "title": "Roman Writer", "text": "Roman Writer lived in Rome.",
         "sentences": ["Roman Writer lived in Rome."]},
        {"doc_id": "d3", "title": "Jane Author", "text": "Jane Author lived in Paris.",
         "sentences": ["Jane Author lived in Paris."]},
        {"doc_id": "d4", "title": "Helen Poet", "text": "Helen Poet lived in Athens.",
         "sentences": ["Helen Poet lived in Athens."]},
    ]
    docs += [{"doc_id": f"z{i}", "title": f"Filler Item {i}",
              "text": "Unrelated filler.", "sentences": ["Unrelated filler."]}
             for i in range(500)]
    lookup = {doc["doc_id"]: doc for doc in docs}
    title_lookup = QueryExpander(docs).title_lookup
    expanders = {p: VariantExpander(policy=p, title_lookup=title_lookup) for p in POLICIES}
    question = {"id": "synthetic_fixture", "question":
                "Were Silver River and Golden Book written by the same author?",
                "split": "synthetic_technical"}

    for policy in BASE_POLICIES:
        cls = (OrderExpander if policy == "first_two_valid_titles" else
               CoreExpander if policy == "evidence_feedback" else EntityExpander)
        reference = cls(policy=policy, title_lookup=title_lookup)
        assert expanders[policy].params == reference.params
        if policy != "hybrid_rrf":
            for initial in ([], ["d1"], ["d0", "d1"]):
                assert expanders[policy].bridge_query(question["question"], initial, lookup) == \
                    reference.bridge_query(question["question"], initial, lookup)

    normal = expanders["no_second_branch"].bridge_query(question["question"], ["d0", "d1"], lookup)
    reverse = expanders["score_two_reverse"].bridge_query(question["question"], ["d0", "d1"], lookup)
    entities = normal["provenance"]["entities"]
    assert len(entities) == 2
    assert reverse["provenance"]["entities"] == list(reversed(entities))
    assert reverse["provenance"]["residual_keywords"] == normal["provenance"]["residual_keywords"]
    assert reverse["provenance"]["reviewer_order_control"]["reference_entities"] == entities
    for initial in ([], ["d1"]):
        a = expanders["no_second_branch"].bridge_query(question["question"], initial, lookup)
        b = expanders["score_two_reverse"].bridge_query(question["question"], initial, lookup)
        assert a["query"] == b["query"]
        assert a["provenance"]["entities"] == b["provenance"]["entities"]

    class FakeBackend:
        doc_lookup = lookup

        def __init__(self):
            self.actual_searches = []

        def search(self, query, route, depth):
            self.actual_searches.append((query, route, depth))
            results = [{"doc_id": doc["doc_id"], "rank": n + 1, "raw_score": 1000 - n}
                       for n, doc in enumerate(docs[:depth])]
            return {"query": query, "route": route, "requested_depth": depth,
                    "returned": len(results), "results": results,
                    "timing": {"query_encoding_seconds": 0., "search_seconds": 0.,
                               "total_seconds": 0., "query_cache_hit": False,
                               "query_cache_enabled": False, "device": "synthetic",
                               "threads": 8}}

    budgets = {}
    for policy, expander in expanders.items():
        backend = FakeBackend()
        identity = group_identity("2wiki", question, "confirmation",
                                  {"policy": policy, "alpha": .3, "repeat": 0, "source_top_n": 5})
        row = execute_group(backend, expander, question, identity, retrieve_group, quota_selection)
        assert row["status"] == "complete", row
        expected_calls = 2 if policy == "hybrid_rrf" else 4
        assert row["j_count"] == len(backend.actual_searches) == expected_calls
        assert row["q_count"] == (1 if policy == "hybrid_rrf" else 2)
        assert row["d_count"] == 400
        assert set(row["selections"]) == {"10", "20", "30", "50"}
        for k in (10, 20, 30, 50):
            assert row["selections"][str(k)] == row["ranking"][:k]
        diag = row["reviewer_diagnostics"]
        assert diag["candidate_pool_distinct_count"] >= diag["selected_count"]
        assert diag["first_round_F50"] is None
        assert diag["initial_selections"]["50"] == row["initial"][:50]
        if policy == "repeat_q0":
            assert all(call[0] == question["question"] and call[2] == 100
                       for call in backend.actual_searches)
            assert [call["query_id"] for call in row["calls"]] == ["q0", "q0", "q1", "q1"]
            assert row["initial"] == row["ranking"]
            assert row["calls"][0]["returned"] is not row["calls"][2]["returned"]
        budgets[policy] = {"actual_synthetic_searches": expected_calls, "returned_records": 400}

    timing = [plan_conditions("timing", "2wiki", q, 20261106) for q in range(100)]
    orders = [tuple(x["policy"] for x in group[k:k + 3])
              for group in timing for k in range(0, 15, 3)]
    counts = Counter(orders)
    assert len(counts) == 6 and max(counts.values()) - min(counts.values()) <= 1
    assert all(len(group) == 15 for group in timing)
    assert Counter(x["policy"] for group in timing for x in group) == \
        {"hybrid_rrf": 500, "no_second_branch": 500, "bridge_expand": 500}

    with tempfile.TemporaryDirectory(prefix="reviewer_freeze_fixture_") as temporary:
        path = Path(temporary) / "protocol.json"
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(__file__).parent.glob("*.py")}
        path.write_text(json.dumps({"code_sha256": hashes}))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert verify_protocol(path, digest) == digest
        try:
            verify_protocol(path, "0" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid freeze hash accepted")

    return {"status": "passed", "scope": "synthetic fixtures only",
            "dataset_questions_loaded": False, "model_loaded": False,
            "retrieval_experiment_run": False, "effectiveness_scoring_run": False,
            "real_backend_construction_forbidden_by_synthetic_stub": True,
            "base_six_parameters_and_query_outputs_equal_original": True,
            "reverse_changes_only_two_selected_title_order": True,
            "zero_and_one_title_fallback_preserved": True,
            "repeat_forces_four_actual_fake_backend_searches": True,
            "K_caps_share_identical_calls": True,
            "candidate_counts_and_label_blind_first_round_ids": True,
            "six_timing_orders_cycle_with_500_repeats_per_method": True,
            "protocol_and_source_hash_gate": True, "budgets": budgets}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
