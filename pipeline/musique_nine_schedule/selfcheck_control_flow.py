"""Synthetic software control-flow checks. No retrieval or scientific results."""
import json
from types import SimpleNamespace
from run_multiround import ARMS, retrieve


def fake_call(backend, query, route, depth, query_id, call_id):
    return {"query": query, "route": route, "depth": depth, "query_id": query_id,
            "call_id": call_id, "cache_hit": False,
            "returned": [{"doc_id": f"synthetic:{query}:{route}:{i}", "rank": i + 1} for i in range(depth)]}


def fake_fusion(calls, alpha):
    assert alpha == .5
    return list(dict.fromkeys(row["doc_id"] for call in calls for row in call["returned"]))


class FakeExpander:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def bridge_query(self, question_text, initial_ids, doc_lookup):
        assert isinstance(question_text, str)
        assert all(isinstance(x, str) for x in initial_ids)
        assert doc_lookup == {}  # No gold labels or scientific data are supplied.
        query = next(self.outputs)
        return {"query": query, "provenance": {"entities": []}}


def main():
    question = {"id": "synthetic-only", "question": "original", "split": "synthetic"}
    backend = SimpleNamespace(doc_lookup={})
    full = {}
    for arm, config in ARMS.items():
        row = retrieve(backend, FakeExpander(["next1", "next2", "next3"]), question, arm, fake_call, fake_fusion)
        assert row["requested_records"] == 2 * sum(config["depths"])
        assert row["returned_records"] == row["requested_records"] <= 400
        assert row["physical_route_calls"] == 2 * len(config["depths"])
        full[arm] = row
    for arm in ARMS:
        row = retrieve(backend, FakeExpander([None]), question, arm, fake_call, fake_fusion)
        assert row["supplementary_queries"] == 0 and row["physical_route_calls"] == 2
    row = retrieve(backend, FakeExpander(["next1", "  NEXT1  "]), question, "iter2_100_50_50", fake_call, fake_fusion)
    assert row["supplementary_queries"] == 1 and row["returned_records"] == 300
    row = retrieve(backend, FakeExpander(["original"]), question, "single100", fake_call, fake_fusion)
    assert row["physical_route_calls"] == 2
    repeated = full["repeat2_100_50_50"]
    prefix = full["stop100_50"]
    assert repeated["rounds"][1]["query"] == repeated["rounds"][2]["query"]
    assert set(repeated["ranking"]) == set(prefix["ranking"])
    repeated3 = full["repeat3_50_50_50_50"]
    assert repeated3["rounds"][2]["query"] == repeated3["rounds"][3]["query"]
    for longer, shorter in [("iter2_100_50_50", "stop100_50"), ("iter3_50_50_50_50", "stop50_50_50")]:
        assert full[longer]["rounds"][:len(full[shorter]["rounds"])] == full[shorter]["rounds"]
    print(json.dumps({"status": "SOFTWARE_SELFCHECK_PASSED", "synthetic_only": True,
        "scientific_results": False, "checks": ["9 arm budgets", "empty expansion early stop",
        "unchanged q1", "normalized duplicate later query", "repeat has no new candidates", "matched prefixes"],
        "inference_or_real_retrieval_executed": False}, indent=2))


if __name__ == "__main__":
    main()
