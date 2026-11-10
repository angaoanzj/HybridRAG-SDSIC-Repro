"""Two targeted controls and label-blind diagnostics using frozen primitives."""
from pathlib import Path
import hashlib
import json
import time

from query_expansion import (WORD_PATTERN, MAX_SOURCE_WORDS, RELATION_WINDOW,
                             RELATION_WORDS, words, containing_sentence)

BASE_POLICIES = ("hybrid_rrf", "one_target", "first_two_valid_titles",
                 "no_second_branch", "bridge_expand", "evidence_feedback")
DIAGNOSTIC_POLICIES = ("repeat_q0", "score_two_reverse")
TIMING_POLICIES = ("hybrid_rrf", "score2_u24", "family_u24")
K_VALUES = (10, 20, 30, 50)


def verify_protocol(path, expected_sha256):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Frozen protocol SHA-256 mismatch")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError("Frozen protocol must be a JSON object")
    hashes = record.get("code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Frozen protocol requires code_sha256 for every code/*.py")
    normalized = {Path(name).name: item.get("sha256") if isinstance(item, dict) else item
                  for name, item in hashes.items()}
    for code in Path(__file__).resolve().parent.glob("*.py"):
        actual = hashlib.sha256(code.read_bytes()).hexdigest()
        if normalized.get(code.name) != actual:
            raise ValueError(f"Code changed after protocol freeze: {code.name}")
    return digest


def retrieve_group(backend, expander, question, policy, d_budget, alpha):
    from run_retrieval import (retrieve_group as reference_retrieve,
                               measured_call, fused_rank)
    if policy != "repeat_q0":
        return reference_retrieve(backend, expander, question, policy, d_budget, alpha)
    if d_budget != 400:
        raise ValueError("repeat_q0 is frozen at D=400")
    started = time.perf_counter()
    calls = []
    original = str(question["question"])
    fusion_seconds = 0.0
    for query_id in ("q0", "q1"):
        # These are four actual backend.search invocations. No result/timing
        # copying and no query-vector cache are permitted by the runner.
        for route in ("bm25", "dense"):
            calls.append(measured_call(backend, original, route, 100, query_id,
                                       f"repeat_{question['id']}_{len(calls)}"))
        if query_id == "q0":
            before = time.perf_counter()
            initial = fused_rank(calls, alpha)
            fusion_seconds += time.perf_counter() - before
    before = time.perf_counter()
    ranking = fused_rank(calls, alpha)
    fusion_seconds += time.perf_counter() - before
    return {"calls": calls, "ranking": ranking, "initial": initial,
            "expanded_query": original,
            "expansion_provenance": {"kind": "forced_actual_original_query_repeat",
                                     "entities": [], "residual_keywords": [],
                                     "forced_unchanged_query": True},
            "reason": "actual_q0_repeated_as_q1",
            "query_generation_seconds": 0.0,
            "fusion_seconds": fusion_seconds,
            "total_seconds": time.perf_counter() - started}


def candidate_pool_audit(expander, question_text, initial_ids, doc_lookup):
    """Enumerate the unchanged eligible pool outside measured retrieval scope.

    Counts apply before any title selection. This helper uses public text and
    the original matching/sentence-validity primitives; it never sees labels.
    """
    reference = expander.delegate
    _, explicit_titles, explicit_keys, _ = reference._question_context(question_text)
    occurrences = []
    missing = []
    for rank, doc_id in enumerate(initial_ids[:5], 1):
        doc = doc_lookup.get(doc_id)
        if doc is None:
            missing.append(doc_id)
            continue
        text = str(doc["text"])
        matches = list(WORD_PATTERN.finditer(text))[:MAX_SOURCE_WORDS]
        tokens = [match.group().lower() for match in matches]
        source_key = tuple(words(doc["title"]))
        explicit = doc["title"] in explicit_titles or source_key in explicit_keys
        for start, end, key, title in reference.title_mentions(tokens):
            if key == source_key or key in explicit_keys:
                continue
            a, b = matches[start].start(), matches[end - 1].end()
            source_sentence = containing_sentence(doc, a, b)
            if source_sentence is None:
                continue
            sentence, sentence_span = source_sentence
            preceding = words(sentence[:sentence_span[0]])[-RELATION_WINDOW:]
            has_cue = any(token in reference.cue_matcher for token in preceding)
            score = (4.0 * float(explicit) + 1.5 * float(has_cue)
                     + 1.0 / rank + 0.1 / (1 + start) + 0.01 * len(key))
            occurrences.append({"entity_key": list(key), "mentioned_title": title,
                                "source_doc_id": doc_id,
                                "source_family": list(source_key),
                                "source_rank": rank, "character_start": a,
                                "reference_score": score})
    distinct_keys = {tuple(item["entity_key"]) for item in occurrences}
    return {"candidate_pool_occurrence_count": len(occurrences),
            "candidate_pool_distinct_count": len(distinct_keys),
            "candidate_pool_distinct_keys": [list(key) for key in sorted(distinct_keys)],
            "candidate_occurrences": occurrences,
            "missing_source_doc_ids": missing,
            "pool_scope": "initial_top5_first180_words_before_title_selection"}


def group_diagnostics(backend, expander, question, group, mode):
    before = time.perf_counter()
    provenance = group.get("expansion_provenance") or {}
    selected = provenance.get("entities", []) if isinstance(provenance, dict) else []
    result = {"selected_count": len(selected),
              "selected_titles": [item["mentioned_title"] for item in selected],
              "selected_keys": [item["entity_tokens"] for item in selected],
              "selected_feedback_token_count": len(provenance.get("evidence_tokens", []))
                  if isinstance(provenance, dict) else 0,
              "initial_selections": {"50": group["initial"][:50]},
              "first_round_F50": None,
              "first_round_F50_status": "offline_evaluator_requires_separate_labels",
              "diagnostics_excluded_from_total_seconds": True}
    if mode == "timing":
        result.update({"candidate_pool_distinct_count": None,
                       "candidate_pool_status": "not_enumerated_in_timing_mode"})
    else:
        result.update(candidate_pool_audit(expander, question["question"],
                                          group["initial"], backend.doc_lookup))
    result["diagnostic_seconds"] = time.perf_counter() - before
    return result
