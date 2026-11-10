"""Predeclared bridge ablations and a public-text evidence-keyword control.

The reference implementation remains unchanged. The full, top-five condition
delegates to it exactly; other bridge conditions copy only its bridge procedure
with the named switches. No fitting, gold labels, or generated queries are used.
"""
from __future__ import annotations

from query_expansion import (
    QueryExpander, WORD_PATTERN, STOPWORDS, RELATION_WORDS, RELATION_WINDOW,
    MAX_SOURCE_WORDS, MAX_TITLE_WORDS, MIN_TITLE_WORDS, RULE_VERSION,
    containing_sentence, unique_keywords, words,
)

BASE_POLICIES = ("bm25", "dense", "hybrid_rrf", "hybrid_quota",
                 "query_expand", "bridge_expand")
ABLATIONS = ("no_source_priority", "no_relation_cue", "no_second_branch",
             "one_target", "no_residual")
POLICIES = ("first_two_valid_titles", "no_second_branch")
VARIANT_VERSION = "order_two_posthoc_supplement_v1"


def variant_parameters(policy="bridge_expand", source_top_n=5):
    if policy not in POLICIES or source_top_n not in (3, 5, 10):
        raise ValueError("condition outside the frozen protocol")
    return {"variant_version": VARIANT_VERSION, "policy": policy,
            "source_document_limit": source_top_n, "source_word_limit": 180,
            "source_priority_weight": 0.0 if policy == "no_source_priority" else 4.0,
            "relation_cue_weight": 0.0 if policy == "no_relation_cue" else 1.5,
            "prefer_second_branch": policy not in ("no_second_branch", "first_two_valid_titles"),
            "max_bridge_entities": 1 if policy in ("one_target", "first_valid_title") else 2,
            "include_residual": policy != "no_residual",
            "max_residual_keywords": 24, "max_feedback_tokens": 24,
            "feedback_order": "source_rank_then_token_position",
            "feedback_excludes": "stopwords_and_all_question_tokens",
            "feedback_query_order": "evidence_tokens_then_bridge_residual_keywords",
            "feedback_missing_action": "skip_q1",
            "selection_order": "source_rank_then_character_position_then_title" if policy in ("first_valid_title", "first_two_valid_titles") else "reference_score_then_reference_ties",
            "tokenizer": "ASCII [a-z0-9]+ lowercase"}


class VariantExpander(QueryExpander):
    def __init__(self, docs=None, policy="bridge_expand", source_top_n=5,
                 title_lookup=None):
        self.params = variant_parameters(policy, source_top_n)
        self.policy = policy
        if title_lookup is None:
            super().__init__(docs)
        else:
            self.title_lookup = title_lookup

    def _question_context(self, question_text):
        question_words = words(question_text)
        matches = list(self.title_mentions(question_words))
        titles = {item[3] for item in matches}
        keys = {item[2] for item in matches}
        entity_words = {token for key in keys for token in key}
        return question_words, titles, keys, unique_keywords(question_words, entity_words)

    def bridge_query(self, question_text, initial_ids, doc_lookup):
        if self.policy == "bridge_expand" and self.params["source_document_limit"] == 5:
            return super().bridge_query(question_text, initial_ids, doc_lookup)
        if self.policy == "evidence_feedback":
            return self.feedback_query(question_text, initial_ids, doc_lookup)
        _, explicit_titles, explicit_keys, residual = self._question_context(question_text)
        if not self.params["include_residual"]:
            residual = []
        candidates, missing = [], []
        for rank, doc_id in enumerate(initial_ids[:self.params["source_document_limit"]], 1):
            doc = doc_lookup.get(doc_id)
            if doc is None:
                missing.append(doc_id)
                continue
            text = str(doc["text"])
            matches = list(WORD_PATTERN.finditer(text))[:MAX_SOURCE_WORDS]
            tokens = [match.group().lower() for match in matches]
            source_title = str(doc["title"])
            source_key = tuple(words(source_title))
            source_explicit = source_title in explicit_titles or source_key in explicit_keys
            for start, end, key, title in self.title_mentions(tokens):
                if key == source_key or key in explicit_keys:
                    continue
                text_start, text_end = matches[start].start(), matches[end - 1].end()
                source_sentence = containing_sentence(doc, text_start, text_end)
                if source_sentence is None:
                    continue
                sentence, sentence_span = source_sentence
                preceding = words(sentence[:sentence_span[0]])[-RELATION_WINDOW:]
                cues = [token for token in preceding if token in self.cue_matcher]
                score = (self.params["source_priority_weight"] * float(source_explicit)
                         + self.params["relation_cue_weight"] * float(bool(cues))
                         + 1.0 / rank + 0.1 / (1 + start) + 0.01 * len(key))
                candidates.append({"mentioned_title": title, "entity_tokens": list(key),
                    "doc_id": doc_id, "source_title": source_title, "source_rank": rank,
                    "source_is_question_entity": source_explicit, "sentence": sentence,
                    "sentence_span": sentence_span, "source_text_span": [text_start, text_end],
                    "mention_text": text[text_start:text_end],
                    "preceding_relation_cues": cues, "score": score, "rule_version": RULE_VERSION})
        candidates.sort(key=lambda item: (-item["score"], item["source_rank"],
                            item["source_text_span"][0], item["mentioned_title"]))
        if self.policy in ("first_valid_title", "first_two_valid_titles"):
            candidates.sort(key=lambda item: (item["source_rank"], item["source_text_span"][0], item["mentioned_title"]))
        chosen, chosen_keys, chosen_sources = [], set(), set()
        while candidates and len(chosen) < self.params["max_bridge_entities"]:
            new_branch = [item for item in candidates
                          if item["source_is_question_entity"]
                          and tuple(words(item["source_title"])) not in chosen_sources
                          and tuple(item["entity_tokens"]) not in chosen_keys]
            pool = (new_branch if self.params["prefer_second_branch"]
                    and chosen and new_branch else candidates)
            selected = next((item for item in pool
                             if tuple(item["entity_tokens"]) not in chosen_keys), None)
            if selected is None:
                break
            chosen.append(selected)
            chosen_keys.add(tuple(selected["entity_tokens"]))
            chosen_sources.add(tuple(words(selected["source_title"])))
            candidates = [item for item in candidates
                          if tuple(item["entity_tokens"]) not in chosen_keys]
        parts = [item["mentioned_title"] for item in chosen] + residual
        return {"query": " ".join(parts) if chosen else None,
                "provenance": {"kind": "public_title_bridge", "rule_version": RULE_VERSION,
                    "entities": chosen, "residual_keywords": residual,
                    "excluded_question_titles": sorted(explicit_titles),
                    "source_document_limit": self.params["source_document_limit"],
                    "source_word_limit": MAX_SOURCE_WORDS,
                    "title_word_range": [MIN_TITLE_WORDS, MAX_TITLE_WORDS],
                    "missing_source_doc_ids": missing},
                "reason": "traceable_bridge_query" if chosen else "no_traceable_bridge_entity"}

    def feedback_query(self, question_text, initial_ids, doc_lookup):
        question_tokens, explicit_titles, _, residual = self._question_context(question_text)
        excluded = set(question_tokens) | set(STOPWORDS)
        selected, seen, missing = [], set(), []
        for rank, doc_id in enumerate(initial_ids[:self.params["source_document_limit"]], 1):
            doc = doc_lookup.get(doc_id)
            if doc is None:
                missing.append(doc_id)
                continue
            matches = list(WORD_PATTERN.finditer(str(doc["text"])))[:MAX_SOURCE_WORDS]
            for position, match in enumerate(matches):
                token = match.group().lower()
                if token in excluded or token in seen:
                    continue
                selected.append({"token": token, "doc_id": doc_id,
                    "source_title": str(doc["title"]), "source_rank": rank,
                    "token_position": position, "source_text_span": [match.start(), match.end()],
                    "mention_text": match.group()})
                seen.add(token)
                if len(selected) == 24:
                    break
            if len(selected) == 24:
                break
        evidence = [item["token"] for item in selected]
        return {"query": " ".join(evidence + residual) if evidence else None,
                "provenance": {"kind": "predeclared_evidence_keyword_feedback",
                    "rule_version": VARIANT_VERSION, "evidence_tokens": evidence,
                    "token_sources": selected, "residual_keywords": residual,
                    "excluded_question_titles": sorted(explicit_titles),
                    "source_document_limit": self.params["source_document_limit"],
                    "source_word_limit": MAX_SOURCE_WORDS,
                    "missing_source_doc_ids": missing,
                    "uses_labels": False, "official_AIR_or_RM3": False},
                "reason": "evidence_keyword_feedback" if evidence else "no_new_evidence_tokens"}
