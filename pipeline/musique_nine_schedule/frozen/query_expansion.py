"""Fixed, public-text query-expansion baselines with traceable sources."""
from __future__ import annotations

import re
from typing import Any


RULE_VERSION = "public_title_bridge_v1"
WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MAX_TITLE_WORDS = 8
MIN_TITLE_WORDS = 2
MAX_SOURCE_DOCUMENTS = 5
MAX_SOURCE_WORDS = 180
MAX_BRIDGE_ENTITIES = 2
MAX_QUERY_KEYWORDS = 24
RELATION_WINDOW = 5
STOPWORDS = frozenset("""a an the and or but if then of in on at to for from by
with without is are was were be been being do does did who what which when
where how why whose whom both either neither than that this these those it
its as has have had can could would should will may might also please tell
give name named called two these those each between among""".split())
RELATION_WORDS = frozenset("""directed director written writer author authored
born birthplace spouse married husband wife father mother son daughter
founded founder created creator produced producer composed composer""".split())


def words(text: str) -> list[str]:
    """Use one frozen ASCII-word tokenizer for titles, questions and sources."""
    return [match.group().lower() for match in WORD_PATTERN.finditer(text)]


def unique_keywords(tokens: list[str], excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    output = []
    for token in tokens:
        if token not in STOPWORDS and token not in excluded and token not in output:
            output.append(token)
        if len(output) >= MAX_QUERY_KEYWORDS:
            break
    return output


def containing_sentence(doc: dict[str, Any], start: int, end: int):
    """Return an original sentence and offsets for a mention inside it."""
    text = str(doc["text"])
    cursor = 0
    for sentence in doc.get("sentences", []):
        stripped = sentence.strip()
        if not stripped:
            continue
        position = text.find(stripped, cursor)
        if position < 0:
            continue
        cursor = position + len(stripped)
        if position <= start and end <= cursor:
            leading_space = len(sentence) - len(sentence.lstrip())
            return sentence, [start - position + leading_space,
                              end - position + leading_space]
    return None


class QueryExpander:
    def __init__(self, docs: list[dict[str, Any]]):
        titles: dict[tuple[str, ...], set[str]] = {}
        for doc in docs:
            title = str(doc["title"])
            key = tuple(words(title))
            if MIN_TITLE_WORDS <= len(key) <= MAX_TITLE_WORDS:
                if any(token not in STOPWORDS for token in key):
                    titles.setdefault(key, set()).add(title)
        self.title_lookup = {key: sorted(values) for key, values in titles.items()}

    def title_mentions(self, tokens: list[str]):
        """Yield longest non-overlapping dictionary matches without a corpus scan."""
        position = 0
        while position < len(tokens):
            matched = False
            maximum = min(MAX_TITLE_WORDS, len(tokens) - position)
            for size in range(maximum, MIN_TITLE_WORDS - 1, -1):
                key = tuple(tokens[position:position + size])
                titles = self.title_lookup.get(key)
                if titles:
                    for title in titles:
                        yield position, position + size, key, title
                    position += size
                    matched = True
                    break
            if not matched:
                position += 1

    def question_query(self, question_text: str) -> dict[str, Any]:
        kept = unique_keywords(words(question_text))
        return {
            "query": " ".join(kept) if kept else None,
            "provenance": {"kind": "question_keywords", "kept_keywords": kept,
                           "tokenization": "ASCII [a-z0-9]+ lowercase",
                           "rule_version": RULE_VERSION},
            "reason": "keyword_query" if kept else "no_question_keywords",
        }

    def bridge_query(self, question_text: str, initial_ids: list[str],
                     doc_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
        question_words = words(question_text)
        explicit_matches = list(self.title_mentions(question_words))
        explicit_titles = {item[3] for item in explicit_matches}
        explicit_keys = {item[2] for item in explicit_matches}
        entity_words = {token for key in explicit_keys for token in key}
        residual = unique_keywords(question_words, entity_words)
        candidates = []
        missing = []
        for rank, doc_id in enumerate(initial_ids[:MAX_SOURCE_DOCUMENTS], start=1):
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
                    continue  # Reject phrases crossing sentence boundaries.
                sentence, sentence_span = source_sentence
                preceding = words(sentence[:sentence_span[0]])[-RELATION_WINDOW:]
                relation_cues = [token for token in preceding if token in RELATION_WORDS]
                score = (4.0 * float(source_explicit)
                         + 1.5 * float(bool(relation_cues))
                         + 1.0 / rank + 0.1 / (1 + start) + 0.01 * len(key))
                candidates.append({
                    "mentioned_title": title, "entity_tokens": list(key),
                    "doc_id": doc_id, "source_title": source_title,
                    "source_rank": rank, "source_is_question_entity": source_explicit,
                    "sentence": sentence, "sentence_span": sentence_span,
                    "source_text_span": [text_start, text_end],
                    "mention_text": text[text_start:text_end],
                    "preceding_relation_cues": relation_cues, "score": score,
                    "rule_version": RULE_VERSION,
                })
        candidates.sort(key=lambda item: (-item["score"], item["source_rank"],
                                          item["source_text_span"][0],
                                          item["mentioned_title"]))
        chosen, chosen_keys, chosen_sources = [], set(), set()
        while candidates and len(chosen) < MAX_BRIDGE_ENTITIES:
            # After the first entity, prefer another explicit question-entity
            # document when available. This keeps comparison branches represented.
            new_branch = [item for item in candidates
                          if item["source_is_question_entity"]
                          and tuple(words(item["source_title"])) not in chosen_sources
                          and tuple(item["entity_tokens"]) not in chosen_keys]
            pool = new_branch if chosen and new_branch else candidates
            selected = next((item for item in pool
                             if tuple(item["entity_tokens"]) not in chosen_keys), None)
            if selected is None:
                break
            chosen.append(selected)
            chosen_keys.add(tuple(selected["entity_tokens"]))
            chosen_sources.add(tuple(words(selected["source_title"])))
            candidates = [item for item in candidates
                          if tuple(item["entity_tokens"]) not in chosen_keys]
        query_parts = [item["mentioned_title"] for item in chosen] + residual
        return {
            "query": " ".join(query_parts) if chosen else None,
            "provenance": {
                "kind": "public_title_bridge", "rule_version": RULE_VERSION,
                "entities": chosen, "residual_keywords": residual,
                "excluded_question_titles": sorted(explicit_titles),
                "source_document_limit": MAX_SOURCE_DOCUMENTS,
                "source_word_limit": MAX_SOURCE_WORDS,
                "title_word_range": [MIN_TITLE_WORDS, MAX_TITLE_WORDS],
                "missing_source_doc_ids": missing,
            },
            "reason": "traceable_bridge_query" if chosen else "no_traceable_bridge_entity",
        }
