#!/usr/bin/env python3
"""Evaluate frozen global-corpus retrieval runs without invoking a model.

The main metrics use distinct annotated support titles. Strict fact metrics
additionally check the original support sentences in the returned text version.
All rates in machine-readable output are fractions. HTML/Markdown display
percentages. Paired bootstrap units are questions in the shared global corpus.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


RATE_METRICS = (
    "support_recall", "complete_support", "strict_fact_recall",
    "strict_complete_support", "support_sentence_recall", "title_precision",
    "duplicate_fraction", "cumulative_support_recall", "initial_support_recall",
)
COUNT_METRICS = (
    "n_selected", "n_selected_titles", "q_count", "j_count", "d_count",
    "new_support", "lost_support", "net_support", "cache_hit_calls",
)
TIME_METRICS = (
    "total_seconds", "fusion_seconds", "query_generation_seconds",
    "query_encode_seconds", "search_seconds", "call_total_seconds",
)
PAIR_METRICS = ("support_recall", "complete_support", "strict_fact_recall")
LIMITATIONS = [
    "这是验证集上下文联合公共语料上的诊断先导，结果范围为该受控检索语料。",
    "本轮主终点为证据检索，回答质量需要另行生成实验支持。",
    "支持标题覆盖按每题不同标注标题计算，同标题多版本只计一次。",
    "严格事实覆盖要求至少一个返回版本完整包含该标题对应的全部标注支持原句，仅归一化空白。",
    "未标注文档可能仍有用，标注精确率受支持标签完整度影响。",
    "同一返回行数或调用上限仍可能具有不同计算成本，实际用量和缓存状态另报。",
    "计时范围按运行器记录，文档编码、索引构建和模型初始化成本由运行清单另行说明。",
    "配对区间以问题为重采样单位，共享实体和主题可能带来额外相关性。",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def normalized_whitespace(value: str) -> str:
    return " ".join(value.split())


def nonnegative_number(value: Any, name: str, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def integer(value: Any, name: str, minimum: int = 0) -> int:
    result = nonnegative_number(value, name)
    if result is None or result != int(result) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def describe(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(len(array)), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
        "min": float(array.min()), "max": float(array.max()),
    }


def load_dataset(folder: Path) -> dict[str, Any]:
    corpus_path, queries_path = folder / "corpus.jsonl", folder / "queries.json"
    documents, by_title = {}, collections.defaultdict(list)
    with corpus_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            document = json.loads(line)
            doc_id = str(document["doc_id"])
            if doc_id in documents:
                raise ValueError(f"duplicate corpus doc_id at {corpus_path}:{line_number}")
            if not isinstance(document.get("title"), str) or not isinstance(document.get("text"), str):
                raise ValueError("corpus title/text must be strings")
            document["normalized_text"] = normalized_whitespace(document["text"])
            documents[doc_id] = document
            by_title[document["title"]].append(doc_id)
    query_file = read_json(queries_path)
    raw_queries = query_file["queries"]
    questions, annotations = {}, {}
    for question in raw_queries:
        query_id = str(question["id"])
        if query_id in questions:
            raise ValueError(f"duplicate question ID in {queries_path}")
        gold_titles = set(question["gold_titles"])
        if not gold_titles or any(not isinstance(t, str) for t in gold_titles):
            raise ValueError(f"invalid gold titles for {query_id}")
        gold_doc_ids = set(question.get("gold_doc_ids", []))
        if not gold_doc_ids <= documents.keys():
            raise ValueError(f"source gold_doc_ids absent from corpus for {query_id}")
        support = question.get("supporting_sentences_by_title", {})
        annotation_complete = all(
            isinstance(support.get(title), list) and support[title]
            and all(isinstance(sentence, str) and sentence.strip() for sentence in support[title])
            for title in gold_titles
        )
        fact_ids, sentence_matches = set(), {}
        expected_sentences = set()
        if annotation_complete:
            for title in gold_titles:
                sentences = {normalized_whitespace(s) for s in support[title]}
                expected_sentences.update((title, sentence) for sentence in sentences)
                for doc_id in by_title.get(title, []):
                    text = documents[doc_id]["normalized_text"]
                    hits = {(title, sentence) for sentence in sentences if sentence in text}
                    sentence_matches[doc_id] = hits
                    if len(hits) == len(sentences):
                        fact_ids.add(doc_id)
        reachable_titles = gold_titles & by_title.keys()
        reachable_fact_titles = {documents[d]["title"] for d in fact_ids}
        reachable_sentences = set().union(*sentence_matches.values()) if sentence_matches else set()
        annotations[query_id] = {
            "gold_titles": gold_titles, "fact_ids": fact_ids,
            "sentence_matches": sentence_matches, "expected_sentences": expected_sentences,
            "fact_annotation_complete": annotation_complete,
            "corpus_title_recall": len(reachable_titles) / len(gold_titles),
            "corpus_complete_support": float(gold_titles <= by_title.keys()),
            "corpus_strict_fact_recall": len(reachable_fact_titles) / len(gold_titles)
                if annotation_complete else None,
            "corpus_strict_complete_support": float(gold_titles <= reachable_fact_titles)
                if annotation_complete else None,
            "corpus_sentence_recall": len(reachable_sentences) / len(expected_sentences)
                if expected_sentences else None,
        }
        questions[query_id] = question
    return {
        "documents": documents, "questions": questions, "annotations": annotations,
        "corpus_sha256": sha256(corpus_path), "queries_sha256": sha256(queries_path),
        "query_protocol": query_file.get("protocol", {}),
    }


def configuration_key(configuration: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(configuration["policy"]), integer(configuration["k"], "k", 1),
        integer(configuration["d_budget"], "d_budget", 1),
        integer(configuration["j_budget"], "j_budget", 1),
    )


def group_key(dataset: str, split: str, configuration: dict[str, Any]) -> tuple[Any, ...]:
    return (dataset, split) + configuration_key(configuration)


def group_name(key: tuple[Any, ...]) -> str:
    dataset, split, policy, k, d_budget, j_budget = key
    return f"{dataset}/{split}/{policy}/K{k}/D{d_budget}/J{j_budget}"


def expected_groups(protocol: dict[str, Any], datasets: dict[str, Any]) -> dict[tuple[Any, ...], set[str]]:
    configurations = protocol.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("manifest.protocol.configurations must be a nonempty list")
    expected = {}
    global_splits = set(protocol.get("splits", ["dev", "test"]))
    query_subset = protocol.get("query_ids", {})
    for configuration in configurations:
        configuration_key(configuration)
        names = configuration.get("datasets", protocol.get("datasets", list(datasets)))
        if "dataset" in configuration:
            names = [configuration["dataset"]]
        splits = set(configuration.get("splits", global_splits))
        if "split" in configuration:
            splits = {configuration["split"]}
        for name in names:
            if name not in datasets:
                raise ValueError(f"configuration refers to absent dataset {name}")
            questions = datasets[name]["questions"]
            allowed = set(query_subset[name]) if name in query_subset else set(questions)
            if not allowed <= questions.keys():
                raise ValueError(f"manifest query_ids outside {name}")
            for split in splits:
                ids = {qid for qid in allowed if questions[qid]["split"] == split}
                if ids:
                    key = group_key(name, split, configuration)
                    if key in expected:
                        raise ValueError(f"duplicate manifest configuration {group_name(key)}")
                    expected[key] = ids
    if not expected:
        raise ValueError("manifest configurations select no questions")
    return expected


def id_list(value: Any, name: str, known: dict[str, Any]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a list of document IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate IDs")
    if not set(value) <= known.keys():
        raise ValueError(f"{name} contains unknown corpus IDs")
    return value


def enrich_row(raw: dict[str, Any], data: dict[str, Any],
               call_lookup: dict[str, Any] | None = None) -> dict[str, Any]:
    query_id = str(raw["id"])
    questions, documents = data["questions"], data["documents"]
    if query_id not in questions:
        raise ValueError(f"unknown query ID {query_id}")
    question, annotation = questions[query_id], data["annotations"][query_id]
    if raw["split"] != question["split"]:
        raise ValueError(f"query split mismatch for {query_id}")
    policy, k, d_budget, j_budget = configuration_key(raw)
    selected = id_list(raw["selected_ids"], "selected_ids", documents)
    cumulative = id_list(raw["cumulative_ids"], "cumulative_ids", documents)
    initial = id_list(raw["initial_ids"], "initial_ids", documents)
    if len(selected) > k or len(initial) > k:
        raise ValueError(f"selected/initial IDs exceed K for {query_id}")
    if not set(selected + initial) <= set(cumulative):
        raise ValueError(f"selected/initial IDs outside cumulative candidates for {query_id}")
    raw_calls = raw.get("calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        raise ValueError(f"nonempty calls list required for {query_id}")
    calls, call_ids = [], []
    for value in raw_calls:
        if isinstance(value, str):
            if call_lookup is None or value not in call_lookup:
                raise ValueError(f"unknown call_id {value} for {query_id}")
            calls.append(call_lookup[value])
            call_ids.append(value)
        elif isinstance(value, dict):
            calls.append(value)
            call_ids.append(str(value.get("call_id", f"inline_{len(calls)}")))
        else:
            raise ValueError("calls must contain call IDs or inline records")
    if len(call_ids) != len(set(call_ids)):
        raise ValueError(f"one result references a retrieval call more than once for {query_id}")
    query_ids, returned_ids = set(), set()
    query_texts = {}
    d_count = 0
    call_times = {name: 0.0 for name in ("query_encode_seconds", "search_seconds", "total_seconds")}
    cache_hits = 0
    for call in calls:
        call_query_id = str(call["query_id"])
        query_ids.add(call_query_id)
        query = call.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("call query must be a nonempty string")
        if call_query_id in query_texts and query_texts[call_query_id] != query:
            raise ValueError("one query_id refers to inconsistent query texts")
        query_texts[call_query_id] = query
        returned = call.get("returned")
        if not isinstance(returned, list):
            raise ValueError("call.returned must be a list")
        depth = integer(call["depth"], "call.depth", 1)
        if len(returned) > depth:
            raise ValueError("call returned more rows than requested depth")
        returned_doc_ids = [str(item["doc_id"]) for item in returned]
        id_list(returned_doc_ids, "call.returned IDs", documents)
        if [integer(item["rank"], "rank", 1) for item in returned] != list(range(1, len(returned) + 1)):
            raise ValueError("call ranks must be contiguous and one-based")
        if any(not math.isfinite(float(item["raw_score"])) for item in returned):
            raise ValueError("nonfinite retrieval score")
        returned_ids.update(returned_doc_ids)
        d_count += len(returned)
        timing = call.get("timings", {})
        for name in call_times:
            call_times[name] += nonnegative_number(call.get(name, timing.get(name)), name, 0.0)
        cache_hits += int(bool(call.get("cache_hit", False)))
    if set(cumulative) != returned_ids:
        raise ValueError(f"cumulative_ids differ from actual returned records for {query_id}")
    counts = {"q_count": len(query_ids), "j_count": len(calls), "d_count": d_count}
    for name, observed in counts.items():
        if integer(raw[name], name) != observed:
            raise ValueError(f"{name} mismatch for {query_id}")
    if counts["j_count"] > j_budget or d_count > d_budget:
        raise ValueError(f"retrieval budget exceeded for {query_id}")
    if not raw.get("signature"):
        raise ValueError(f"missing execution signature for {query_id}")
    gold = annotation["gold_titles"]
    selected_titles = {documents[doc_id]["title"] for doc_id in selected}
    cumulative_titles = {documents[doc_id]["title"] for doc_id in cumulative}
    initial_titles = {documents[doc_id]["title"] for doc_id in initial}
    hit_titles = gold & selected_titles
    initial_hits = gold & initial_titles
    fact_titles = {documents[d]["title"] for d in selected if d in annotation["fact_ids"]}
    sentence_hits = set().union(*(annotation["sentence_matches"].get(d, set()) for d in selected))
    fact_valid = annotation["fact_annotation_complete"]
    expected_sentences = annotation["expected_sentences"]
    row = {
        "id": query_id, "dataset": str(raw["dataset"]), "split": str(raw["split"]),
        "policy": policy, "k": k, "d_budget": d_budget, "j_budget": j_budget,
        "type": question.get("type", "unknown"), "n_gold": len(gold),
        "support_recall": len(hit_titles) / len(gold),
        "complete_support": float(gold <= selected_titles),
        "strict_fact_recall": len(fact_titles) / len(gold) if fact_valid else None,
        "strict_complete_support": float(gold <= fact_titles) if fact_valid else None,
        "support_sentence_recall": len(sentence_hits) / len(expected_sentences)
            if expected_sentences else None,
        "title_precision": len(hit_titles) / len(selected_titles) if selected_titles else 0.0,
        "n_selected": len(selected), "n_selected_titles": len(selected_titles),
        "cumulative_support_recall": len(gold & cumulative_titles) / len(gold),
        "initial_support_recall": len(initial_hits) / len(gold),
        "initial_complete_support": float(gold <= initial_titles),
        "new_support": len(hit_titles - initial_hits),
        "lost_support": len(initial_hits - hit_titles),
        "net_support": len(hit_titles) - len(initial_hits),
        "new_complete_support": int(gold <= selected_titles and not gold <= initial_titles),
        "lost_complete_support": int(gold <= initial_titles and not gold <= selected_titles),
        "duplicate_fraction": 1 - len(cumulative) / d_count if d_count else 0.0,
        **counts, "cache_hit_calls": cache_hits,
        "selected_ids": selected, "selected_titles": sorted(selected_titles),
        "initial_ids": initial, "hit_titles": sorted(hit_titles),
        "initial_hit_titles": sorted(initial_hits),
        "new_support_titles": sorted(hit_titles - initial_hits),
        "lost_support_titles": sorted(initial_hits - hit_titles),
        "fact_hit_titles": sorted(fact_titles),
        "total_seconds": nonnegative_number(raw.get("total_seconds"), "total_seconds"),
        "fusion_seconds": nonnegative_number(raw.get("fusion_seconds"), "fusion_seconds"),
        "query_generation_seconds": nonnegative_number(raw.get("query_generation_seconds"),
                                                        "query_generation_seconds"),
        "query_encode_seconds": call_times["query_encode_seconds"],
        "search_seconds": call_times["search_seconds"],
        "call_total_seconds": call_times["total_seconds"],
        "expanded_query": raw.get("expanded_query"),
        "expansion_provenance": raw.get("expansion_provenance"),
        "reason": raw.get("reason", ""), "signature": raw["signature"], "call_ids": call_ids,
        "corpus_title_reachable": annotation["corpus_complete_support"],
        "corpus_fact_reachable": annotation["corpus_strict_complete_support"],
    }
    return row


def aggregate(rows: list[dict[str, Any]], expected_ids: set[str]) -> dict[str, Any]:
    seen = {row["id"] for row in rows}
    result = {
        "n": len(rows), "expected_n": len(expected_ids), "complete": seen == expected_ids,
        "missing_ids": sorted(expected_ids - seen),
        "target_k_shortfalls": sum(row["n_selected"] < row["k"] for row in rows),
        "signature_count": len({row["signature"] for row in rows}),
        "type_counts": dict(collections.Counter(row["type"] for row in rows)),
        "new_complete_support_count": sum(row["new_complete_support"] for row in rows),
        "lost_complete_support_count": sum(row["lost_complete_support"] for row in rows),
    }
    for name in RATE_METRICS + COUNT_METRICS + TIME_METRICS:
        result[name] = describe([row.get(name) for row in rows])
    for reachable_name, label in (("corpus_title_reachable", "title_reachable"),
                                  ("corpus_fact_reachable", "fact_reachable")):
        conditional = [row for row in rows if row[reachable_name] == 1]
        result[label] = {"n": len(conditional)}
        for name in ("support_recall", "complete_support", "strict_fact_recall", "strict_complete_support"):
            result[label][name] = describe([row[name] for row in conditional])
    return result


def paired_bootstrap(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]],
                     samples: int, seed: int) -> dict[str, Any]:
    a, b = {row["id"]: row for row in candidate}, {row["id"]: row for row in baseline}
    ids = sorted(a.keys() & b.keys())
    result = {
        "n_pairs": len(ids), "candidate_only_ids": sorted(a.keys() - b.keys()),
        "baseline_only_ids": sorted(b.keys() - a.keys()),
        "resampling_unit": "question", "bootstrap_samples": samples, "seed": seed,
        "complete_pairing": a.keys() == b.keys(),
    }
    for metric in PAIR_METRICS:
        differences = np.asarray([a[qid][metric] - b[qid][metric] for qid in ids
                                  if a[qid][metric] is not None and b[qid][metric] is not None])
        if not len(differences):
            result[metric] = {"n": 0, "difference": None, "ci95": None}
            continue
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, len(differences), size=(samples, len(differences)))
        sampled = differences[indices].mean(axis=1)
        result[metric] = {
            "n": len(differences), "difference": float(differences.mean()),
            "ci95": [float(v) for v in np.percentile(sampled, [2.5, 97.5])],
            "wins": int(np.sum(differences > 1e-12)),
            "losses": int(np.sum(differences < -1e-12)),
            "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        }
    return result


def paired_cases(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]],
                 questions: dict[str, Any]) -> dict[str, Any]:
    a, b = {row["id"]: row for row in candidate}, {row["id"]: row for row in baseline}
    output = {"rescued": [], "lost": []}
    for qid in sorted(a.keys() & b.keys()):
        delta = a[qid]["support_recall"] - b[qid]["support_recall"]
        if abs(delta) < 1e-12:
            continue
        kind = "rescued" if delta > 0 else "lost"
        if len(output[kind]) >= 3:
            continue
        output[kind].append({
            "id": qid, "question": questions[qid]["question"],
            "gold_titles": questions[qid]["gold_titles"],
            "candidate_recall": a[qid]["support_recall"],
            "baseline_recall": b[qid]["support_recall"],
            "candidate_titles": a[qid]["selected_titles"],
            "baseline_titles": b[qid]["selected_titles"],
            "candidate_expanded_query": a[qid]["expanded_query"],
            "candidate_expansion_provenance": a[qid]["expansion_provenance"],
        })
    return output


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def report_tables(summary: dict[str, Any]) -> tuple[str, str]:
    markdown, markup = [], []
    groups = summary["groups"]
    for dataset in summary["datasets"]:
        for split in ("test", "dev"):
            selected = [(name, group) for name, group in groups.items()
                        if group["dataset"] == dataset and group["split"] == split]
            if not selected:
                continue
            heading = f"{dataset} · {split}"
            headers = ["策略", "K", "D上限", "完成/应完成", "支持召回%", "完整支持%",
                       "严格事实%", "严格完整%", "平均J", "平均D", "总时延p50秒", "总时延p95秒"]
            markdown.extend([f"## {heading}", "", "| " + " | ".join(headers) + " |",
                             "|" + "|".join(["---"] * len(headers)) + "|"])
            markup.append(f"<h2>{html.escape(heading)}</h2><div class='scroll'><table><thead><tr>"
                          + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>")
            for _, group in sorted(selected, key=lambda item: (item[1]["d_budget"], item[1]["k"], item[1]["policy"])):
                values = [group["policy"], str(group["k"]), str(group["d_budget"]),
                          f"{group['n']}/{group['expected_n']}",
                          percent(group["support_recall"]["mean"]),
                          percent(group["complete_support"]["mean"]),
                          percent(group["strict_fact_recall"]["mean"]),
                          percent(group["strict_complete_support"]["mean"]),
                          number(group["j_count"]["mean"]), number(group["d_count"]["mean"]),
                          number(group["total_seconds"]["p50"], 4),
                          number(group["total_seconds"]["p95"], 4)]
                markdown.append("| " + " | ".join(values) + " |")
                markup.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
            markdown.append("")
            markup.append("</tbody></table></div>")
    return "\n".join(markdown), "\n".join(markup)


def write_reports(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    atomic_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    atomic_text(out / "scored_rows.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    markdown_table, html_table = report_tables(summary)
    metadata = summary["metadata"]
    title = "HybridRAG 公共语料检索扩展先导"
    lead = (f"检索结果 {metadata['observed_rows']}/{metadata['expected_rows']} 条。"
            f"完整性检查{'通过' if metadata['complete'] else '未通过'}，"
            f"协议检查{'通过' if metadata['protocol_valid'] else '未通过'}。")
    markdown = [f"# {title}", "", lead, "", "本报告以支持证据检索为终点，分数列为百分比。", "", markdown_table]
    markup = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
              "<meta name='viewport' content='width=device-width,initial-scale=1'>",
              f"<title>{title}</title><style>",
              "body{margin:0;background:#f3f6f8;color:#193042;font:15px/1.8 -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}",
              "main{max-width:1320px;margin:24px auto;padding:28px 34px;background:#fff;border:1px solid #dce3e9;border-radius:12px}",
              "h1{font-size:28px}h2{font-size:21px;margin-top:30px}.lead{padding:15px 20px;background:#edf6f8;border-left:5px solid #17677b}",
              ".scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}th,td{padding:8px 10px;border-bottom:1px solid #dce3e9;text-align:left;white-space:nowrap}th{background:#eff3f6}",
              ".case{padding:12px 18px;background:#f6f8fa;margin:12px 0;border-radius:6px}.note{font-size:13px;color:#586a78}code{overflow-wrap:anywhere}",
              "@media(max-width:700px){main{margin:0;padding:20px 15px;border:0}h1{font-size:23px}}</style></head><body><main>",
              f"<h1>{title}</h1><div class='lead'>{html.escape(lead)}</div>",
              "<p>本报告以支持证据检索为终点，分数列为百分比。表中 D 为累计返回行数上限，J 为实际检索器调用数。</p>", html_table]
    markdown.extend(["## 预设配对比较", "", "差值为候选策略减基线，区间按问题配对重采样，单位为百分点。", "",
                     "| 数据/条件 | 比较 | 配对题数 | Δ召回 [95%CI] | Δ完整支持 [95%CI] |", "|---|---|---:|---|---|"])
    markup.append("<h2>预设配对比较</h2><p>差值为候选策略减基线，区间按问题配对重采样，单位为百分点。</p><div class='scroll'><table><thead><tr><th>数据/条件</th><th>比较</th><th>配对题数</th><th>Δ召回 [95%CI]</th><th>Δ完整支持 [95%CI]</th></tr></thead><tbody>")
    for pair in summary["comparisons"]:
        deltas = []
        for metric in ("support_recall", "complete_support"):
            value = pair[metric]
            deltas.append("—" if value["difference"] is None else
                          f"{100 * value['difference']:+.2f} [{100 * value['ci95'][0]:+.2f}, {100 * value['ci95'][1]:+.2f}]")
        values = [f"{pair['dataset']} K{pair['k']} D{pair['d_budget']}",
                  f"{pair['candidate']} − {pair['baseline']}", str(pair["n_pairs"]), *deltas]
        markdown.append("| " + " | ".join(values) + " |")
        markup.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    markup.append("</tbody></table></div>")
    markdown.extend(["", "## 补回与丢失案例", "", "案例按问题 ID 排序取各方向前三题，用于核查实际查询与返回证据。", ""])
    markup.append("<h2>补回与丢失案例</h2><p>案例按问题 ID 排序取各方向前三题，用于核查实际查询与返回证据。</p>")
    # Repeated K/D settings can surface the same question, so display it once.
    displayed = set()
    for pair in summary["comparisons"]:
        for direction, cases in pair["cases"].items():
            for case in cases:
                key = (pair["dataset"], pair["candidate"], pair["baseline"], direction, case["id"])
                if key in displayed:
                    continue
                displayed.add(key)
                label = "补回" if direction == "rescued" else "丢失"
                heading = f"{label} · {pair['dataset']} · {pair['candidate']} 对 {pair['baseline']} · K{pair['k']} D{pair['d_budget']}"
                detail = (f"支持召回 {percent(case['baseline_recall'])}% → {percent(case['candidate_recall'])}%。"
                          f"标注支持为 {'、'.join(case['gold_titles'])}。")
                query = str(case.get("candidate_expanded_query") or "本条件没有执行扩展查询")
                markdown.extend([f"### {heading}", "", case["question"], "", detail, "", f"扩展查询为 `{query}`。", ""])
                markup.append(f"<div class='case'><strong>{html.escape(heading)}</strong><p>{html.escape(case['question'])}</p><p>{html.escape(detail)}</p><p>扩展查询为 <code>{html.escape(query)}</code></p></div>")
                if len(displayed) >= 12:
                    break
            if len(displayed) >= 12:
                break
        if len(displayed) >= 12:
            break
    markdown.extend(["## 解释范围", "", *[f"- {value}" for value in LIMITATIONS], "", "机器可读统计及逐题结果分别见 summary.json 和 scored_rows.jsonl。", ""])
    markup.append("<h2>解释范围</h2><ul>" + "".join(f"<li>{html.escape(v)}</li>" for v in LIMITATIONS) + "</ul>")
    markup.append("<p class='note'>机器可读统计及逐题结果分别见 summary.json 和 scored_rows.jsonl。运行协议、语料与结果哈希保存在 summary.json 的 metadata 中。</p></main></body></html>")
    atomic_text(out / "report.md", "\n".join(markdown))
    atomic_text(out / "report.html", "\n".join(markup))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    manifest_path = next((args.run / name for name in ("manifest.json", "run_manifest.json")
                          if (args.run / name).exists()), None)
    if manifest_path is None:
        raise FileNotFoundError("run manifest.json or run_manifest.json is required")
    manifest = read_json(manifest_path)
    protocol = manifest.get("protocol", manifest)
    dataset_names = protocol.get("datasets")
    if dataset_names is None:
        dataset_names = sorted(p.name for p in args.data_root.iterdir()
                               if p.is_dir() and (p / "queries.json").exists())
    datasets = {name: load_dataset(args.data_root / name) for name in dataset_names}
    identity = manifest.get("identity", {})
    identity_dataset = identity.get("dataset")
    if identity_dataset in datasets:
        recorded_hashes = {
            "corpus_sha256": datasets[identity_dataset]["corpus_sha256"],
            "questions_sha256": datasets[identity_dataset]["queries_sha256"],
        }
        for name, actual in recorded_hashes.items():
            if identity.get(name) is not None and identity[name] != actual:
                raise ValueError(f"manifest {name} differs from evaluation input")
    expected = expected_groups(protocol, datasets)
    results_path = args.run / "results.jsonl"
    calls_path = args.run / "calls.jsonl"
    for name, path in (("results_sha256", results_path), ("calls_sha256", calls_path)):
        if manifest.get(name) is not None and manifest[name] != sha256(path):
            raise ValueError(f"manifest {name} differs from raw output")
    call_lookup = {}
    if calls_path.exists():
        with calls_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                call = json.loads(line)
                call_id = str(call["call_id"])
                if call_id in call_lookup:
                    raise ValueError(f"duplicate call_id in calls.jsonl at line {line_number}")
                call_lookup[call_id] = call
    rows, grouped, seen, signatures = [], collections.defaultdict(list), set(), set()
    with results_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if manifest.get("signature") and raw.get("signature") != manifest["signature"]:
                raise ValueError(f"execution signature mismatch at line {line_number}")
            dataset_name = str(raw["dataset"])
            if dataset_name not in datasets:
                raise ValueError(f"unknown dataset in results line {line_number}")
            key = group_key(dataset_name, str(raw["split"]), raw)
            query_id = str(raw["id"])
            if key not in expected or query_id not in expected[key]:
                raise ValueError(f"row outside frozen protocol at line {line_number}")
            identity = key + (query_id,)
            if identity in seen:
                raise ValueError(f"duplicate result row at line {line_number}")
            seen.add(identity)
            row = enrich_row(raw, datasets[dataset_name], call_lookup)
            rows.append(row)
            grouped[key].append(row)
            signatures.add(row["signature"])
    groups = {}
    for key, ids in expected.items():
        dataset, split, policy, k, d_budget, j_budget = key
        groups[group_name(key)] = {"dataset": dataset, "split": split, "policy": policy,
                                  "k": k, "d_budget": d_budget, "j_budget": j_budget,
                                  **aggregate(grouped[key], ids)}
    comparisons = []
    pair_specs = protocol.get("primary_comparisons", [
        {"candidate": "bridge_expand", "baseline": "hybrid_rrf"},
        {"candidate": "query_expand", "baseline": "hybrid_rrf"},
        {"candidate": "bridge_expand", "baseline": "query_expand"},
        {"candidate": "hybrid_quota", "baseline": "hybrid_rrf"},
    ])
    for specification in pair_specs:
        candidate, baseline = specification["candidate"], specification["baseline"]
        for key in sorted(expected):
            dataset, split, policy, k, d_budget, j_budget = key
            reference_key = (dataset, split, baseline, k, d_budget, j_budget)
            if split != "test" or policy != candidate or reference_key not in expected:
                continue
            pair = paired_bootstrap(grouped[key], grouped[reference_key], args.bootstrap_samples, args.seed)
            pair.update({"dataset": dataset, "split": split, "candidate": candidate, "baseline": baseline,
                         "k": k, "d_budget": d_budget, "j_budget": j_budget,
                         "protocol_complete": groups[group_name(key)]["complete"]
                             and groups[group_name(reference_key)]["complete"],
                         "cases": paired_cases(grouped[key], grouped[reference_key], datasets[dataset]["questions"])})
            comparisons.append(pair)
    reachability = {}
    for name, data in datasets.items():
        reachability[name] = {}
        for split in ("dev", "test"):
            annotations = [data["annotations"][qid] for qid, question in data["questions"].items()
                           if question["split"] == split]
            reachability[name][split] = {"n": len(annotations)}
            for metric in ("corpus_title_recall", "corpus_complete_support", "corpus_strict_fact_recall",
                           "corpus_strict_complete_support", "corpus_sentence_recall"):
                reachability[name][split][metric] = describe([a[metric] for a in annotations])
    complete = all(group["complete"] for group in groups.values())
    metadata = {
        "created_at": dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "global_context_union_historical_sample_matched_100_200",
        "complete": complete, "protocol_valid": complete and len(signatures) == 1,
        "observed_rows": len(rows), "expected_rows": sum(len(ids) for ids in expected.values()),
        "execution_signatures": sorted(signatures), "validation_errors": 0,
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
        "resampling_unit": "question", "rate_units": "fractions",
        "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "results_path": str(results_path), "results_sha256": sha256(results_path),
        "calls_sha256": sha256(calls_path) if calls_path.exists() else None,
        "unique_call_records": len(call_lookup),
        "referenced_unique_call_records": len({call_id for row in rows for call_id in row["call_ids"]}),
        "latency_samples_shared_across_K": bool(protocol.get("latency_samples_shared_across_K",
                                                            manifest.get("latency_samples_shared_across_K", False))),
        "evaluator_sha256": sha256(Path(__file__)),
        "datasets": {name: {"corpus_sha256": data["corpus_sha256"],
                             "queries_sha256": data["queries_sha256"],
                             "corpus_documents": len(data["documents"]),
                             "query_protocol": data["query_protocol"]}
                     for name, data in datasets.items()},
        "timing_scope": protocol.get("timing_scope", "运行器逐调用与逐配置记录，索引构建及文档编码另报"),
    }
    summary = {"metadata": metadata, "datasets": dataset_names, "groups": groups,
               "corpus_reachability": reachability, "comparisons": comparisons,
               "limitations": LIMITATIONS}
    write_reports(args.out, summary, rows)
    print(json.dumps({"report": str(args.out / "report.html"), "complete": complete,
                      "protocol_valid": metadata["protocol_valid"], "rows": len(rows),
                      "expected_rows": metadata["expected_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
