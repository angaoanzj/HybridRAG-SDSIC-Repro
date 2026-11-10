#!/usr/bin/env python3
"""Recompute saved-result evidence using Python 3.9+ standard library only.

This does not perform retrieval, training, title selection or answer generation.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def id_hash(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


class Audit:
    def __init__(self):
        self.counts = Counter()

    def check(self, condition, name):
        self.counts[name] += 1
        if not condition:
            raise ValueError("Evidence check failed: " + name)


def exact_mcnemar(wins, losses):
    """Two-sided exact conditional test of discordant pairs under p=1/2."""
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0


def holm(values):
    out = [0.0] * len(values)
    previous = 0.0
    for rank, i in enumerate(sorted(range(len(values)), key=lambda i: (values[i], i))):
        previous = max(previous, min(1.0, (len(values) - rank) * values[i]))
        out[i] = previous
    return out


def statistics(data, audit):
    j = read_json(data / "statistics_inputs.json")
    rows = []
    for domain in ("coverage", "reader_EM"):
        for row in j[domain]:
            audit.check(row["n"] == row["wins"] + row["losses"] + row["ties"] == 300, "statistics_denominator")
            p = exact_mcnemar(row["wins"], row["losses"])
            audit.check(math.isclose(p, row["raw_p_saved"], rel_tol=1e-13, abs_tol=1e-15), "exact_mcnemar")
            item = dict(row, domain=domain, raw_p_recomputed=p)
            if row["adjustment"] == "Bonferroni":
                adjusted = min(1.0, row["family_size"] * p)
                audit.check(adjusted == row["adjusted_p_saved"], "bonferroni")
                item["adjusted_p_recomputed"] = adjusted
            rows.append(item)
    for family in ("primary", "lexical_extension", "family"):
        group = [r for r in rows if r["domain"] == "coverage" and r["family"] == family]
        audit.check(len(group) == group[0]["family_size"], "holm_family_size")
        for row, adjusted in zip(group, holm([r["raw_p_recomputed"] for r in group])):
            audit.check(adjusted == row["adjusted_p_saved"], "holm")
            row["adjusted_p_recomputed"] = adjusted
    em = [r for r in rows if r["domain"] == "reader_EM"]
    audit.check(len(em) == 48 and len(rows) == 60, "statistics_test_counts")
    audit.check(len(j["original_reader_comparisons"]) * len(j["cutoffs"]) * len(j["datasets"]) == 56, "original_reader_family")
    audit.check(len(j["retained_reader_comparisons"]) * len(j["cutoffs"]) * len(j["datasets"]) == 48, "retained_reader_family")
    audit.check(set(map(tuple, j["original_reader_comparisons"])) - set(map(tuple, j["retained_reader_comparisons"])) == {("cue_source", "external_vs_manual", "score2_u24", "score2_old24")}, "excluded_reader_contrast")
    audit.check(all(r["adjusted_p_recomputed"] == 1 for r in em), "all_retained_reader_p_one")
    audit.check(not any("reverse" in r["left"] or "reverse" in r["right"] for r in em), "reversal_outside_family")
    return {"test": "two-sided exact conditional McNemar", "coverage_tests": 12, "retained_reader_tests": 48, "original_reader_adjustment_size": 56, "all_retained_reader_adjusted_p_one": True, "results": rows}


def idf_feedback(data, audit):
    by = {}
    with gzip.open(data / "idf_f2_records.jsonl.gz", "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            key = (row["dataset"], row["id"], row["method"])
            audit.check(key not in by, "idf_unique_record")
            by[key] = row
    expected = read_json(data / "idf_expected.json")
    summaries = {}
    for row in expected:
        ds, method = row["dataset"], row["method"]
        records = [r for (dataset, _, arm), r in by.items() if dataset == ds and arm == method]
        valid = [r for r in records if r["strict_valid"]]
        outcomes = {r["id"]: int(r["first_complete_rank"] is not None and r["first_complete_rank"] <= 2) for r in valid}
        audit.check(len(records) == int(row["questions"]), "idf_question_count")
        audit.check(len(outcomes) == int(row["strict_valid_n"]), "idf_valid_count")
        audit.check(sum(outcomes.values()) == int(row["support_complete_n"]), "idf_curve_count")
        summaries[ds, method] = outcomes
    results = {}
    for ds in ("2wiki", "hotpotqa", "musique"):
        arms = {label: summaries[ds, method] for label, method in [("IDF8", "idf_m3_head180_w8"), ("S2", "no_second_branch"), ("RRF", "hybrid_rrf")]}
        audit.check(set(arms["IDF8"]) == set(arms["S2"]) == set(arms["RRF"]), "idf_same_valid_ids")
        a, b = arms["IDF8"], arms["S2"]
        wins, losses = sum(a[q] > b[q] for q in a), sum(a[q] < b[q] for q in a)
        results[ds] = {"N": len(a), "K": 2, "counts": {label: sum(v.values()) for label, v in arms.items()}, "IDF8_minus_S2": {"wins": wins, "losses": losses, "ties": len(a) - wins - losses, "delta_pp": 100 * (wins - losses) / len(a)}, "IDF8_minus_RRF_pp": 100 * (sum(a.values()) - sum(arms["RRF"].values())) / len(a)}
    return {"scope": "Strict support from projected per-question first-complete ranks; shared title-selection state does not imply identical supplementary-query triggers or realized calls.", "results": results}


def sampling(data, protocol, official, audit):
    j = read_json(data / "musique_sampling.json")
    roles = j["role_ids"]
    cal, eff = roles["calibration_and_timing"], roles["effectiveness"]
    original = protocol["question_ids"]
    removed = {r["removed_id"] for r in j["duplicates"]}
    audit.check(len(original) == 2417 and len(set(original)) == 2417, "original_musique_ids")
    audit.check(len(removed) == 6 and removed <= set(original), "duplicate_id_count")
    audit.check(len(cal) == 100 and len(eff) == 2311 and not set(cal) & set(eff), "musique_split_counts")
    audit.check(set(cal + eff) == set(original) - removed, "musique_partition")
    audit.check(all(r["retained_first_id"] in set(cal + eff) for r in j["duplicates"]), "duplicate_representatives")
    audit.check(id_hash(cal) == j["calibration_timing_ids_sha256"], "calibration_order_hash")
    audit.check(id_hash(eff) == j["effectiveness_ids_sha256"], "effectiveness_order_hash")
    verified_from_questions = False
    if official:
        if zipfile.is_zipfile(official):
            with zipfile.ZipFile(official) as z:
                rows = [json.loads(line) for line in z.read("data/musique_ans_v1.0_dev.jsonl").decode().splitlines() if line.strip()]
        else:
            rows = [json.loads(line) for line in official.read_text(encoding="utf-8").splitlines() if line.strip()]
        seen, duplicates = {}, []
        for row in rows:
            key = " ".join(unicodedata.normalize("NFKC", row["question"]).casefold().split())
            if key in seen:
                duplicates.append({"removed_id": row["id"], "retained_first_id": seen[key]})
            else:
                seen[key] = row["id"]
        audit.check([r["id"] for r in rows] == original, "official_dev_order")
        audit.check(duplicates == j["duplicates"], "official_normalized_duplicates")
        audit.check(set(seen.values()) == set(cal + eff), "official_partition")
        verified_from_questions = True
    return {"original": 2417, "duplicate_rows_removed": 6, "unique_representatives": 2411, "calibration_timing": 100, "effectiveness": 2311, "normalization": j["normalization"], "normalization_rerun_from_official_question_text": verified_from_questions, "default_scope": "Bundled ID partition and hashes; original question text is not bundled."}


def multiround(data, audit):
    protocol = read_json(data / "multiround_protocol.json")
    expected = read_json(data / "multiround_expected.json")
    with gzip.open(data / "multiround_quality.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    ids, arms, ks = protocol["question_ids"], protocol["arms"], protocol["K"]
    n = len(ids)
    by = {(r["id"], r["arm"]): r for r in rows}
    audit.check(n == 2417 and len(arms) == 9 and len(by) == len(rows) == 21753, "multiround_dimensions")
    audit.check(set(by) == {(qid, arm) for qid in ids for arm in arms}, "multiround_cartesian_product")
    audit.check(ks == [10, 20, 30, 50] and protocol["D_cap"] == 400, "multiround_grid_budget")
    aggregates, table = {}, []
    for arm in arms:
        selected = [by[qid, arm] for qid in ids]
        audit.check([r["id"] for r in rows if r["arm"] == arm] == ids, "multiround_arm_order")
        result = {"n": n, "supplementary_distribution": dict(Counter(str(r["supplementary_queries"]) for r in selected)), "mean_physical_route_calls": sum(r["physical_route_calls"] for r in selected) / n, "mean_returned_records": sum(r["returned_records"] for r in selected) / n, "at_k": {str(k): {m: sum(r["at_k"][str(k)][m] for r in selected) / n for m in ("FP", "FT", "paragraph_recall")} for k in ks}}
        audit.check(same(result, expected["arms"][arm]), "multiround_aggregate")
        aggregates[arm] = result
        for k in ks:
            table.append({"arm": arm, "K": k, "N": n, "FP_count": sum(r["at_k"][str(k)]["FP"] for r in selected), "FT_count": sum(r["at_k"][str(k)]["FT"] for r in selected), "FP_percent": 100 * result["at_k"][str(k)]["FP"], "FT_percent": 100 * result["at_k"][str(k)]["FT"], "paragraph_recall_percent": 100 * result["at_k"][str(k)]["paragraph_recall"], "mean_route_calls": result["mean_physical_route_calls"], "mean_returned_records": result["mean_returned_records"]})
    paired = {}
    for name, explanation in protocol["comparisons"].items():
        left, right = name.split("-minus-")
        result = {"interpretation": explanation, "same_realized_D_n": sum(by[qid, left]["returned_records"] == by[qid, right]["returned_records"] for qid in ids), "at_k": {}}
        for k in ks:
            delta = [by[qid, left]["at_k"][str(k)]["FP"] - by[qid, right]["at_k"][str(k)]["FP"] for qid in ids]
            result["at_k"][str(k)] = {"wins": sum(d > 0 for d in delta), "losses": sum(d < 0 for d in delta), "delta_pp": 100 * sum(delta) / n}
        audit.check(same(result, expected["paired"][name]), "multiround_paired")
        paired[name] = result
    return {"quality_rows": len(rows), "arms": aggregates, "paired": paired, "table": table, "scope": "FP/FT/recall reaggregated from saved quality. Raw rankings and original label semantics are not reconstructed by this default mode."}, protocol, by


def raw_replay(path, data, protocol, by, audit):
    complete = read_json(data / "original_completion.json")
    audit.check(digest(path) == complete["raw_sha256"], "external_raw_hash")
    counts, seen, mechanisms = Counter(), set(), {}
    ks = protocol["K"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for raw in map(json.loads, stream):
            key = (raw["id"], raw["arm"])
            audit.check(key in by and key not in seen, "raw_unique_quality_pair")
            seen.add(key)
            q, calls, rounds = by[key], raw["calls"], raw["rounds"]
            counts["rows"] += 1
            counts["physical_calls"] += len(calls)
            counts["rounds"] += len(rounds)
            audit.check(len(calls) == raw["physical_route_calls"] == q["physical_route_calls"] == 2 * len(rounds), "raw_physical_calls")
            audit.check(sum(len(c["returned"]) for c in calls) == raw["returned_records"] == q["returned_records"], "raw_returned_records")
            audit.check(sum(c["depth"] for c in calls) == raw["requested_records"] <= 400, "raw_budget_cap")
            gold, previous_pool = set(q["gold_ranks"]), set()
            for ri, rd in enumerate(rounds):
                prefix = [c for c in calls if c["round"] <= ri]
                instances = len({c["query_id"] for c in prefix})
                votes = defaultdict(float)
                for call in prefix:
                    weight = (protocol["alpha"] if call["route"] == "dense" else 1 - protocol["alpha"]) / instances
                    route_seen = set()
                    for document in call["returned"]:
                        docid = document["doc_id"]
                        if docid not in route_seen:
                            votes[docid] += weight / (60 + document["rank"])
                            route_seen.add(docid)
                ranking = sorted(votes, key=lambda docid: (-votes[docid], docid))
                audit.check(ranking == rd["ranking_after_round"], "raw_replayed_ranking")
                positions = {docid: i for i, docid in enumerate(ranking, 1)}
                metric = q["metrics_after_round"][ri]
                audit.check({docid: positions.get(docid) for docid in gold} == metric["gold_ranks"], "raw_gold_ranks")
                for k in ks:
                    hits = len(set(ranking[:k]) & gold)
                    audit.check(int(hits == len(gold)) == metric["at_k"][str(k)]["FP"] and math.isclose(hits / len(gold), metric["at_k"][str(k)]["paragraph_recall"]), "raw_FP_recall")
                pool = set(ranking)
                if ri >= 2:
                    label = raw["arm"] + ":q" + str(ri)
                    item = mechanisms.setdefault(label, {"executed": 0, "mode": rd["mode"], "new_candidate_total": 0, "new_gold_total": 0, "at_k": {str(k): {"wins": 0, "losses": 0, "win_new_gold_acquisition": 0, "win_ranking_promotion": 0} for k in ks}})
                    item["executed"] += 1
                    item["new_candidate_total"] += len(pool - previous_pool)
                    item["new_gold_total"] += len((pool - previous_pool) & gold)
                    for k in ks:
                        delta = metric["at_k"][str(k)]["FP"] - q["metrics_after_round"][ri - 1]["at_k"][str(k)]["FP"]
                        value = item["at_k"][str(k)]
                        if delta > 0:
                            value["wins"] += 1
                            value["win_new_gold_acquisition" if not gold <= previous_pool else "win_ranking_promotion"] += 1
                        elif delta < 0:
                            value["losses"] += 1
                previous_pool = pool
            audit.check(ranking == raw["ranking"], "raw_final_ranking")
            audit.check(q["gold_ranks"] == q["metrics_after_round"][-1]["gold_ranks"], "raw_final_gold_ranks")
    audit.check(seen == set(by), "raw_all_pairs_present")
    prior = read_json(data / "prior_raw_check.json")
    audit.check(dict(counts) == prior["raw_counts"], "raw_counts_match_prior")
    audit.check(same(mechanisms, prior["later_query_mechanisms"]), "raw_mechanisms_match_prior")
    return {"raw_counts": dict(counts), "later_query_mechanisms": mechanisms, "scope": "Saved route returns replayed; no retrieval, selector rerun, answer generation, timing rerun, or original-title-label FT reconstruction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional directory for newly derived reports; inputs remain unchanged.")
    parser.add_argument("--raw-returns", type=Path, help="Optional original raw_rankings_and_calls.jsonl.gz; not bundled.")
    parser.add_argument("--musique-data", type=Path, help="Optional official musique_data_v1.0.zip or MuSiQue-Ans dev JSONL; not bundled.")
    args = parser.parse_args()
    audit = Audit()
    for relative, expected in read_json(ROOT / "MANIFEST.json").items():
        audit.check(digest(ROOT / relative) == expected, "manifest_sha256")
    data = ROOT / "data"
    report = {"status": "PASS", "scope": "Saved-result verification only; no end-to-end experimental reproduction."}
    report["idf_feedback"] = idf_feedback(data, audit)
    report["statistics"] = statistics(data, audit)
    multi, protocol, by = multiround(data, audit)
    report["multiround"] = multi
    report["sampling"] = sampling(data, protocol, args.musique_data, audit)
    report["raw_replay"] = raw_replay(args.raw_returns, data, protocol, by, audit) if args.raw_returns else {"executed": False, "reason": "External raw-return archive not requested; not included in this bundle."}
    report["checks"] = dict(audit.counts)
    report["checks_total"] = sum(audit.counts.values())
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (args.output / "multiround_table.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=multi["table"][0])
            writer.writeheader()
            writer.writerows(multi["table"])
        with (args.output / "idf_feedback.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset", "N", "K", "IDF8_count", "S2_count", "RRF_count", "IDF8_minus_S2_pp", "IDF8_minus_RRF_pp"])
            for ds, row in report["idf_feedback"]["results"].items():
                writer.writerow([ds, row["N"], row["K"], row["counts"]["IDF8"], row["counts"]["S2"], row["counts"]["RRF"], row["IDF8_minus_S2"]["delta_pp"], row["IDF8_minus_RRF_pp"]])
    print(json.dumps({"status": "PASS", "checks": report["checks_total"], "multiround_rows": multi["quality_rows"], "multiround_arms": len(multi["arms"]), "paired_comparisons": len(multi["paired"]), "statistics_tests": 60, "raw_replay_executed": bool(args.raw_returns), "official_normalization_rerun": bool(args.musique_data)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
