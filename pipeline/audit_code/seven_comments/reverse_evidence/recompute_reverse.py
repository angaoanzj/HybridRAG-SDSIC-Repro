"""Offline reverse-order audit of saved current-cohort retrieval and reader records.

Uses only Python's standard library. Does not import retrieval/reader runners,
contact a server, train a model, or generate an answer. All writes stay beside
this script, apart from the parent reverse_audit.json receipt.
"""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent / 'reverse_audit.json'
BASE = Path('/path/to/local/home/.codex/visualizations/2026/11/07/01a1b6f1-912d-7e41-9fce-a1cec97f186a')
SOURCE = BASE / 'HybridRAG_原题集_固定索引_修正复跑_20261108'
VIEW = BASE / 'HybridRAG_排除人工词表_20261109'
READER = VIEW / 'reader_18条件_去人工缓存'
PROJECT = Path('/path/to/local/home/Documents/待审核论文/HybridRAG_PaperAgent_SDSIC_新项目_20261110')
PAPER_DATA = PROJECT / 'runs/review_response_06/work/manuscript/data'
PAIRS = [
    ('fewrel_exact', 'score2_u24', 'reverse_u24'),
    ('fewrel_wordnet', 'no_second_branch', 'score_two_reverse'),
]
POLICIES = {p for _, a, b in PAIRS for p in [a, b]}
DATASETS = ['2wiki', 'hotpotqa']
FOUR_K = [10, 20, 30, 50]
PATHS = {
    'ordered_top50': SOURCE / '四项证据核查_20261108/reader_inputs_export/current_retrieval_groups.jsonl.gz',
    'selected_document_texts': SOURCE / '四项证据核查_20261108/reader_inputs_export/selected_documents.jsonl.gz',
    'coverage_per_question': VIEW / 'analysis/coverage_per_question_18.jsonl.gz',
    'reader_per_question': READER / 'analysis/scored_answers/per_question_answers.jsonl.gz',
    'existing_reader_paired': READER / 'analysis/scored_answers/paired_contrasts.json',
    'reader_aggregate_in_manuscript': PAPER_DATA / 'reader_answer_all.csv',
    'coverage_aggregate_in_manuscript': PAPER_DATA / 'reader_coverage_all.csv',
    'admissible_plan': VIEW / 'method_plan_18.json',
    'method_code': SOURCE / 'code/expansion_variants.py',
    'cue_code': SOURCE / 'code/cue_runtime.py',
    'coverage_metric_code': SOURCE / 'evaluator/fresh_audit_score.py',
    'reader_plan': READER / 'analysis_plan.json',
    'reader_source_refresh': READER / 'receipts/source_refresh_final_audit.json',
    **{ds + '_support_labels': SOURCE / f'analysis/scored/{ds}_scoring_labels.json' for ds in DATASETS},
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gzrows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    assert rows
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def norm(value):
    return ' '.join(value.split())

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
                     for name, p in PATHS.items()}
    plan = json.loads(PATHS['admissible_plan'].read_text())
    assert POLICIES <= set(plan['policies'])
    assert 'score2_old24' not in plan['policies']
    groups = {}
    for row in gzrows(PATHS['ordered_top50']):
        if row['policy'] in POLICIES:
            key = row['dataset'], row['policy'], row['id']
            assert key not in groups
            groups[key] = row
    assert len(groups) == 2400
    wanted = {(row['dataset'], doc) for row in groups.values() for doc in row['selected_doc_ids_50']}
    docs = {}
    for row in gzrows(PATHS['selected_document_texts']):
        key = row['dataset'], row['doc_id']
        if key in wanted:
            assert key not in docs
            docs[key] = {'title': row['title'], 'text': norm(row['text'])}
    assert set(docs) == wanted
    labels = {ds: json.loads(PATHS[ds + '_support_labels'].read_text()) for ds in DATASETS}
    thresholds = {}
    for (ds, policy, qid), row in groups.items():
        label = labels[ds][qid]
        gold = set(label['gold_titles'])
        assert gold
        first, strict = {}, {}
        rank = row['selected_doc_ids_50']
        assert len(rank) == 50 and len(set(rank)) == 50
        for pos, did in enumerate(rank, 1):
            doc = docs[ds, did]
            title = doc['title']
            if title not in gold:
                continue
            first.setdefault(title, pos)
            sentences = label['supporting_sentences_by_title'].get(title, [])
            if sentences and all(norm(s) and norm(s) in doc['text'] for s in sentences):
                strict.setdefault(title, pos)
        thresholds[ds, policy, qid] = {
            'FT_rank': max(first.values()) if gold <= first.keys() else 51,
            'strictFT_rank': max(strict.values()) if gold <= strict.keys() else 51,
            'title_first_ranks': first, 'gold_n': len(gold),
        }
    cov = {}
    for row in gzrows(PATHS['coverage_per_question']):
        if row['policy'] in POLICIES:
            key = row['dataset'], row['policy'], row['K'], row['id']
            assert key not in cov
            cov[key] = row
    assert len(cov) == 9600
    for (ds, policy, k, qid), row in cov.items():
        derived = thresholds[ds, policy, qid]
        assert row['selected_ids'] == groups[ds, policy, qid]['selected_doc_ids_50'][:k]
        assert row['F'] == int(derived['FT_rank'] <= k)
        assert row['strictF'] == int(derived['strictFT_rank'] <= k)
        recall = sum(rank <= k for rank in derived['title_first_ranks'].values()) / derived['gold_n']
        assert abs(row['R'] - recall) < 1e-12
    coverage_aggregates = {(r['dataset'], r['policy'], int(r['K'])): r
                           for r in csvrows(PATHS['coverage_aggregate_in_manuscript'])}
    for ds in DATASETS:
        for policy in POLICIES:
            for k in FOUR_K:
                rows = [r for (d, p, kk, _), r in cov.items() if (d, p, kk) == (ds, policy, k)]
                assert len(rows) == 300
                aggregate = coverage_aggregates[ds, policy, k]
                assert sum(r['F'] for r in rows) == int(aggregate['F_count'])
                assert sum(r['strictF'] for r in rows) == int(aggregate['strictF_count'])

    curves, minimal, summary, invariants = [], [], [], []
    for ds in DATASETS:
        for cue, reference, variant in PAIRS:
            ids = sorted(i for d, p, i in groups if (d, p) == (ds, reference))
            assert len(ids) == 300
            assert ids == sorted(i for d, p, i in groups if (d, p) == (ds, variant))
            title_counts = Counter()
            changed = 0
            for qid in ids:
                a, b = cov[ds, reference, 50, qid], cov[ds, variant, 50, qid]
                title_counts[len(a['selected_titles'])] += 1
                assert b['selected_titles'] == (list(reversed(a['selected_titles']))
                                                if len(a['selected_titles']) == 2 else a['selected_titles'])
                assert (a['j_count'], a['d_count']) == (b['j_count'], b['d_count'])
                changed += a['expanded_query'] != b['expanded_query']
                t0, t1 = thresholds[ds, reference, qid], thresholds[ds, variant, qid]
                minimal.append({'dataset': ds, 'cue': cue, 'reference': reference, 'variant': variant,
                                'id': qid, 'reference_FT_first_complete_K': t0['FT_rank'],
                                'variant_FT_first_complete_K': t1['FT_rank'],
                                'reference_strictFT_first_complete_K': t0['strictFT_rank'],
                                'variant_strictFT_first_complete_K': t1['strictFT_rank'],
                                'gold_title_count': t0['gold_n'], 'selected_title_count': len(a['selected_titles']),
                                'expanded_query_changed': a['expanded_query'] != b['expanded_query']})
            invariants.append({'dataset': ds, 'cue': cue, 'N': 300,
                               'reference_selected_title_counts': dict(title_counts),
                               'expanded_queries_changed': changed,
                               'same_selected_targets_and_exact_two_title_reversal': True,
                               'per_question_logical_calls_and_return_records_match': True})
            local = []
            for k in range(1, 51):
                row = {'dataset': ds, 'cue': cue, 'reference': reference, 'variant': variant,
                       'K': k, 'N': 300}
                for metric in ['FT', 'strictFT']:
                    pairs = [(int(thresholds[ds, reference, qid][metric + '_rank'] <= k),
                              int(thresholds[ds, variant, qid][metric + '_rank'] <= k)) for qid in ids]
                    before, after = sum(a for a, _ in pairs), sum(b for _, b in pairs)
                    wins, losses = sum(b > a for a, b in pairs), sum(b < a for a, b in pairs)
                    assert after - before == wins - losses
                    row.update({metric + '_reference_n': before, metric + '_variant_n': after,
                                metric + '_wins': wins, metric + '_losses': losses,
                                metric + '_net_n': after - before, metric + '_delta_pp': (after - before) / 3})
                curves.append(row)
                local.append(row)
            deltas = [r['FT_delta_pp'] for r in local]
            summary.append({'dataset': ds, 'cue': cue,
                            'FT_delta_pp_min': min(deltas), 'FT_delta_pp_max': max(deltas),
                            'FT_min_K': [r['K'] for r in local if r['FT_delta_pp'] == min(deltas)],
                            'FT_max_K': [r['K'] for r in local if r['FT_delta_pp'] == max(deltas)],
                            'any_FT_change_questions': sum(thresholds[ds, reference, i]['FT_rank'] !=
                                                           thresholds[ds, variant, i]['FT_rank'] for i in ids),
                            'FT_and_strictFT_curves_identical': all(all(r['FT_' + suffix] == r['strictFT_' + suffix]
                                for suffix in ['reference_n', 'variant_n', 'wins', 'losses']) for r in local)})

    answers = {}
    for row in gzrows(PATHS['reader_per_question']):
        if row['policy'] in POLICIES:
            key = row['dataset'], row['policy'], row['K'], row['id']
            assert key not in answers
            answers[key] = row
    assert len(answers) == 9600
    aggregate = {(r['dataset'], r['policy'], int(r['K'])): r
                 for r in csvrows(PATHS['reader_aggregate_in_manuscript'])}
    answer_pairs, answer_minimal = [], []
    for ds in DATASETS:
        for cue, reference, variant in PAIRS:
            for k in FOUR_K:
                ids = sorted(i for d, p, kk, i in answers if (d, p, kk) == (ds, reference, k))
                assert len(ids) == 300 and ids == sorted(i for d, p, kk, i in answers
                                                        if (d, p, kk) == (ds, variant, k))
                pairs = [(answers[ds, reference, k, i], answers[ds, variant, k, i]) for i in ids]
                row = {'dataset': ds, 'cue': cue, 'reference': reference, 'variant': variant, 'K': k, 'N': 300}
                for side, policy, index in [('reference', reference, 0), ('variant', variant, 1)]:
                    em = sum(pair[index]['EM'] for pair in pairs)
                    f1 = sum(pair[index]['F1'] for pair in pairs) / 3
                    a = aggregate[ds, policy, k]
                    assert em == int(a['EM_count']) and abs(f1 - float(a['F1_points'])) < 1e-9
                    row.update({side + '_EM_n': int(em), side + '_EM_pct': em / 3,
                                side + '_F1_points': f1})
                for metric in ['EM', 'F1']:
                    wins = sum(b[metric] - a[metric] > 1e-12 for a, b in pairs)
                    losses = sum(a[metric] - b[metric] > 1e-12 for a, b in pairs)
                    row.update({metric + '_wins': wins, metric + '_losses': losses,
                                metric + '_ties': 300 - wins - losses,
                                metric + '_delta_points': sum(b[metric] - a[metric] for a, b in pairs) / 3})
                row['in_original_48_main_contrasts'] = False
                answer_pairs.append(row)
                for qid, (a, b) in zip(ids, pairs):
                    answer_minimal.append({'dataset': ds, 'cue': cue, 'reference': reference,
                        'variant': variant, 'K': k, 'id': qid, 'reference_EM': a['EM'],
                        'variant_EM': b['EM'], 'reference_F1': a['F1'], 'variant_F1': b['F1'],
                        'reference_input_key': a['input_key'], 'variant_input_key': b['input_key']})
    existing_pairs = json.loads(PATHS['existing_reader_paired'].read_text())
    assert len(existing_pairs) == 48
    assert all(r['left'] not in ['reverse_u24', 'score_two_reverse'] and
               r['right'] not in ['reverse_u24', 'score_two_reverse'] for r in existing_pairs)

    write_csv('coverage_curves_K1_50.csv', curves)
    write_csv('coverage_pair_thresholds.csv', minimal)
    write_csv('reader_reverse_fourK.csv', answer_pairs)
    write_csv('reader_pair_values.csv', answer_minimal)
    report = {
        'schema_version': 1, 'date': '2026-11-10', 'status': 'pass',
        'analysis_type': 'post_hoc_descriptive_retruncation_and_paired_reanalysis_of_saved_current_records',
        'inputs': source_hashes,
        'scope': {'datasets': DATASETS, 'N_each': 300, 'pairs': PAIRS,
                  'coverage_integer_K': [1, 50], 'reader_K': FOUR_K,
                  'historically_used_questions': True, 'new_retrieval': False,
                  'new_generation': False, 'new_training': False,
                  'current_K_above_50_supported': False, 'pool_completeness_computed': False},
        'definitions': {
            'FT': 'All annotated support titles present in a retained prefix, based on saved full document titles.',
            'strictFT': 'For every gold title, a retained version contains all its annotated support sentences after whitespace normalization.',
            'threshold_51': 'Not complete within the saved Top50; does not mean absent from the full candidate pool.',
            'wins_losses': 'Variant succeeds/reference fails, and reference succeeds/variant fails, on the same question.',
            'fewrel_exact': 'score2_u24 versus reverse_u24; exact matching to the externally induced FewRel cue inventory.',
            'fewrel_wordnet': 'no_second_branch versus score_two_reverse; FewRel cues with WordNet morphology and synonym matching.',
            'order_control': 'Reverse exactly two already-selected title mentions before unchanged residual keywords; zero/single-title fallback unchanged.',
        },
        'checks': {'current_top50_groups': len(groups), 'saved_fourK_coverage_rows_checked': len(cov),
                   'checks_per_coverage_row': ['ordered_prefix_ids', 'FT', 'strictFT', 'support_recall'],
                   'coverage_curves_rows': len(curves), 'coverage_pair_threshold_rows': len(minimal),
                   'reader_rows_checked': len(answers), 'reader_summary_rows': len(answer_pairs),
                   'reader_pair_value_rows': len(answer_minimal),
                   'coverage_and_reader_match_current_manuscript_aggregates': True,
                   'both_reverse_variants_in_admissible_18': True,
                   'excluded_manual_condition': 'score2_old24'},
        'invariants': invariants, 'coverage_summaries': summary,
        'reader_inference_boundary': 'Neither reverse contrast belongs to the retained 48 main EM contrasts. These paired counts are descriptive additions; no new significance tests, bootstrap intervals or multiplicity adjustment were computed.',
        'historical_branch_boundary': 'The older manual-cue reviewer_resolution_20261107 K1-50 reverse audit exists, but its 300 IDs per corpus have zero intersection with this current FewRel cohort; it is not reused as current evidence.',
        'limits': ['Coverage is reconstructed only from the saved ordered Top50 and full selected-document text.',
                   'Uncovered at K50 is not a pool miss; full candidate-pool completeness is not inferred.',
                   'All 50 cutoffs reuse each saved ranking and are correlated descriptive truncations.',
                   'Reader outcomes are available only at K10, K20, K30 and K50.',
                   'No claim of score-order superiority, order invariance, or independent confirmation follows.'],
        'artifacts': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                      for p in sorted(OUT.iterdir()) if p.is_file()},
    }
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'coverage_summaries': summary,
                      'checks': report['checks'], 'receipt': str(AUDIT)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
