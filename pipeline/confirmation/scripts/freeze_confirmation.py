"""Select before evaluation; inputs contain source metadata, never model outcomes."""
from pathlib import Path
import collections, datetime, gzip, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / 'HybridRAG_新评估题源核查_20261109'
ORIGINAL = ROOT.parent / 'HybridRAG_固定池干预与查询替换_20261109/rerank/recovered_ac/inputs/protocol.json'
SEED = 2026111001
TARGET = 1000

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rows(p):
    with gzip.open(p, 'rt') as f:
        return [json.loads(s) for s in f]

def write(name, obj):
    p = ROOT / 'frozen' / name
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if p.exists():
        assert p.read_text() == text, ('refuse changing frozen output', p)
    else:
        p.write_text(text)

def key(ds, i):
    return hashlib.sha256(f'{SEED}|{ds}|{i}'.encode()).hexdigest()

def main():
    assert not (ROOT / 'frozen/protocol.json').exists(), 'Already frozen; do not resample'
    old = json.loads(ORIGINAL.read_text())
    sources = {'2wiki': 'source_audit/r2/2wiki/version_stable_fixed_corpus_candidates.jsonl.gz',
               'hotpotqa': 'results/hotpotqa_hard_fixed_corpus_candidate_pool.jsonl.gz',
               'musique': 'results/musique_official_dev_old_train_candidate_groups.jsonl.gz'}
    receipt = {}
    for ds in ['2wiki', 'hotpotqa']:
        pool = rows(AUDIT / sources[ds])
        strata = collections.defaultdict(list)
        for r in pool:
            strata[f"{r['support_count']}|{r['type']}"] .append(r)
        allocations = {s: TARGET * len(v) // len(pool) for s, v in strata.items()}
        remainder_order = sorted(strata, key=lambda s: (-(TARGET * len(strata[s]) % len(pool)), s))
        for s in remainder_order[:TARGET - sum(allocations.values())]:
            allocations[s] += 1
        selected = []
        for s in sorted(strata):
            selected += sorted(strata[s], key=lambda r: (key(ds, r['original_id']), r['original_id']))[:allocations[s]]
        selected.sort(key=lambda r: (key(ds, r['original_id']), r['original_id']))
        assert len(selected) == len({r['original_id'] for r in selected}) == TARGET
        selected_ids = {r['original_id'] for r in selected}
        qs = {}
        for r in rows(AUDIT / f'source_audit/r2/{ds}/candidate_manifest.jsonl.gz'):
            if r['original_id'] not in selected_ids:
                continue
            assert r['support_mapping']['all_support_mappable'] and not r['annotation_errors']
            assert not r['historical_original_id_match'] and not r['historical_question_hash_matches']
            qs[r['original_id']] = {'id': r['original_id'], 'question': r['question'],
                'gold_titles': r['support_titles'], 'supporting_sentences_by_title': r['required_support_sentences'],
                'original_ids': [r['original_id']], 'origin': 'official_train_unused_screened',
                'split': 'confirmation', 'type': r['type'], 'level': r['level'],
                'support_count': r['support_count'], 'group_id': r['candidate_group_id']}
        assert set(qs) == selected_ids
        write(f'{ds}_selection.json', selected)
        write(f'{ds}_questions.json', [qs[r['original_id']] for r in selected])
        receipt[ds] = {'candidate_n': len(pool), 'confirmation_n': TARGET,
                       'stratum_population': {s: len(v) for s, v in strata.items()}, 'stratum_selected': allocations,
                       'unique_groups': len({r['candidate_group_id'] for r in selected}),
                       'scope': 'Source-audited new questions, restricted to fixed-corpus support reachability; not official test'}
    groups = sorted(rows(AUDIT / sources['musique']), key=lambda g: (key('musique', g['group_id']), g['group_id']))
    # Exact-size whole-group selection: seeded group priority, include when the
    # residual target is achievable from the suffix. No result-dependent retries.
    suffix = [0] * (len(groups) + 1)
    suffix[-1] = 1
    mask = (1 << (TARGET + 1)) - 1
    for j in range(len(groups) - 1, -1, -1):
        n = len(groups[j]['train_canonical_ids'])
        suffix[j] = (suffix[j+1] | (suffix[j+1] << n)) & mask
    assert (suffix[0] >> TARGET) & 1
    chosen, remaining = [], TARGET
    for j, g in enumerate(groups):
        n = len(g['train_canonical_ids'])
        if n <= remaining and (suffix[j+1] >> (remaining - n)) & 1:
            chosen.append(g)
            remaining -= n
    assert remaining == 0
    confirm = [i for g in chosen for i in sorted(g['train_canonical_ids'])]
    assert len(confirm) == len(set(confirm)) == TARGET
    feasibility = json.loads((AUDIT / 'musique/fallback_group_feasibility.json').read_text())
    buffer = feasibility['buffer_old_train_ids']
    oldtrain = old['datasets']['musique']['ids']['train']
    dev = old['datasets']['musique']['ids']['validation']
    excluded = set(confirm) | set(buffer)
    train = [i for i in oldtrain if i not in excluded]
    assert excluded <= set(oldtrain) and len(excluded) == 1003 and len(train) == 20990
    assert not set(train) & set(confirm) and not set(dev) & excluded
    all_remaining = [json.loads(l) for l in (AUDIT / 'musique/fallback_train_components.jsonl').read_text().splitlines()]
    seen = set()
    for g in all_remaining:
        ids = set(g['train_canonical_ids'])
        assert not g['dev_canonical_ids'] and not ids & set(buffer)
        assert not (ids & set(train) and ids & set(confirm))
        assert not seen & ids
        seen |= ids
    assert seen == set(train) | set(confirm)
    write('musique_selected_groups.json', chosen)
    write('musique_ids.json', {'train': train, 'development': dev, 'confirmation': confirm, 'buffer': buffer})
    receipt['musique'] = {'candidate_groups': len(groups), 'candidate_n': sum(len(g['train_canonical_ids']) for g in groups),
        'confirmation_n': len(confirm), 'confirmation_groups': len(chosen), 'train_n': len(train),
        'development_n': len(dev), 'buffer_n': len(buffer), 'group_disjoint': True,
        'scope': 'Retrained, group-isolated internal confirmation; these questions had historical training exposure'}
    write('selection_receipt.json', receipt)
    protocol = {'version': 'CONFIRM-20261110-v1', 'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'seed': SEED, 'target_per_dataset': TARGET,
      'selection': {'2wiki_hotpotqa': 'largest-remainder proportional allocation by support-count and type; SHA256(seed|dataset|id) priority',
                    'musique': 'whole conservative-D groups, SHA256(seed|dataset|group) priority and suffix feasibility to exact 1000; official-dev-source only'},
      'original_protocol_sha256': sha(ORIGINAL),
      'source_hashes': {k: {'path': str(AUDIT / v), 'sha256': sha(AUDIT / v)} for k, v in sources.items()},
      'datasets': receipt,
      'conditions': {'S2': 'no_second_branch', 'S2_rank': 'same S2 pool and frozen ranker', 'A': 'mix_ranked16', 'C': 'same A pool and same frozen ranker'},
      'primary_K': 5, 'secondary_K': [2, 10, 20, 30, 40, 50], 'saved_cutoffs': list(range(1, 51)),
      'retrieval': {'route_depth': 100, 'q0_routes': ['bm25','dense'], 'q1_routes': ['bm25','dense'],
                    'max_returned_records': 400, 'alpha': {'2wiki': .3, 'hotpotqa': .7, 'musique': .5},
                    'backend': 'original P1.1 GPU supervision FastBulk; fixed corpus/index/BGE, no fine tuning'},
      'ranker': {'2wiki_hotpotqa': 'current frozen document_ranker and weight=.5, unchanged',
                 'musique': {'fresh_fit': True, 'warm_start': False, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 40,
                    'l2_regularization': 10, 'early_stopping': False, 'random_state': 2026110711,
                    'routes': ['no_second_branch','evidence_feedback','mix_ranked16'],
                    'gamma_grid': [0., .25, .5], 'gamma_selection': 'old 300 development, mean A-pool completeness over K5/10/20/30/40/50, first maximum',
                    'training_rows': 'exclude all 1000 confirmation IDs plus 3 buffered IDs; prove per-question row mapping from original sampler and original shard ID order; regenerate if proof fails'}},
      'metrics': ['strict pool completeness P/N', 'strict complete@K F/N', 'conditional retention F/P',
                  'rescue W/(P-F_before)', 'harm L/F_before', 'net (W-L)/P'],
      'strata': ['q1 executed/skipped for each pool', 'support unit count', 'original first complete rank bins 6-10/11-20/21-50/>50'],
      'uncertainty': {'method': 'paired percentile bootstrap, 2000 draws, seed 2026111002',
                      'unit': 'question within prespecified strata for 2wiki/hotpotqa; whole selected group for musique',
                      'scope': 'descriptive 95% intervals; no multiplicity-adjusted significance claims'},
      'prohibitions': ['no confirmation-driven resampling/tuning', 'no replacing failed retrieval questions',
                       'no corpus expansion using gold evidence', 'no warm starting MuSiQue from exposed weights'],
      'frozen_files': {p.name: sha(p) for p in sorted((ROOT / 'frozen').glob('*.json'))}}
    write('protocol.json', protocol)
    (ROOT / 'frozen/protocol.sha256').write_text(sha(ROOT / 'frozen/protocol.json') + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
