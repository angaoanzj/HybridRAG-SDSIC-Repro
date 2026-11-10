"""Read-only reanalysis of completed corpus-retrieval trials; writes only beside this script."""
from pathlib import Path
import collections, gzip, hashlib, json, math, random, sys

OUT = Path(__file__).resolve().parent
PRIOR = Path('/path/to/local/home/Downloads/返修版本')
STUDY = PRIOR / 'studies/reviewer_validation_20261106'
EVIDENCE = PRIOR / 'HybridRAG_去重敏感性核验_20261106'
RAW = EVIDENCE / 'raw'
sys.path.insert(0, str(STUDY / 'evaluator'))
from independent_audit import IndependentRules, audit_group, normalized

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def load(p): return json.loads(p.read_text())
def rank_thresholds(ranking, label, docs):
    first_title, first_strict = {}, {}
    for rank, did in enumerate(ranking, 1):
        d = docs[did]; t = d['title']
        if t not in label['gold_titles']: continue
        first_title.setdefault(t, rank)
        if all(normalized(s) in d['normalized_text'] for s in label['supporting_sentences_by_title'][t]):
            first_strict.setdefault(t, rank)
    def threshold(x):
        return max(x.values()) if set(label['gold_titles']) <= set(x) else None
    return {'title': threshold(first_title), 'strict': threshold(first_strict)}

def success(threshold, k): return int(threshold is not None and threshold <= k)
def paired(a, b):
    n = len(a)
    return {'n': n, 'original_successes': sum(a), 'single_successes': sum(b),
            'original_percent': 100 * sum(a) / n if n else None,
            'single_percent': 100 * sum(b) / n if n else None,
            'delta_pp': 100 * (sum(b) - sum(a)) / n if n else None,
            'wins': sum(y > x for x,y in zip(a,b)), 'losses': sum(y < x for x,y in zip(a,b)),
            'ties': sum(y == x for x,y in zip(a,b))}

def failure_partition(records, prefix, k):
    ans = collections.Counter()
    for r in records:
        t = r[prefix]['strict']
        if prefix == 'single' and not r['strict_reachable']: ans['corpus_unavailable'] += 1
        elif t is None: ans['available_but_missing_from_cumulative_returns'] += 1
        elif t > k: ans['returned_but_excluded_by_final_top_k'] += 1
        else: ans['success'] += 1
    return {key:ans[key] for key in ['corpus_unavailable','available_but_missing_from_cumulative_returns','returned_but_excluded_by_final_top_k','success']}

def main():
    protocol = load(EVIDENCE/'corpus_protocol.json'); signature = sha(EVIDENCE/'corpus_protocol.json')
    collected = load(RAW/'collection_manifest.json')
    assert sha(RAW/'run_corpus_sensitivity.py') == protocol['runner_sha256']
    assert sha(RAW/'original_selected.jsonl.gz') == collected['original_selected.jsonl.gz']
    old = {}
    with gzip.open(RAW/'original_selected.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line)
            if r['policy'] in protocol['policies']:
                old[r['dataset'],r['id'],r['policy']] = {k:r[k] for k in ['ranking','selections']}
    existing = {}
    with gzip.open(EVIDENCE/'corpus_per_question.jsonl.gz','rt') as f:
        for line in f:
            r = json.loads(line); existing[r['dataset'],r['id'],r['policy'],r['K']] = r
    manifest = load(STUDY/'data_audit/manifest.json')
    result = {'scope':'Post-hoc reanalysis of previously exposed 600 questions; no new retrieval, no latency claim, no unseen confirmation.',
              'new_analysis':'Independent raw-trace check; exact full-sample loss decomposition; fixed reachable-subset comparison; corpus/candidate-pool/TopK partition; every integer K=1..50.',
              'original_groups':3600,'original_effectiveness_questions':600,'original_metric_records':14400,
              'new_retrieval_calls':0,'datasets':{},'sources':{str(EVIDENCE/'corpus_protocol.json'):signature,
              str(RAW/'run_corpus_sensitivity.py'):sha(RAW/'run_corpus_sensitivity.py'),
              str(RAW/'original_selected.jsonl.gz'):sha(RAW/'original_selected.jsonl.gz'),
              str(EVIDENCE/'corpus_per_question.jsonl.gz'):sha(EVIDENCE/'corpus_per_question.jsonl.gz')}}
    all_records = []
    for ds in ['2wiki','hotpotqa']:
        cp = Path(manifest['datasets'][ds]['corpus_path'])
        assert sha(cp) == protocol['corpus_sha256'][ds]
        result['sources'][str(cp)] = sha(cp)
        docs = {}
        for line in cp.open():
            d = json.loads(line)
            body = hashlib.sha256(d['text'].encode()).hexdigest()
            assert body == d['text_sha256']
            assert d['doc_id'] == 'doc_' + hashlib.sha256((d['title']+'\0'+body).encode()).hexdigest()
            assert d['doc_id'] not in docs
            docs[d['doc_id']] = {**d,'normalized_text':normalized(d['text'])}
        chosen = {}; originals_by_title = collections.defaultdict(list)
        for d in docs.values():
            originals_by_title[d['title']].append(d)
            if d['title'] not in chosen or d['doc_id'] < chosen[d['title']]: chosen[d['title']] = d['doc_id']
        retained = load(RAW/ds/'retained_doc_ids.json')
        chosen_ids = set(chosen.values())
        assert retained == [did for did in docs if did in chosen_ids]
        single = {did:docs[did] for did in retained}; bytitle = {d['title']:d for d in single.values()}
        rules = IndependentRules(single); old_rules = IndependentRules(docs)
        assert rules.dictionary == old_rules.dictionary
        labels = {r['id']:r for r in load(STUDY/'data_audit/offline'/f'{ds}_labels.json')['labels']}
        result['sources'][str(STUDY/'data_audit/offline'/f'{ds}_labels.json')] = sha(STUDY/'data_audit/offline'/f'{ds}_labels.json')
        reachable = {}; lost_bytype = collections.Counter(); all_bytype = collections.Counter(); rows = []; audited = 0
        for q,label in labels.items():
            assert all(any(all(normalized(s) in d['normalized_text'] for s in ss) for d in originals_by_title[t]) for t,ss in label['supporting_sentences_by_title'].items())
            reachable[q] = all(all(normalized(s) in bytitle[t]['normalized_text'] for s in ss) for t,ss in label['supporting_sentences_by_title'].items())
            all_bytype[label['type']] += 1
            if not reachable[q]: lost_bytype[label['type']] += 1
            path = RAW/ds/(q+'.jsonl.gz'); assert sha(path)==collected[f'{ds}/{q}.jsonl.gz']
            with gzip.open(path,'rt') as f: rs = [json.loads(line) for line in f]
            assert len(rs)==6 and {r['policy'] for r in rs}==set(protocol['policies'])
            for r in rs:
                assert r['sensitivity_protocol_sha256']==signature
                pool = rules.pool(r['question'],r['initial'])
                audit_group(r,single,r['alpha'],r['question'],rules=rules,pool=pool); audited += 1
                a = old[ds,q,r['policy']]
                thresholds = {'original':rank_thresholds(a['ranking'],label,docs), 'single':rank_thresholds(r['ranking'],label,single)}
                row = {'dataset':ds,'id':q,'policy':r['policy'],'type':label['type'],'strict_reachable':reachable[q],**thresholds}
                for k in protocol['K']:
                    e = existing[ds,q,r['policy'],k]
                    assert a['selections'][str(k)]==a['ranking'][:k]
                    assert e['strict_reachable']==reachable[q]
                    for prior,new in [('original','original'),('single_version','single')]:
                        for metric,tag in [('F','title'),('strictF','strict')]:
                            assert e[prior+'_'+metric]==success(thresholds[new][tag],k)
                if not reachable[q]: assert thresholds['single']['strict'] is None
                rows.append(row)
        sub = {'n':len(labels),'raw_groups_reaudited':audited,'original_documents':len(docs),'single_documents':len(single),
               'title_dictionary_identical':True,'retained_order_identical':True,'all_original_doc_id_and_body_hashes_verified':True,
               'strict_reachable_n':sum(reachable.values()),'strict_unreachable_n':len(labels)-sum(reachable.values()),
               'strict_unreachable_ids':sorted(q for q,v in reachable.items() if not v),
               'question_type_composition':{t:{'all_n':n,'unreachable_n':lost_bytype[t],'unreachable_percent_within_type':100*lost_bytype[t]/n} for t,n in sorted(all_bytype.items())},
               'policies':{},'reachable_method_contrasts_at50':{}}
        for p in protocol['policies']:
            pr = [r for r in rows if r['policy']==p]; sr = [r for r in pr if r['strict_reachable']]; ur = [r for r in pr if not r['strict_reachable']]
            grid = {}
            for k in range(1,51):
                a = [success(r['original']['strict'],k) for r in pr]; b = [success(r['single']['strict'],k) for r in pr]
                pa = [success(r['original']['strict'],k) for r in sr]; pb = [success(r['single']['strict'],k) for r in sr]
                removed_successes = sum(success(r['original']['strict'],k) for r in ur)
                net_reachable = sum(pb)-sum(pa)
                assert sum(b)-sum(a)==net_reachable-removed_successes
                grid[str(k)] = {'strict_full_sample':paired(a,b),'strict_fixed_reachable_subset':paired(pa,pb),
                    'title_full_sample':paired([success(r['original']['title'],k) for r in pr],[success(r['single']['title'],k) for r in pr]),
                    'decomposition_counts':{'unavailable_original_successes_lost':removed_successes,'reachable_net_success_change':net_reachable,'total_net_success_change':sum(b)-sum(a)},
                    'strict_failure_partition_original':failure_partition(pr,'original',k),'strict_failure_partition_single':failure_partition(pr,'single',k)}
            sub['policies'][p] = grid
        for control in ['one_target','first_two_valid_titles']:
            lookup = {(r['id'],r['policy']):r for r in rows}
            ids = sorted(q for q,v in reachable.items() if v)
            gaps = {}
            for variant in ['original','single']:
                vector = [success(lookup[q,'no_second_branch'][variant]['strict'],50)-success(lookup[q,control][variant]['strict'],50) for q in ids]
                gaps[variant] = {'n':len(ids),'net_successes':sum(vector),'gap_pp':100*sum(vector)/len(ids),'difference_vector':vector}
            delta = [b-a for a,b in zip(gaps['original']['difference_vector'],gaps['single']['difference_vector'])]
            gaps['change_in_gap_pp'] = 100*sum(delta)/len(delta)
            gaps['paired_difference_vector_identical'] = not any(delta)
            for v in ['original','single']: del gaps[v]['difference_vector']
            sub['reachable_method_contrasts_at50'][control] = gaps
        result['datasets'][ds] = sub; all_records.extend(rows)
        print(ds, 'raw groups verified',audited,'reachable',sum(reachable.values()),flush=True)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with gzip.open(OUT/'paired_rank_thresholds.jsonl.gz','wt') as f:
        for r in all_records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print('saved',OUT/'results.json',flush=True)

if __name__=='__main__': main()
