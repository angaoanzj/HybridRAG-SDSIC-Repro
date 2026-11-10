"""Read-only input audit: aggregate saved quality and replay saved route rankings.

No retrieval, training, title selection, or model execution is performed.
"""
from pathlib import Path
from collections import Counter, defaultdict
import csv, gzip, hashlib, json, math, time, zipfile

ROOT = Path('/path/to/local/home/Downloads/hybridrag-musique-multiround-results-20261109T074918Z-151828')
OUT = Path(__file__).resolve().parent
ARCHIVE = Path('/path/to/local/home/Downloads/HybridRAG_SDSIC_Clarity_R2_20261109/MuSiQue_MultiRound_Audit_20261110.zip')
protocol = json.loads((ROOT/'protocol.json').read_text())
summary = json.loads((ROOT/'summary.json').read_text())
complete = json.loads((ROOT/'complete.json').read_text())
checks = Counter(); errors = []
def check(condition, name, detail=None):
    checks[name] += 1
    if not condition and len(errors) < 100:
        errors.append({'check':name,'detail':detail})
def same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b
hashes={}
for fn, key in [('protocol.json','protocol_sha256'),('summary.json','summary_sha256'),('quality_rows.jsonl.gz','quality_sha256'),('raw_rankings_and_calls.jsonl.gz','raw_sha256')]:
    h=hashlib.sha256()
    with (ROOT/fn).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    hashes[fn]=h.hexdigest()
    check(h.hexdigest()==complete[key],'completion_hash',fn)
with gzip.open(ROOT/'quality_rows.jsonl.gz','rt') as f:
    rows=[json.loads(line) for line in f]
ids=protocol['question_ids']; arms=protocol['arms']; ks=protocol['K']; n=len(ids)
by={(r['id'],r['arm']):r for r in rows}
check(n==2417 and len(set(ids))==n,'question_ids')
check(len(by)==len(rows)==n*len(arms)==21753,'dimensions')
check(set(by)=={(qid,arm) for qid in ids for arm in arms},'exact_cartesian_product')
check(complete['status']==summary['status']=='COMPLETED','completion_status')
check(ks==[10,20,30,50] and protocol['D_cap']==400,'cutoffs_and_budget')
aggregates={}; table=[]
for arm in arms:
    selected=[by[qid,arm] for qid in ids]
    check([r['id'] for r in rows if r['arm']==arm]==ids,'arm_order',arm)
    result={'n':n,'supplementary_distribution':dict(Counter(str(r['supplementary_queries']) for r in selected)),
            'mean_physical_route_calls':sum(r['physical_route_calls'] for r in selected)/n,
            'mean_returned_records':sum(r['returned_records'] for r in selected)/n,
            'at_k':{str(k):{m:sum(r['at_k'][str(k)][m] for r in selected)/n for m in ['FP','FT','paragraph_recall']} for k in ks}}
    check(same(result,summary['arms'][arm]),'arm_summary',arm)
    aggregates[arm]=result
    for k in ks:
        table.append({'arm':arm,'K':k,'N':n,'FP_count':sum(r['at_k'][str(k)]['FP'] for r in selected),
                      'FT_count':sum(r['at_k'][str(k)]['FT'] for r in selected),
                      'FP_percent':100*result['at_k'][str(k)]['FP'],
                      'FT_percent':100*result['at_k'][str(k)]['FT'],
                      'paragraph_recall_percent':100*result['at_k'][str(k)]['paragraph_recall'],
                      'mean_route_calls':result['mean_physical_route_calls'],
                      'mean_returned_records':result['mean_returned_records']})
paired={}
for name, explanation in protocol['comparisons'].items():
    a,b=name.split('-minus-')
    result={'interpretation':explanation,'same_realized_D_n':sum(by[qid,a]['returned_records']==by[qid,b]['returned_records'] for qid in ids),'at_k':{}}
    for k in ks:
        delta=[by[qid,a]['at_k'][str(k)]['FP']-by[qid,b]['at_k'][str(k)]['FP'] for qid in ids]
        result['at_k'][str(k)]={'wins':sum(d>0 for d in delta),'losses':sum(d<0 for d in delta),'delta_pp':100*sum(delta)/n}
    paired[name]=result
    check(same(result,summary['paired'][name]),'paired_summary',name)
raw_counts=Counter(); seen=set(); mechanisms={}; start=time.monotonic()
with gzip.open(ROOT/'raw_rankings_and_calls.jsonl.gz','rt') as stream:
    for line in stream:
        raw=json.loads(line); key=(raw['id'],raw['arm']); q=by[key]
        check(key not in seen,'raw_unique_key',key);seen.add(key)
        calls=raw['calls']; rounds=raw['rounds']; raw_counts['rows']+=1; raw_counts['physical_calls']+=len(calls);raw_counts['rounds']+=len(rounds)
        check(len(calls)==raw['physical_route_calls']==q['physical_route_calls']==2*len(rounds),'physical_calls',key)
        check(sum(len(c['returned']) for c in calls)==raw['returned_records']==q['returned_records'],'returned_records',key)
        check(sum(c['depth'] for c in calls)==raw['requested_records']<=400,'budget_cap',key)
        gold=set(q['gold_ranks']);previous_pool=set()
        for ri,rd in enumerate(rounds):
            prefix=[c for c in calls if c['round']<=ri]
            counts=len({c['query_id'] for c in prefix});votes=defaultdict(float)
            for c in prefix:
                weight=(protocol['alpha'] if c['route']=='dense' else 1-protocol['alpha'])/counts
                route_seen=set()
                for d in c['returned']:
                    if d['doc_id'] not in route_seen:
                        votes[d['doc_id']]+=weight/(60+d['rank']);route_seen.add(d['doc_id'])
            ranking=sorted(votes,key=lambda d:(-votes[d],d))
            check(ranking==rd['ranking_after_round'],'replayed_round_ranking',key)
            positions={d:i for i,d in enumerate(ranking,1)}
            qm=q['metrics_after_round'][ri]
            check({d:positions.get(d) for d in gold}==qm['gold_ranks'],'gold_ranks',key)
            for k in ks:
                hits=len(set(ranking[:k])&gold)
                check(int(hits==len(gold))==qm['at_k'][str(k)]['FP'] and math.isclose(hits/len(gold),qm['at_k'][str(k)]['paragraph_recall']),'FP_and_paragraph_recall',key)
            pool=set(ranking)
            if ri>=2:
                label=raw['arm']+':q'+str(ri)
                m=mechanisms.setdefault(label,{'executed':0,'mode':rd['mode'],'new_candidate_total':0,'new_gold_total':0,'at_k':{str(k):{'wins':0,'losses':0,'win_new_gold_acquisition':0,'win_ranking_promotion':0} for k in ks}})
                m['executed']+=1;m['new_candidate_total']+=len(pool-previous_pool);m['new_gold_total']+=len((pool-previous_pool)&gold)
                for k in ks:
                    delta=qm['at_k'][str(k)]['FP']-q['metrics_after_round'][ri-1]['at_k'][str(k)]['FP']
                    v=m['at_k'][str(k)]
                    if delta>0:
                        v['wins']+=1
                        v['win_new_gold_acquisition' if not gold<=previous_pool else 'win_ranking_promotion']+=1
                    elif delta<0:v['losses']+=1
            previous_pool=pool
        check(ranking==raw['ranking'],'replayed_final_ranking',key)
        check(q['gold_ranks']==q['metrics_after_round'][-1]['gold_ranks'],'final_gold_ranks',key)
check(seen==set(by),'raw_quality_identity')
with zipfile.ZipFile(ARCHIVE) as archive:
    raw_audit=json.loads(archive.read('MuSiQue_MultiRound_Audit_20261110/raw_audit.json'))
    quality_audit=json.loads(archive.read('MuSiQue_MultiRound_Audit_20261110/quality_audit.json'))
check(raw_audit['status']=='PASS' and not raw_audit['failure_counts'],'archived_raw_audit_pass')
check(quality_audit['status']=='PASS_SAVED_QUALITY_AUDIT' and not quality_audit['failures'],'archived_quality_audit_pass')
check(same(mechanisms,raw_audit['per_executed_later_query_mechanisms']),'later_query_mechanisms_match_existing_independent_audit')
with (OUT/'all_nine_arms_recomputed.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
report={'status':'PASS' if not errors else 'FAIL','scope':'Independent local reaggregation and saved-route replay; no retrieval/model execution.',
        'inputs':str(ROOT),'archive':str(ARCHIVE),'hashes':hashes,'n':n,'arms':len(arms),'K':ks,'D_cap':protocol['D_cap'],
        'raw_counts':dict(raw_counts),'checks':dict(checks),'failures':errors,'arm_summary':aggregates,'paired_summary':paired,
        'later_query_mechanisms':mechanisms,'raw_replay_seconds':time.monotonic()-start,
        'limits':['No title selector rerun and no unexecuted proposal reconstruction.','FT independently aggregated from quality records; original corpus/title-label replay relies on the packaged prior independent audit because external source volume is absent.','No new significance inference, reader EM/F1, or matched latency evaluation.','K grid is four points 10/20/30/50, not full integer range.','Public development cohort is historical descriptive evidence; it must remain separate from later source-screened cohorts.']}
(OUT/'independent_reaggregation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'counts':dict(raw_counts),'checks':sum(checks.values()),'failures':errors,'seconds':report['raw_replay_seconds']},ensure_ascii=False))
