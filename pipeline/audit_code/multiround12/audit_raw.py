"""Independent saved-record replay. Never imports or executes retrieval/model code."""
import collections
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

RUN=Path('/path/to/local/home/Downloads/hybridrag-musique-multiround-results-20261109T074918Z-151828')
DATA=Path('/Volumes/algorithm/HybridRAG_workspace/rag返修2/musique_extension_20261107/data/musique')
OUT=Path('/tmp/hybridrag_multiround_audit_20261110/raw')
K=(10,20,30,50)
protocol=json.loads((RUN/'protocol.json').read_text())
expected=protocol['original_manifest']
checks=collections.Counter()
failures=collections.Counter()
examples=[]

def check(ok, name, context=None):
    checks[name]+=1
    if not ok:
        failures[name]+=1
        if len(examples)<40: examples.append({'check':name,'context':context})

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

hashes={}
for fn,key in [('public.json','public_sha256'),('labels.json','labels_sha256'),('corpus.jsonl','corpus_sha256')]:
    actual=sha(DATA/fn); hashes[fn]=actual
    check(actual==expected[key],'original_input_hash',fn)
if failures: raise RuntimeError('Original source hashes mismatch')
public_list=json.loads((DATA/'public.json').read_text())
public={r['id']:r for r in public_list}
labels={r['id']:r for r in json.loads((DATA/'labels.json').read_text())}
with (DATA/'corpus.jsonl').open() as f: corpus={r['doc_id']:r for r in map(json.loads,f)}
corpus_titles={r['title'] for r in corpus.values()}
check([r['id'] for r in public_list]==protocol['question_ids'],'original_question_order')
check(len(public)==2417 and len(labels)==2417,'original_question_label_size')
with gzip.open(RUN/'quality_rows.jsonl.gz','rt') as f:
    quality_list=[json.loads(s) for s in f]
quality={(r['id'],r['arm']):r for r in quality_list}
check(len(quality_list)==len(quality)==21753,'quality_unique_rows')

# Independent implementation of the archived rank-vote formula.
def fusion(calls):
    count=len({c['query_id'] for c in calls})
    votes=collections.defaultdict(float)
    for c in calls:
        w=(protocol['alpha'] if c['route']=='dense' else 1-protocol['alpha'])/max(count,1)
        seen=set()
        for d in c['returned']:
            if d['doc_id'] not in seen:
                votes[d['doc_id']]+=w/(60+d['rank'])
                seen.add(d['doc_id'])
    return sorted(votes,key=lambda d:(-votes[d],d))

def metric(ranked,label):
    gold=set(label['gold_doc_ids']); titles=set(label['gold_titles'])
    pos={d:i for i,d in enumerate(ranked,1)}
    out={}
    for k in K:
        selected=set(ranked[:k]); actual_titles={corpus[d]['title'] for d in selected}
        out[str(k)]={'FP':int(gold<=selected),'paragraph_recall':len(gold&selected)/len(gold),'FT':int(titles<=actual_titles)}
    return {'at_k':out,'gold_ranks':{d:pos.get(d) for d in sorted(gold)}}

def norm(s): return ' '.join(s.casefold().split())

rows_seen=set(); call_ids=set(); question_order=[]; last_qid=None
counts=collections.Counter(); mechanisms={}; stop_reasons=collections.defaultdict(collections.Counter)
repeat_info=collections.Counter(); deterministic={}
start=time.time()
with gzip.open(RUN/'raw_rankings_and_calls.jsonl.gz','rt') as stream:
    for line_no,line in enumerate(stream,1):
        raw=json.loads(line); qid=raw['id']; arm=raw['arm']; key=(qid,arm)
        ctx={'row':line_no,'id':qid,'arm':arm}
        check(key not in rows_seen,'raw_unique_row',ctx); rows_seen.add(key)
        if qid!=last_qid: question_order.append(qid); last_qid=qid
        check(key in quality,'raw_quality_key',ctx)
        q=quality[key]; label=labels[qid]; conf=protocol['arms'][arm]
        rounds=raw['rounds']; calls=raw['calls']; pool=set(); seen_queries=set(); used_titles=set()
        check(1<=len(rounds)<=len(conf['depths']),'executed_round_bounds',ctx)
        check(len(calls)==2*len(rounds),'two_routes_per_round',ctx)
        check(raw['executed_query_instances']==len(rounds),'executed_instances',ctx)
        check(raw['supplementary_queries']==len(rounds)-1,'supplementary_count',ctx)
        check(raw['physical_route_calls']==len(calls),'physical_calls',ctx)
        requested=sum(c['depth'] for c in calls); returned=sum(len(c['returned']) for c in calls)
        check(requested==raw['requested_records'] and returned==raw['returned_records'],'record_totals',ctx)
        check(0<=returned<=requested<=400,'budget_cap',ctx)
        for field in ('physical_route_calls','executed_query_instances','supplementary_queries','requested_records','returned_records','stop_reason'):
            check(raw[field]==q[field],'raw_quality_'+field,ctx)
        check(q['hops']==label['hops'],'hops_match',ctx)
        stop_reasons[arm][raw['stop_reason']]+=1
        if len(rounds)==len(conf['depths']):check(raw['stop_reason']=='scheduled_depths_completed','full_schedule_stop',ctx)
        else:check(raw['stop_reason'].startswith('q'+str(len(rounds))+'_'),'early_stop_round',ctx)
        prior_rank=None; prior_metrics=None
        for j,r in enumerate(rounds):
            rc={**ctx,'round':j}; round_calls=calls[2*j:2*j+2]
            check(r['round']==j,'round_index',rc)
            check(r['depth_per_route']==conf['depths'][j],'depth_schedule',rc)
            mode='original' if j==0 else conf['modes'][j-1]
            check(r['mode']==mode,'mode_schedule',rc)
            if j==0:check(r['query']==public[qid]['question'],'original_query_exact',rc)
            elif mode=='repeat':check(r['query']==rounds[j-1]['query'],'repeated_query_exact',rc)
            elif j==1:check(bool(r['query']) and r['query'].strip()!=public[qid]['question'].strip(),'first_expansion_trigger',rc)
            else:check(norm(r['query']) not in seen_queries,'new_query_not_executed',rc)
            seen_queries.add(norm(r['query']))
            check([c['route'] for c in round_calls]==['bm25','dense'],'route_order',rc)
            old_pool=set(pool)
            for c in round_calls:
                check(c['round']==j and c['query_id']=='q'+str(j),'call_round_query_id',rc)
                check(c['query']==r['query'] and c['depth']==r['depth_per_route'],'call_query_depth',rc)
                check(c['call_id'] not in call_ids,'unique_call_id',rc); call_ids.add(c['call_id'])
                check(not c['cache_hit'] and not c['query_cache_enabled'],'query_cache_disabled',rc)
                rec=c['returned']; docids=[d['doc_id'] for d in rec]
                check(len(rec)<=c['depth'],'return_depth',rc)
                check(len(docids)==len(set(docids)),'route_unique_documents',rc)
                check([d['rank'] for d in rec]==list(range(1,len(rec)+1)),'consecutive_route_ranks',rc)
                check(all(d in corpus for d in docids),'returned_documents_in_original_corpus',rc)
                check(all(math.isfinite(d['raw_score']) for d in rec),'finite_route_scores',rc)
                check(all(rec[t]['raw_score']>=rec[t+1]['raw_score'] for t in range(len(rec)-1)),'score_rank_descending',rc)
                pool.update(docids)
                identity=(c['query'],c['route'],c['depth'])
                digest=hashlib.sha256(json.dumps(rec,sort_keys=True).encode()).hexdigest()
                if identity in deterministic:check(digest==deterministic[identity],'deterministic_same_query_route_depth',rc)
                else:deterministic[identity]=digest
                if mode=='repeat':
                    previous=calls[2*(j-1)+(0 if c['route']=='bm25' else 1)]
                    check(rec==previous['returned'],'repeat_identical_returns',rc)
                    repeat_info['repeated_route_calls']+=1
            if mode=='repeat':
                check(pool==old_pool,'repeat_adds_no_candidate',rc)
                repeat_info['executed_repeated_queries']+=1
            ranking=fusion(calls[:2*(j+1)])
            check(ranking==r['ranking_after_round'],'replayed_round_ranking',rc)
            if j==len(rounds)-1:check(ranking==raw['ranking'],'replayed_final_ranking',rc)
            scored=metric(ranking,label)
            check(scored==q['metrics_after_round'][j],'independent_round_metrics_gold_ranks',rc)
            digest=hashlib.sha256(json.dumps({'query':r['query'],'ranking':ranking},sort_keys=True).encode()).hexdigest()
            check(digest==q['trajectory_digests'][j],'trajectory_digest',rc)
            check(r['selected_titles']==q['round_titles'][j],'quality_selected_titles',rc)
            novelty=sorted(set(r['selected_titles'])-used_titles)
            check(novelty==r['new_titles'] and len(novelty)==q['new_title_counts'][j],'new_titles',rc)
            used_titles.update(r['selected_titles'])
            if j:
                proposal=r['proposal']; entities=proposal.get('provenance',{}).get('entities',[])
                check(r['selected_titles']==[e['mentioned_title'] for e in entities],'proposal_selected_titles',rc)
                check(proposal['query']==r['query'],'proposal_query',rc)
                if mode=='new':
                    for e in entities:
                        counts['title_mentions']+=1
                        doc=corpus.get(e['doc_id'])
                        check(doc is not None and e['mentioned_title'] in corpus_titles,'entity_source_and_title_exist',rc)
                        check(e['source_rank']>=1 and e['source_rank']<=5 and prior_rank[e['source_rank']-1]==e['doc_id'],'entity_in_prior_top5',rc)
                        check(doc['title']==e['source_title'],'entity_source_title',rc)
                        span=e.get('source_text_span')
                        if span is not None:check(doc['text'][span[0]:span[1]]==e['mention_text'],'entity_exact_source_span',rc)
                        check(e['sentence'] in doc['text'],'entity_sentence_in_source',rc)
            if j>=2:
                mkey=arm+':q'+str(j)
                m=mechanisms.setdefault(mkey,{'executed':0,'mode':mode,'new_candidate_total':0,'new_gold_total':0,'at_k':{str(k):{'wins':0,'losses':0,'win_new_gold_acquisition':0,'win_ranking_promotion':0} for k in K}})
                m['executed']+=1; m['new_candidate_total']+=len(pool-old_pool)
                m['new_gold_total']+=len((set(label['gold_doc_ids'])&pool)-old_pool)
                for k in K:
                    delta=scored['at_k'][str(k)]['FP']-prior_metrics['at_k'][str(k)]['FP']; out=m['at_k'][str(k)]
                    if delta>0:
                        out['wins']+=1
                        out['win_ranking_promotion' if set(label['gold_doc_ids'])<=old_pool else 'win_new_gold_acquisition']+=1
                    if delta<0:out['losses']+=1
            prior_rank=ranking; prior_metrics=scored
            counts['rounds']+=1
        check(scored['at_k']==q['at_k'] and scored['gold_ranks']==q['gold_ranks'],'independent_final_metrics_gold_ranks',ctx)
        check(set(raw['ranking'])==pool,'final_candidate_pool_union',ctx)
        counts['rows']+=1; counts['physical_calls']+=len(calls)
        if line_no%2500==0:print(json.dumps({'processed':line_no,'failures':dict(failures),'audit_seconds':round(time.time()-start,2)}),flush=True)
check(rows_seen==set(quality),'raw_quality_complete_key_set')
check(question_order==protocol['question_ids'],'raw_question_block_order')
report={'status':'PASS' if not failures else 'FAIL','scope':'Independent replay of saved route returns and rank fusion; no fresh retrieval/model execution. Original labels/corpus independently loaded and hashed. Selector logic and unexecuted stopping proposals not regenerated. Timing fields not validated as a latency experiment.','input_directory':str(RUN),'original_input_hashes':hashes,'counts':dict(counts),'checks':dict(checks),'failure_counts':dict(failures),'failure_examples':examples,'repeat_checks':dict(repeat_info),'distinct_query_route_depth_signatures':len(deterministic),'stop_reasons':{a:dict(v) for a,v in stop_reasons.items()},'per_executed_later_query_mechanisms':mechanisms,'audit_wall_seconds':time.time()-start}
(OUT/'raw_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'counts':report['counts'],'failures':report['failure_counts'],'report':str(OUT/'raw_audit.json'),'seconds':report['audit_wall_seconds']},ensure_ascii=False),flush=True)
