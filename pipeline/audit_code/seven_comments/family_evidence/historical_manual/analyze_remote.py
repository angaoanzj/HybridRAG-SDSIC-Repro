"""Offline full-history Family audit; no retriever, encoder, network, or training.
Phase 1 validates retained traces and computes paired title coverage by changed group.
Phase 2 reconstructs the frozen candidate pool independently for the activation funnel.
"""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,itertools,json,os,re,time,sys,traceback
P=Path(__file__).resolve().parent
H=Path('/root/autodl-tmp/hybridrag-full-title-policy-20261106')
C=Path('/root/autodl-tmp/hybridrag-paper-repair-20261105/data')
KS=[2,5,10,15,20,30,40,50]
word=lambda s:tuple(re.findall('[a-z0-9]+',s.lower()))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def write(name,x):
 t=P/(name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(P/name)
def ent(r):return r['expansion_provenance'].get('entities',[]) if isinstance(r['expansion_provenance'],dict) else []
def sig(e):return [e['entity_tokens'],e['doc_id'],e['source_text_span'][0],e['mentioned_title']]
def raw_calls(r,q0=False):
 return [(c['query_id'],c['route'],c['query'],c['depth'],c['returned_compact']) for c in r['calls'] if not q0 or c['query_id']=='q0']
def first_ranks(ranking,titles,gold):
 out={}
 for rank,di in enumerate(ranking,1):
  t=titles[di]
  if t in gold:out.setdefault(t,rank)
 return out

def main():
 os.nice(19);start=time.time();protocol=json.loads((H/'protocol.json').read_text())
 expected=json.loads((P/'expected_effects.json').read_text())
 code_checks={n:sha(H/'code'/n)==v for n,v in protocol['code_sha256'].items()};assert all(code_checks.values())
 out={'scope':'Expanded historical fixed-context evaluation. Original300 is a subset, not an independent sample. Metrics below are title F, not answer accuracy or strictF. No new retrieval or timing.','K':KS,'phase':'trace','datasets':{},'code_hashes_match_frozen_protocol':code_checks,'source_protocol_sha256':sha(H/'protocol.json'),'shard_hashes_verified':0,'inference_rule':'After selecting the first entity, the frozen family policy takes the second from an unused explicit source family if any exists. Thus its selected second occurrence has that property iff this preferred pool exists. Phase2 independently rebuilds candidates to check this implication.','source_files':{}}
 for ds in ('2wiki','hotpotqa'):
  docids=json.loads((H/'data'/f'{ds}_docids.json').read_text());di={d:i for i,d in enumerate(docids)};titles=[None]*len(docids)
  corpus=C/ds/'corpus.jsonl';assert sha(corpus)==protocol['datasets'][ds]['corpus_sha256']
  for ln in corpus.open():
   d=json.loads(ln);titles[di[d['doc_id']]]=d['title']
  labels={x['id']:x for x in json.loads((H/'data'/f'{ds}_offline_labels.json').read_text())};public={x['id']:x['question'] for x in json.loads((H/'data'/f'{ds}_public.json').read_text())}
  oldids=set(expected['original_ids'][ds]);assert oldids<=labels.keys()
  counts=Counter();pair=defaultdict(lambda:Counter());changes=[];seen=set()
  with gzip.open(P/f'{ds}_selection_inputs.jsonl.gz','wt') as inp:
   for shard in sorted((H/'runs').glob(ds+'_*.jsonl.gz')):
    receipt=json.loads((H/'receipts'/(shard.name.removesuffix('.jsonl.gz')+'.json')).read_text());assert sha(shard)==receipt['sha256'];out['shard_hashes_verified']+=1
    with gzip.open(shard,'rt') as f:
     for qid,rr in itertools.groupby((json.loads(ln) for ln in f),key=lambda r:r['id']):
      rows={r['policy']:r for r in rr};assert len(rows)==8;assert qid not in seen;seen.add(qid)
      a=rows['no_second_branch'];b=rows['bridge_expand'];ae,be=ent(a),ent(b);gold=set(labels[qid]['gold_titles'])
      assert a['initial']==b['initial'];assert raw_calls(a,True)==raw_calls(b,True)
      ac,bc=[sig(e) for e in ae],[sig(e) for e in be];at=[e['mentioned_title'] for e in ae];bt=[e['mentioned_title'] for e in be]
      first_changed=ac[:1]!=bc[:1];occ=ac!=bc;tc=at!=bt;qc=a['expanded_query']!=b['expanded_query']
      assert not first_changed;assert (a['expansion_provenance'].get('residual_keywords')==b['expansion_provenance'].get('residual_keywords'))
      pref=bool(len(be)==2 and be[1]['source_is_question_entity'] and word(be[1]['source_title'])!=word(be[0]['source_title']))
      group='query_changed' if qc else ('occurrence_only_same_query' if occ else ('preferred_pool_same_occurrence' if pref else 'no_preferred_pool'))
      for k,v in {'questions':1,'initial_identical':1,'q0_physical_calls_identical':1,'first_occurrence_changed':first_changed,'first_title_changed':at[:1]!=bt[:1],'occurrences_changed':occ,'titles_changed':tc,'query_changed':qc,'occurrence_changed_query_same':occ and not qc,'top50_changed':a['ranking'][:50]!=b['ranking'][:50],'full_ranking_changed':a['ranking']!=b['ranking'],'all_physical_calls_identical':raw_calls(a)==raw_calls(b),'preferred_pool_inferred_from_selected_provenance':pref,'preferred_pool_same_occurrence':pref and not occ}.items():counts[k]+=v
      ar=first_ranks(a['ranking'],titles,gold);br=first_ranks(b['ranking'],titles,gold)
      lim_a=max(ar.values()) if gold<=ar.keys() else 10**9;lim_b=max(br.values()) if gold<=br.keys() else 10**9
      subtype='original_300' if qid in oldids else 'remainder_excluding_original_300'
      for k in KS:
       fa,fb=int(lim_a<=k),int(lim_b<=k);outcome='wins' if fb>fa else ('losses' if fb<fa else 'ties')
       groups=set(['all',group,'query_changed' if qc else 'same_query',subtype])
       for g in groups:
        t=pair[g,k];t['n']+=1;t['reference_successes']+=fa;t['family_successes']+=fb;t[outcome]+=1
        if outcome=='losses':t['loss_all_support_titles_still_returned']+=int(lim_b<10**9)
        if outcome=='wins':t['win_support_absent_from_reference_pool']+=int(lim_a==10**9)
      if occ or qc:
       changes.append({'id':qid,'original_300':qid in oldids,'group':group,'score_titles':at,'family_titles':bt,'score_sources':[e['source_title'] for e in ae],'family_sources':[e['source_title'] for e in be],'score_complete_rank':None if lim_a==10**9 else lim_a,'family_complete_rank':None if lim_b==10**9 else lim_b,'support_ranks':{t:{'score':ar.get(t),'family':br.get(t)} for t in gold}})
      inp.write(json.dumps({'id':qid,'initial':[docids[i] for i in a['initial'][:5]],'score_chosen':ac,'family_chosen':bc,'score_query':a['expanded_query'],'family_query':b['expanded_query'],'preferred_pool_inferred':pref,'group':group},ensure_ascii=False,separators=(',',':'))+'\n')
    write('progress.json',{'phase':'trace','dataset':ds,'processed':len(seen),'elapsed_seconds':time.time()-start})
  assert seen==labels.keys()
  grids=[]
  for (group,k),c in sorted(pair.items()):
   d=dict(c);d.update({'group':group,'K':k,'delta_pp':100*(c['family_successes']-c['reference_successes'])/c['n']});grids.append(d)
   if group=='all':
    e=next(r for r in expected['paired_tables'] if r['dataset']==ds and r['contrast']=='family' and r['subset']=='full' and r['metric']=='F' and r['K']==k)
    assert all(c[key]==e[key] for key in ('n','wins','losses','ties'))
  # Any unchanged query must reproduce returned lists and rankings in the historical run.
  assert counts['all_physical_calls_identical']==counts['questions']-counts['query_changed']
  assert counts['full_ranking_changed']==counts['query_changed']
  out['datasets'][ds]={'counts':dict(counts),'paired_by_group':grids,'changed_cases':changes,'conditional_effect_notice':'Changed-group percentages are post-hoc conditional diagnostics, not replacement main evaluation denominators.'}
  out['source_files'][ds]={'corpus_sha256':sha(corpus),'public_sha256':sha(H/'data'/f'{ds}_public.json'),'labels_sha256':sha(H/'data'/f'{ds}_offline_labels.json'),'docids_sha256':sha(H/'data'/f'{ds}_docids.json')}
  write('trace_summary.json',out)
  print(ds,dict(counts),flush=True)
 out['trace_complete']=True;out['trace_elapsed_seconds']=time.time()-start;write('trace_summary.json',out)
 # Rebuild only candidate extraction/selection; no retrieval or neural models imported.
 from frozen_independent_rules import IndependentRules
 checks={}
 for ds in ('2wiki','hotpotqa'):
  docs={d['doc_id']:d for d in map(json.loads,(C/ds/'corpus.jsonl').open())};rules=IndependentRules(docs)
  public={x['id']:x['question'] for x in json.loads((H/'data'/f'{ds}_public.json').read_text())}
  count=Counter();mismatches=[];group_counts=Counter()
  with gzip.open(P/f'{ds}_selection_inputs.jsonl.gz','rt') as f:
   for ln in f:
    r=json.loads(ln);qid=r['id'];pool=rules.pool(public[qid],r['initial']);ca,qa=rules.expected('no_second_branch',public[qid],r['initial'],pool);cb,qb=rules.expected('bridge_expand',public[qid],r['initial'],pool)
    cs=lambda cc:[[list(c['key']),c['doc_id'],c['start'],c['title']] for c in cc]
    match=cs(ca)==r['score_chosen'] and cs(cb)==r['family_chosen'] and qa==r['score_query'] and qb==r['family_query']
    preferred=bool(ca and any(c['key']!=ca[0]['key'] and c['source_explicit'] and c['source_family']!=ca[0]['source_family'] for c in pool[0]))
    if preferred!=r['preferred_pool_inferred']:match=False
    count['questions']+=1;count['selected_occurrences_and_queries_match']+=match;count['preferred_pool_available']+=preferred
    count['preferred_pool_same_occurrence']+=preferred and r['score_chosen']==r['family_chosen'];count['preferred_pool_occurrence_changed_query_same']+=preferred and r['score_chosen']!=r['family_chosen'] and r['score_query']==r['family_query'];count['preferred_pool_query_changed']+=preferred and r['score_query']!=r['family_query'];group_counts[r['group']]+=1
    if not match and len(mismatches)<10:mismatches.append({'id':qid,'expected_score':cs(ca),'observed_score':r['score_chosen'],'expected_family':cs(cb),'observed_family':r['family_chosen'],'expected_queries':[qa,qb],'observed_queries':[r['score_query'],r['family_query']],'pool_available':preferred,'inferred':r['preferred_pool_inferred']})
    if count['questions']%250==0:write('progress.json',{'phase':'candidate_reconstruction','dataset':ds,'processed':count['questions'],'elapsed_seconds':time.time()-start})
  checks[ds]={'counts':dict(count),'groups':dict(group_counts),'mismatches_first10':mismatches,'verified_all':count['questions']==count['selected_occurrences_and_queries_match']}
  write('candidate_reconstruction.json',checks);print(ds,'candidate',dict(count),flush=True)
 out['phase']='complete';out['candidate_reconstruction']=checks;out['all_candidate_reconstruction_matches']=all(v['verified_all'] for v in checks.values());out['total_elapsed_seconds']=time.time()-start
 write('summary.json',out);write('complete.json',{'status':'complete' if out['all_candidate_reconstruction_matches'] else 'completed_with_candidate_mismatch','questions':sum(d['counts']['questions'] for d in out['datasets'].values()),'shards_sha_verified':out['shard_hashes_verified'],'summary_sha256':sha(P/'summary.json'),'trace_sha256':sha(P/'trace_summary.json'),'script_sha256':sha(Path(__file__)),'rules_sha256':sha(P/'frozen_independent_rules.py'),'elapsed_seconds':time.time()-start})
if __name__=='__main__':
 try:main()
 except Exception:
  write('failed.json',{'error':traceback.format_exc()});raise
