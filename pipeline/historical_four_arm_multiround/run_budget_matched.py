"""New real CPU retrieval: four prespecified allocation/iteration controls.
Uses existing v2 immutable corpus/index; writes only into --output.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,sys,json,time,gzip,hashlib,random,collections,traceback
from pathlib import Path

ARM_NAMES=['A_single100','B_iterative50','C_repeat50','D_single50']
def atomic(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v2-root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--n',type=int,default=100);ap.add_argument('--timing-n',type=int,default=10);args=ap.parse_args()
    root=Path(args.v2_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists() or (out/'started.json').exists():raise RuntimeError('Refuse to overwrite an existing run')
    os.environ['MULTIROUND_V1_ROOT']=str(root/'v1');sys.path.insert(0,str(root/'code'))
    import core
    import numpy as np
    import torch
    torch.set_num_threads(2)
    legacy=core.legacy
    protocol={'created_unix':time.time(),'date':'2026-11-07','n_per_dataset':args.n,'datasets':['2wiki','hotpotqa'],
      'selection':'first n records in v2 already observed confirmation file, frozen before this run; exploratory only',
      'K':list(range(1,51)),'alpha':legacy.ALPHAS,'rrf_constant':60,'title_policy':'frozen no_second_branch score-two; same bridge_query at each round',
      'candidate_source':'actual fused ranking from q0 top100 per route plus q1 top50 per route; bridge_query uses its original top-five source rule',
      'arms':{'A_single100':[100,100],'B_iterative50':[100,50,50],'C_repeat50':[100,50,50],'D_single50':[100,50]},
      'depth_note':'each value is per-route requested depth; two routes; skipped queries never refill budget',
      'q2_rule':'B uses regenerated bridge query only if nonempty and normalized-distinct from q0/q1; otherwise stops. C deliberately reexecutes the same q1 for repeated-vote control.',
      'new_query_vs_new_title':'Record separately; novel query words do not establish a newly selected title.',
      'runtime':'CPU only, 2 threads, one dataset at a time, no new index, no training; common quality q0/q1 results reused across arms',
      'timing':'independent uncached fresh pipeline runs, first timing-n questions, two randomized repetitions; shared host, descriptive only',
      'comparisons':{'B-A':'same D<=400 cap; depth reallocation plus iteration policy, not isolated causal round count','B-C':'new-query policy versus repeated q1 votes; budgets differ if B skips q2','B-D':'incremental new q2 effect including its extra retrieval budget','C-D':'repeated query votes plus extra calls; negative control'},
      'source_files':{},'script_sha256':sha(__file__)}
    for ds in protocol['datasets']:
        for rel in [f'data/{ds}_confirm.json',f'restored/indexes/{ds}/index_manifest.json']:
            protocol['source_files'][rel]=sha(root/rel)
    for rel in ['v1/code/common.py','v1/code/frozen/run_retrieval.py','v1/code/frozen/query_expansion.py','v1/code/frozen/expansion_variants.py','v1/code/frozen/retrieval_backends.py']:
        protocol['source_files'][rel]=sha(root/rel)
    atomic(out/'protocol.json',protocol);atomic(out/'started.json',{'pid':os.getpid(),'protocol_sha256':sha(out/'protocol.json'),'started_unix':time.time()})
    summary={};alltiming=[];start=time.time()
    def progress(**kw):atomic(out/'status.json',{'updated_unix':time.time(),**kw})
    try:
      with gzip.open(out/'quality_rows.jsonl.gz','wt',compresslevel=3) as sink,gzip.open(out/'raw_calls.jsonl.gz','wt',compresslevel=3) as raw:
        for ds in protocol['datasets']:
          progress(stage='loading',dataset=ds)
          labels=json.loads((root/'data'/f'{ds}_confirm.json').read_text())['queries'][:args.n]
          assert len(labels)==args.n
          backend=legacy.RetrievalBackend(root/'restored/data'/ds/'corpus.jsonl',root/'restored/indexes'/ds,root/'models/bge',threads=2,device='cpu',cache_queries=False)
          exp=legacy.VariantExpander(backend.docs,policy='no_second_branch');alpha=legacy.ALPHAS[ds];backend.warmup()
          ev=core.Evaluator(backend.doc_lookup,labels);cache_rows=[]
          def call(text,depth,round_no):
            cs=[legacy.measured_call(backend,text,r,depth,f'q{round_no}',f'q{round_no}_{r}') for r in ['bm25','dense']]
            for c in cs:c['round']=round_no
            assert all(not c['cache_hit'] for c in cs)
            return cs
          def expansion(q,calls,used):
            p=exp.bridge_query(q['question'],legacy.fused_rank(calls,alpha),backend.doc_lookup)
            text=p.get('query');valid=bool(text) and legacy.qnorm(text) not in used
            return (text if valid else None),p
          def fresh(q,arm):
            calls=call(q['question'],100,0);q1,p=expansion(q,calls,{legacy.qnorm(q['question'])});hist=[]
            if q1:
              calls+=call(q1,100 if arm=='A_single100' else 50,1);hist.append(q1)
              if arm=='C_repeat50':calls+=call(q1,50,2);hist.append(q1)
              elif arm=='B_iterative50':
                q2,_=expansion(q,calls,{legacy.qnorm(q['question']),legacy.qnorm(q1)})
                if q2:calls+=call(q2,50,2);hist.append(q2)
            return legacy.fused_rank(calls,alpha),hist,calls
          for qi,label in enumerate(labels):
            q=legacy.public_question(label);c0=call(q['question'],100,0)
            q1,p1=expansion(q,c0,{legacy.qnorm(q['question'])});c100=[];c50=[];c2=[];crep=[];q2=None;p2={}
            if q1:
              c100=call(q1,100,1);c50=call(q1,50,1)
              assert all(a['returned'][:50]==b['returned'] for a,b in zip(c100,c50)), 'Depth-prefix inconsistency'
              q2,p2=expansion(q,c0+c50,{legacy.qnorm(q['question']),legacy.qnorm(q1)})
              if q2:c2=call(q2,50,2)
              crep=call(q1,50,2)
              assert all(a['returned']==b['returned'] for a,b in zip(c50,crep)), 'Repeated-query inconsistency'
            arms={'A_single100':c0+c100,'B_iterative50':c0+c50+c2,'C_repeat50':c0+c50+crep,'D_single50':c0+c50}
            targets1=[a['mentioned_title'] for a in p1.get('provenance',{}).get('entities',[])]
            targets2=[a['mentioned_title'] for a in p2.get('provenance',{}).get('entities',[])] if q2 else []
            raw.write(json.dumps({'dataset':ds,'id':q['id'],'q0':c0,'q1_100':c100,'q1_50':c50,'q2_new_50':c2,'q2_repeat_50':crep,'q1_provenance':p1,'q2_provenance':p2},ensure_ascii=False)+'\n')
            for arm,calls in arms.items():
              rank=legacy.fused_rank(calls,alpha);ids=list(dict.fromkeys(c['query_id'] for c in calls));hist=[c['query'] for c in calls[::2]][1:]
              perround=[]
              for rid in ids:
                prefix=[c for c in calls if int(c['query_id'][1:])<=int(rid[1:])];rr=legacy.fused_rank(prefix,alpha)
                perround.append({'round':int(rid[1:]),'pool':ev.score(rr,q['id'],len(rr)),'strict_by_K':{str(k):ev.score(rr,q['id'],k)['strict'] for k in range(1,51)}})
              row={'dataset':ds,'id':q['id'],'arm':arm,'queries':len(calls)//2,'supplementary':len(hist),'history':hist,'q1_targets':targets1,
                'q2_targets':targets2 if arm=='B_iterative50' else targets1 if arm=='C_repeat50' and q1 else [],
                'q2_new_targets':sorted(set(targets2)-set(targets1)) if arm=='B_iterative50' else [],
                'requested_records':sum(c['depth'] for c in calls),'returned_records':sum(len(c['returned']) for c in calls),'unique_candidate_count':len(rank),
                'q2_stop_reason':('new_query' if q2 else ('no_initial_query' if not q1 else 'empty_or_repeated_regeneration')) if arm=='B_iterative50' else None,
                'metrics':{str(k):ev.score(rank,q['id'],k) for k in range(1,51)},'pool':ev.score(rank,q['id'],len(rank)),'rounds':perround,'top50':rank[:50]}
              assert row['requested_records']<= (300 if arm=='D_single50' else 400)
              sink.write(json.dumps(row,ensure_ascii=False)+'\n');cache_rows.append(row)
            if (qi+1)%10==0:sink.flush();raw.flush();progress(stage='quality',dataset=ds,questions=qi+1,total=args.n,elapsed_seconds=time.time()-start)
          stats={}
          by={(r['id'],r['arm']):r for r in cache_rows}
          for arm in ARM_NAMES:
            rs=[r for r in cache_rows if r['arm']==arm]
            stats[arm]={'n':len(rs),'coverage':{str(k):sum(r['metrics'][str(k)]['strict'] for r in rs)/len(rs) for k in range(1,51)},
              'q2_triggered':sum(r['supplementary']==2 for r in rs),'q2_new_title_cases':sum(bool(r['q2_new_targets']) for r in rs),
              'supplementary_distribution':dict(collections.Counter(r['supplementary'] for r in rs)),
              'mean_requested_records':sum(r['requested_records'] for r in rs)/len(rs),'mean_total_queries':sum(r['queries'] for r in rs)/len(rs),
              'budget_distribution':dict(collections.Counter(r['requested_records'] for r in rs)),
              'pool_complete':sum(r['pool']['strict'] for r in rs)}
          comparisons={}
          for a,b in [('B_iterative50','A_single100'),('B_iterative50','C_repeat50'),('B_iterative50','D_single50'),('C_repeat50','D_single50')]:
            result={}
            for k in range(1,51):
              diffs=[by[q['id'],a]['metrics'][str(k)]['strict']-by[q['id'],b]['metrics'][str(k)]['strict'] for q in labels]
              result[str(k)]={'win':sum(x>0 for x in diffs),'loss':sum(x<0 for x in diffs),'tie':sum(x==0 for x in diffs),'delta_pp':100*sum(diffs)/len(diffs)}
            result['same_realized_budget_n']=sum(by[q['id'],a]['requested_records']==by[q['id'],b]['requested_records'] for q in labels)
            comparisons[a+'-minus-'+b]=result
          summary[ds]={'arms':stats,'comparisons':comparisons,'question_ids':[q['id'] for q in labels]};atomic(out/'summary.json',summary)
          progress(stage='matched_timing',dataset=ds)
          for rep in range(2):
            jobs=[(q,arm) for q in labels[:args.timing_n] for arm in ARM_NAMES];random.Random(2026110703+rep).shuffle(jobs)
            for label,arm in jobs:
              q=legacy.public_question(label);st=time.perf_counter();rank,hist,calls=fresh(q,arm);ms=(time.perf_counter()-st)*1000
              assert rank[:50]==by[q['id'],arm]['top50'] and hist==by[q['id'],arm]['history'],'Timing/quality trajectory mismatch'
              alltiming.append({'dataset':ds,'id':q['id'],'arm':arm,'repeat':rep,'elapsed_ms':ms,'logical_queries':len(calls)//2,'requested_records':sum(c['depth'] for c in calls)})
          atomic(out/'timing_rows.json',alltiming)
          del backend,exp,ev
      latency={}
      for ds in protocol['datasets']:
        latency[ds]={arm:{'n':len(x:=[r['elapsed_ms'] for r in alltiming if r['dataset']==ds and r['arm']==arm]),'p50_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'mean_ms':float(np.mean(x))} for arm in ARM_NAMES}
      atomic(out/'timing_summary.json',latency)
      atomic(out/'complete.json',{'status':'complete','wall_seconds':time.time()-start,'quality_rows':args.n*2*4,'timing_runs':len(alltiming),'protocol_sha256':sha(out/'protocol.json'),'summary_sha256':sha(out/'summary.json'),'quality_sha256':sha(out/'quality_rows.jsonl.gz'),'raw_calls_sha256':sha(out/'raw_calls.jsonl.gz')});progress(stage='complete',elapsed_seconds=time.time()-start)
    except BaseException:
      atomic(out/'failed.json',{'error':traceback.format_exc(),'time':time.time()});raise

if __name__=='__main__':main()
