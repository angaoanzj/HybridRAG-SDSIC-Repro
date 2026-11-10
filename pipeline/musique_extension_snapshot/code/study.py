"""Frozen single-expansion comparison. Labels enter score() only after retrieval."""
import os,sys,json,time,hashlib,gzip,random,collections,traceback
from pathlib import Path
import numpy as np
from retrieval_backends import RetrievalBackend,build_sparse_index,file_sha256,write_json
from run_retrieval import retrieve_group
from expansion_variants import VariantExpander
from order_variants import VariantExpander as OrderExpander
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path(os.environ['EXTENSION_MODEL'])
K=(5,10,20,50);POLICIES=('hybrid_rrf','one_target','first_two_valid_titles','no_second_branch','bridge_expand','evidence_feedback');SEED=2026110703

def status(stage,**kw):
 write_json(ROOT/'status.json',{'stage':stage,'unix':time.time(),**kw});print(json.dumps({'stage':stage,**kw}),flush=True)
def index(ds):
 import torch
 from sentence_transformers import SentenceTransformer
 out=ROOT/'indexes'/ds;out.mkdir(parents=True,exist_ok=True);cp=ROOT/'data'/ds/'corpus.jsonl'
 if (out/'index_manifest.json').exists():return
 docs=[json.loads(x) for x in cp.open()];torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model=SentenceTransformer(str(MODEL),device='cuda',local_files_only=True).float().eval();model.max_seq_length=512
 build_sparse_index(docs,out);write_json(out/'docids.json',[d['doc_id'] for d in docs]);v=np.lib.format.open_memmap(out/'embeddings.npy',mode='w+',dtype='float32',shape=(len(docs),768));start=time.time()
 for s in range(0,len(docs),512):
  batch=docs[s:s+512];v[s:s+len(batch)]=model.encode([d['title']+'\n'+d['text'] for d in batch],batch_size=64,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
  status('index',dataset=ds,documents=min(s+512,len(docs)),total=len(docs),seconds=time.time()-start)
 v.flush();assert np.isfinite(v).all();del v,model;torch.cuda.empty_cache()
 files={str(p.relative_to(MODEL)):{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in MODEL.rglob('*') if p.is_file() and p.name not in ['download_provenance.json']}
 write_json(out/'index_manifest.json',{'status':'complete','corpus_sha256':file_sha256(cp),'model_files':files,'max_sequence_length':512,'documents':len(docs),'device':'cuda offline only','dtype':'float32','tf32':False,'embeddings_sha256':file_sha256(out/'embeddings.npy')})

def score(ids,label,lookup):
 gold=set(label['gold_doc_ids']);gt=set(label['gold_titles']);positions={d:i for i,d in enumerate(ids,1)};ranks=[positions.get(d) for d in gold];min_k=max(ranks) if all(ranks) else None
 m={}
 for k in K:
  selected=set(ids[:k]);titles={lookup[d]['title'] for d in selected};m[str(k)]={'complete':int(gold<=selected),'recall':len(gold&selected)/len(gold),'title_complete':int(gt<=titles),'words':sum(len(lookup[d]['text'].split()) for d in selected),'documents':len(selected)}
 return {'at_k':m,'gold_ranks':{d:positions.get(d) for d in sorted(gold)},'min_k_complete':min_k}

def aggregate(rows):
 out={'n':len(rows),'expansion_rate':float(np.mean([bool(r['expanded_query']) for r in rows])),'mean_physical_calls':float(np.mean([r['physical_calls'] for r in rows])),'mean_returned_records':float(np.mean([r['returned_records'] for r in rows])),'mean_unique_pool':float(np.mean([r['unique_pool'] for r in rows])),'at_k':{}}
 for k in K:out['at_k'][str(k)]={key:float(np.mean([r['at_k'][str(k)][key] for r in rows])) for key in ['complete','recall','title_complete','words','documents']}
 return out

def evaluate(ds):
 backend=RetrievalBackend(ROOT/'data'/ds/'corpus.jsonl',ROOT/'indexes'/ds,MODEL,threads=8,device='cpu',cache_queries=True);backend.warmup()
 expanders={p:(OrderExpander(backend.docs,policy=p) if p=='first_two_valid_titles' else VariantExpander(backend.docs,policy=p if p!='hybrid_rrf' else 'no_second_branch')) for p in POLICIES}
 public=json.loads((ROOT/'data'/ds/'public.json').read_text());labels={q['id']:q for q in json.loads((ROOT/'data'/ds/'labels.json').read_text())};rows=[];start=time.time()
 # Reuse identical backend calls in quality evaluation. Latency is measured below
 # with this memo disabled and encoder caching disabled.
 real_search=backend.search;memo={}
 def cached(query,route,depth):
  key=(query,route,depth)
  if key not in memo:memo[key]=real_search(query,route,depth)
  return memo[key]
 backend.search=cached
 with gzip.open(ROOT/'rankings.jsonl.gz','wt',compresslevel=3) as f:
  for qi,q in enumerate(public):
   assert set(q)=={'id','question','split'}
   memo.clear();order=list(POLICIES);random.Random(SEED+qi).shuffle(order)
   for p in order:
    r=retrieve_group(backend,expanders[p],q,p,400,.5);ids=r['ranking'];label=labels[q['id']]
    row={'id':q['id'],'hops':label['hops'],'policy':p,'ranking':ids,'expanded_query':r['expanded_query'],'provenance':r['expansion_provenance'],'physical_calls':len(r['calls']),'returned_records':sum(len(c['returned']) for c in r['calls']),'unique_pool':len(set(d['doc_id'] for c in r['calls'] for d in c['returned'])),**score(ids,label,backend.doc_lookup)}
    rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
   if (qi+1)%50==0:f.flush();status('evaluate',dataset=ds,questions=qi+1,total=len(public),seconds=time.time()-start)
 summary={str(h):{p:aggregate([r for r in rows if r['policy']==p and (h=='all' or r['hops']==h)]) for p in POLICIES} for h in ['all',2,3,4]}
 pairs={};rng=np.random.default_rng(SEED)
 for h in ['all',2,3,4]:
  pairs[str(h)]={};by={p:{r['id']:r for r in rows if r['policy']==p and (h=='all' or r['hops']==h)} for p in POLICIES}
  for p in [x for x in POLICIES if x!='no_second_branch']:
   pairs[str(h)][p]={}
   for k in K:
    d=np.array([by['no_second_branch'][i]['at_k'][str(k)]['complete']-by[p][i]['at_k'][str(k)]['complete'] for i in by[p]])
    boot=np.mean(d[rng.integers(0,len(d),size=(5000,len(d)))],axis=1)
    pairs[str(h)][p][str(k)]={'delta_pp':float(d.mean()*100),'CI95_pp':(np.percentile(boot,[2.5,97.5])*100).tolist(),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'ties':int((d==0).sum())}
 write_json(ROOT/'quality_summary.json',{'summary':summary,'paired_score_two_minus':pairs,'alpha':.5,'alpha_selection':'fixed balanced route weight, no MuSiQue label tuning','timing_scope':'quality run memoizes identical retrieval calls; no latency claim','intervals':'question bootstrap, 5000 resamples, pointwise and unadjusted; compositional dependence not controlled'})
 backend.search=real_search;backend.cache_queries=False;backend._query_vectors.clear();timing=[];sample=random.Random(SEED).sample(public,30)
 for repeat in range(3):
  for qi,q in enumerate(sample):
   order=list(POLICIES);random.Random(SEED+repeat*100+qi).shuffle(order)
   for p in order:
    r=retrieve_group(backend,expanders[p],q,p,400,.5);timing.append({'id':q['id'],'policy':p,'repeat':repeat,'ms':r['total_seconds']*1000,'calls':len(r['calls'])})
  status('timing',repeat=repeat+1,questions=30)
 write_json(ROOT/'matched_timing.json',{'rows':timing,'summary':{p:{'n':len([r for r in timing if r['policy']==p]),'p50_ms':float(np.median([r['ms'] for r in timing if r['policy']==p])),'p95_ms':float(np.percentile([r['ms'] for r in timing if r['policy']==p],95))} for p in POLICIES},'scope':'CPU8, batch1, warm model, no query-vector cache or result memo; 30 fixed random questions x3; shared-node load not isolated'})
 write_json(ROOT/'complete.json',{'status':'complete','queries':len(public),'policies':list(POLICIES),'ranking_sha256':file_sha256(ROOT/'rankings.jsonl.gz'),'seconds':time.time()-start})
 status('complete',dataset=ds,questions=len(public),seconds=time.time()-start)

def main():
 try:
  from prepare_musique import main as prepare
  prepare();write_json(ROOT/'source_hashes.json',{p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')});index('musique');evaluate('musique')
 except Exception:
  status('failed',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
