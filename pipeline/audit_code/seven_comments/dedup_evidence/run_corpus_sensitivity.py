"""New post-review corpus sensitivity; frozen modules are imported read-only."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='8'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,gzip,hashlib,time,gc,shutil,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=Path('/root/autodl-tmp/hybridrag-reviewer-validation-20261106')
BASE=Path('/root/autodl-tmp/hybridrag-paper-repair-20261105')
sys.path.insert(0,str(OLD/'code'))
from retrieval_backends import RetrievalBackend,build_sparse_index
from expansion_variants import VariantExpander
from reviewer_extensions import retrieve_group
from run_retrieval import quota_selection
from reviewer_runner import execute_group,load_public_questions
import numpy as np
from scipy import sparse
MODEL=Path('/root/autodl-tmp/hybridrag-repro-20261104/project/data/hf_cache/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,r):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def main():
 protocol=json.loads((ROOT/'corpus_protocol.json').read_text());sig=sha(ROOT/'corpus_protocol.json')
 assert sha(Path(__file__))==protocol['runner_sha256']
 for name,h in protocol['frozen_code'].items():assert sha(OLD/'code'/name)==h
 for ds in ['2wiki','hotpotqa']:
  out=ROOT/ds;out.mkdir(exist_ok=True)
  assert sha(BASE/'data'/ds/'corpus.jsonl')==protocol['corpus_sha256'][ds]
  assert sha(OLD/'data_audit/public'/f'{ds}.json')==protocol['questions_sha256'][ds]
  b=RetrievalBackend(BASE/'data'/ds/'corpus.jsonl',BASE/'indexes'/ds,MODEL,threads=8,device='cpu',cache_queries=False)
  chosen={}
  for i,d in enumerate(b.docs):
   if d['title'] not in chosen or d['doc_id']<b.docs[chosen[d['title']]]['doc_id']:chosen[d['title']]=i
  indices=sorted(chosen.values());b.docs=[b.docs[i] for i in indices];b.doc_lookup={d['doc_id']:d for d in b.docs};b.doc_ids=[d['doc_id'] for d in b.docs];b.embeddings=np.asarray(b.embeddings[indices],dtype=np.float32);b._tie_order=np.argsort(np.asarray(b.doc_ids),kind='stable')
  build_sparse_index(b.docs,out);b.bm25=sparse.load_npz(out/'sparse_bm25.npz').tocsc();b.vocabulary=json.loads((out/'vocabulary.json').read_text());b.sparse_metadata=json.loads((out/'sparse_metadata.json').read_text())
  dump(out/'retained_doc_ids.json',b.doc_ids)
  expanders={p:VariantExpander(docs=b.docs,policy=p,source_top_n=5) for p in protocol['policies']}
  questions=load_public_questions(OLD/'data_audit/public'/f'{ds}.json');assert len(questions)==300
  b.warmup()
  for i,q in enumerate(questions):
   target=out/(q['id']+'.jsonl.gz')
   if target.exists():
    with gzip.open(target,'rt') as f:previous=[json.loads(l) for l in f]
    assert len(previous)==6 and all(r['sensitivity_protocol_sha256']==sig for r in previous)
    continue
   assert shutil.disk_usage(ROOT).free>4*1024**3,'Remote disk reserve'
   records=[]
   for p in protocol['policies']:
    ident={'dataset':ds,'id':q['id'],'split':q['split'],'mode':'confirmation','d_budget':400,'policy':p,'alpha':.3 if ds=='2wiki' else .7,'repeat':0,'source_top_n':5}
    r=execute_group(b,expanders[p],q,ident,retrieve_group,quota_selection);assert r['status']=='complete',r.get('error')
    r['sensitivity_protocol_sha256']=sig;r['study_scope']='post_review_corpus_sensitivity_reused_queries';records.append(r)
   tmp=target.with_suffix('.tmp')
   with gzip.open(tmp,'wt') as f:
    for r in records:f.write(json.dumps(r)+'\n')
   tmp.replace(target)
   if (i+1)%10==0:dump(ROOT/'status.json',{'status':'running','dataset':ds,'completed_questions_in_dataset':i+1,'protocol_sha256':sig});print(ds,i+1,flush=True)
  dump(out/'complete.json',{'questions':300,'groups':1800,'retained_documents':len(b.docs),'protocol_sha256':sig})
  del b,expanders;gc.collect()
 dump(ROOT/'status.json',{'status':'complete','questions':600,'groups':3600,'protocol_sha256':sig})
if __name__=='__main__':main()
