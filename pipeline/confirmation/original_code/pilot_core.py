"""QSEL-RANK-P1.0: public K-conditioned query selection and isolated reranking."""
import os,json,time,gzip,random,collections,hashlib,traceback,gc,math
from pathlib import Path
import numpy as np
from scipy.special import expit
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier as Clf,HistGradientBoostingRegressor as Reg
from retrieval_backends import RetrievalBackend,QUERY_PREFIX,write_json,file_sha256
from run_retrieval import fused_rank,measured_call
from feedback import Feedback,BASE,NEW,ALL,RERANK,fuse_weighted
from query_expansion import words,STOPWORDS
R=Path(__file__).resolve().parents[1];KS=[5,10,20,30,40,50];ROUTES=list(ALL[1:]);SEED=2026110706
ALPHA={'2wiki':.3,'hotpotqa':.7,'musique':.5}
def progress(stage,**kw):write_json(R/'status.json',{'stage':stage,'unix':time.time(),**kw});print(json.dumps({'stage':stage,**kw}),flush=True)
def public(q):return {'id':q['id'],'question':q['question'],'split':'P1'}
def top(s,n):
 n=min(n,len(s));t=np.partition(s,len(s)-n)[len(s)-n];ii=np.concatenate([np.flatnonzero(s>t),np.flatnonzero(s==t)[:n-int((s>t).sum())]]);return ii[np.lexsort((ii,-s[ii]))]
class Bulk:
 def __init__(self,b):
  import torch
  from sentence_transformers import SentenceTransformer
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  self.b=b;self.m=SentenceTransformer(str(R/'shared/model'),device='cuda',local_files_only=True).float().eval();self.m.max_seq_length=512;self.emb=torch.tensor(np.asarray(b.embeddings),device='cuda')
 def search(self,texts,qid):
  import torch
  out=[]
  for st in range(0,len(texts),64):
   chunk=texts[st:st+64];v=self.m.encode([QUERY_PREFIX+x for x in chunk],batch_size=64,normalize_embeddings=True,show_progress_bar=False,convert_to_tensor=True)
   with torch.inference_mode():scores=(v@self.emb.T).cpu().numpy()
   for text,score in zip(chunk,scores):
    bm=self.b.search(text,'bm25',100)['results'];ii=top(score,100);de=[{'doc_id':self.b.doc_ids[int(i)],'rank':j+1,'raw_score':float(score[i])} for j,i in enumerate(ii)]
    out.append([{'query_id':qid,'route':route,'query':text,'returned':found} for route,found in [('bm25',bm),('dense',de)]])
  return out
 def close(self):
  import torch
  del self.m,self.emb;torch.cuda.empty_cache()
class Features:
 def __init__(self,b):self.b=b;self.cache={}
 def profile(self,i):
  if i not in self.cache:
   d=self.b.doc_lookup[i];self.cache[i]=(set(words(d['title']))-STOPWORDS,set(words(d['text']))-STOPWORDS,[set(words(s))-STOPWORDS for s in d.get('sentences',[])[:12]],len(d['text'].split()))
  return self.cache[i]
 def query(self,q,initial,exp,p,k):
  assert set(q)=={'id','question','split'}
  qw=set(words(q['question']))-STOPWORDS;ew=set(words(exp or ''))-STOPWORDS;ids=fused_rank(initial,self.alpha)[:5];cover=[len(self.profile(i)[1]&qw)/max(1,len(qw)) for i in ids];sets=[{r['doc_id'] for r in c['returned'][:20]} for c in initial];titles={self.b.doc_lookup[i]['title'] for i in ids}
  return [float(p==z) for z in ROUTES]+[k/50,math.log1p(k)/4,len(words(q['question']))/60,len(ew)/60,len(ew-qw)/max(1,len(ew)),len(ew&qw)/max(1,len(qw)),sum(cover)/max(1,len(cover)),max(cover,default=0),len(titles)/5,len(sets[0]&sets[1])/max(1,len(sets[0]|sets[1])),float(bool(qw&{'both','same','different','earlier','older','longer'}))]
 def doc(self,q,calls,rank):
  qw=set(words(q['question']))-STOPWORDS;ew=set(words(calls[-1]['query']))-STOPWORDS;stats=collections.defaultdict(dict)
  for c in calls:
   for r in c['returned']:stats[r['doc_id']][c['query_id'],c['route']]=r
  out=[]
  for j,i in enumerate(rank):
   st=stats[i];t,b,ss,n=self.profile(i);rr=lambda qi,route:1/st[qi,route]['rank'] if (qi,route) in st else 0
   raw=lambda qi,route:st.get((qi,route),{}).get('raw_score',0)/(100 if route=='bm25' else 1)
   out.append([1/(j+1),rr('q0','bm25'),rr('q0','dense'),rr('q1','bm25'),rr('q1','dense'),raw('q0','bm25'),raw('q0','dense'),raw('q1','bm25'),raw('q1','dense'),len(st)/4,len(t&qw)/max(1,len(qw)),len(b&qw)/max(1,len(qw)),max((len(s&qw)/max(1,len(qw)) for s in ss),default=0),len(b&ew)/max(1,len(ew)),len(t)/10,math.log1p(n)/8])
  return np.asarray(out,np.float32)
class Gold:
 def __init__(self,b,ds):self.b=b;self.ds=ds;self.cache={};self.by=collections.defaultdict(list)
 def good(self,q):
  if q['id'] not in self.cache:
   if self.ds=='musique':g=set(q['gold_paragraph_ids'])
   else:
    if not self.by:
     for d in self.b.docs:self.by[d['title']].append(d)
    g={d['doc_id'] for t in q['gold_titles'] for d in self.by[t] if all(' '.join(s.split()) in ' '.join(d['text'].split()) for s in q['supporting_sentences_by_title'][t])}
   self.cache[q['id']]=g
  return self.cache[q['id']]
 def score(self,rank,q,k):
  good=self.good(q);gold=set(q['gold_paragraph_ids']) if self.ds=='musique' else set(q['gold_titles']);hit=set(rank[:k])&good
  if self.ds!='musique':hit={self.b.doc_lookup[i]['title'] for i in hit}
  return int(gold<=hit),len(gold&hit)/len(gold)
def predict(m,x,classification=True):
 raw=m._raw_predict(np.asarray(x,np.float32),n_threads=1)[:,0];return expit(raw) if classification else raw

def gather(ds,b,fb,feat,gold,split):
 checkpoint=R/'records'/f'{ds}_{split}.joblib';checkpoint.parent.mkdir(exist_ok=True)
 if checkpoint.exists():return joblib.load(checkpoint)
 qs=json.loads((R/'data'/f'{ds}_{split}.json').read_text());records=[];bulk=Bulk(b);start=time.time()
 for st in range(0,len(qs),32):
  batch=qs[st:st+32];initials=bulk.search([q['question'] for q in batch],'q0');plans=[];unique={}
  for q,calls in zip(batch,initials):
   rank=fused_rank(calls,ALPHA[ds]);exps={p:fb.expand(q['question'],rank,p).get('query') for p in ROUTES};plans.append(exps)
   for text in exps.values():
    if text and text.strip()!=q['question'].strip():unique[text]=None
  found=bulk.search(list(unique),'q1') if unique else [];unique.update(zip(unique,found))
  for q,initial,exps in zip(batch,initials,plans):
   metrics={};inputs={};docsets={};saved_ranks={}
   for p,query in exps.items():
    calls=initial+(unique[query] if query and query in unique else []);rank=fused_rank(calls,ALPHA[ds]);saved_ranks[p]=rank[:50];metrics[p]={str(k):gold.score(rank,q,k) for k in KS};inputs[p]=[feat.query(public(q),initial,query,p,k) for k in KS]
    if split=='train' and p in ['no_second_branch','evidence_feedback','mix_ranked16']:
     x=feat.doc(public(q),calls,rank);g=gold.good(q);pos=[j for j,i in enumerate(rank) if i in g][:12];neg=[j for j,i in enumerate(rank) if i not in g];sample=pos+neg[:12]+random.Random(SEED+st).sample(neg[12:],min(12,len(neg[12:])));docsets[p]=(x[sample],np.array([int(rank[j] in g) for j in sample]))
    elif split=='validation':docsets[p]=(rank,feat.doc(public(q),calls,rank))
   records.append({'q':q,'metrics':metrics,'inputs':inputs,'docs':docsets,'candidate_queries':exps,'initial_calls':initial,'saved_top50':saved_ranks})
  progress('collect_'+split,dataset=ds,questions=min(st+len(batch),len(qs)),total=len(qs),seconds=time.time()-start)
 bulk.close();joblib.dump(records,checkpoint,compress=3);return records

def arrays(records):
 x=[];y=[];r=[]
 for row in records:
  for p in ROUTES:
   for j,k in enumerate(KS):x.append(row['inputs'][p][j]);v=row['metrics'][p][str(k)];y.append(v[0]);r.append(v[1])
 return np.asarray(x,np.float32),np.array(y),np.array(r)
def choose(model,inputs,k):
 x=np.asarray([inputs[p][KS.index(k)] for p in ROUTES]);value=predict(model[0],x)+.02*np.clip(predict(model[1],x,False),0,1);return ROUTES[int(value.argmax())]
def blend(rank,x,ranker,weight):
 if not weight:return rank
 p=predict(ranker,x);rr=np.array([1/(60+j+1) for j in range(len(rank))]);rr/=rr.max();s=(1-weight)*rr+weight*p;ii=np.lexsort((np.arange(len(rank)),-s));return [rank[i] for i in ii]
def fit(ds,train,val,gold):
 x,y,r=arrays(train);out=R/'models'/ds;out.mkdir(parents=True,exist_ok=True);models=[];logs=[]
 for leaves in [7,15]:
  args=dict(max_iter=100,max_leaf_nodes=leaves,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=SEED)
  c=Clf(**args).fit(x,y);v=Reg(**args).fit(x,r);model=(c,v);scores=[row['metrics'][choose(model,row['inputs'],k)][str(k)][0] for row in val for k in KS];log={'leaves':leaves,'validation_macro_complete':float(np.mean(scores))};logs.append(log);models.append(model)
 best=max(range(2),key=lambda i:logs[i]['validation_macro_complete']);selector=models[best];joblib.dump(selector,out/'query_selector.joblib')
 dx=np.concatenate([a for row in train for a,y in row['docs'].values()]);dy=np.concatenate([y for row in train for a,y in row['docs'].values()]);ranker=Clf(max_iter=100,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=SEED).fit(dx,dy);joblib.dump(ranker,out/'document_ranker.joblib')
 wl=[]
 for weight in [0.,.25,.5]:
  scores=[]
  for row in val:
   rank,xx=row['docs']['mix_ranked16'];output=blend(rank,xx,ranker,weight);scores += [gold.score(output,row['q'],k)[0] for k in KS]
  wl.append({'weight':weight,'validation_macro_complete':float(np.mean(scores))})
 weight=max(wl,key=lambda a:a['validation_macro_complete'])['weight'];config={'version':'QSEL-RANK-P1.0','selector_candidates':logs,'selected':logs[best],'ranker_rows':len(dy),'ranker_positive_rows':int(dy.sum()),'rerank_candidates':wl,'rerank_weight':weight,'test_used_for_selection':False,'query_selector_sha256':file_sha256(out/'query_selector.joblib'),'ranker_sha256':file_sha256(out/'document_ranker.joblib')};write_json(out/'frozen.json',config);progress('models_frozen',dataset=ds,**config);return selector,ranker,weight

def evaluate(ds,b,fb,feat,gold,selector,ranker,weight):
 qs=json.loads((R/'data'/f'{ds}_test.json').read_text());rows=[];start=time.time();b.warmup();real=b.search;memo={}
 def cached(text,route,depth):
  key=text,route
  if key not in memo:memo[key]=real(text,route,200)
  z=memo[key];return {**z,'results':z['results'][:depth]}
 b.search=cached
 with gzip.open(R/'results'/f'{ds}_test.jsonl.gz','wt',compresslevel=2) as f:
  for qi,q in enumerate(qs):
   memo.clear();initial=[measured_call(b,q['question'],route,100,'q0',route) for route in ['bm25','dense']];initialrank=fused_rank(initial,ALPHA[ds]);exps={p:fb.expand(q['question'],initialrank,p).get('query') for p in ROUTES};inputs={p:[feat.query(public(q),initial,exps[p],p,k) for k in KS] for p in ROUTES};outputs={};calls_all={}
   for p,text in exps.items():
    calls=initial+([measured_call(b,text,route,100,'q1',route) for route in ['bm25','dense']] if text and text.strip()!=q['question'].strip() else []);calls_all[p]=calls;outputs[p]=fused_rank(calls,ALPHA[ds])
   rrf=[measured_call(b,q['question'],route,200,'q0',route) for route in ['bm25','dense']];outputs['hybrid_rrf']=fused_rank(rrf,ALPHA[ds]);calls_all['hybrid_rrf']=rrf
   for p in RERANK:outputs[p+'__q0w70']=fuse_weighted(calls_all[p],ALPHA[ds],.7)
   for k in KS:
    selected=choose(selector,inputs,k);fixed='mix_ranked16';rank_c=blend(outputs[fixed],feat.doc(public(q),calls_all[fixed],outputs[fixed]),ranker,weight);rank_d=blend(outputs[selected],feat.doc(public(q),calls_all[selected],outputs[selected]),ranker,weight)
    ranks={**outputs,'A_fixed_rules':outputs[fixed],'B_learned_query':outputs[selected],'C_learned_rank':rank_c,'D_combined':rank_d}
    for policy,rank in ranks.items():
     score,recall=gold.score(rank,q,k);record={'id':q['id'],'K':k,'policy':policy,'strict':score,'recall':recall,'selected_query_policy':selected if policy in ['B_learned_query','D_combined'] else None,'selected_ids':rank[:k]};rows.append(record);f.write(json.dumps(record)+'\n')
   if (qi+1)%20==0:progress('test_cpu_quality',dataset=ds,questions=qi+1,total=len(qs),seconds=time.time()-start)
 summary={p:{str(k):{'n':len(rr),'complete':float(np.mean([x['strict'] for x in rr])),'recall':float(np.mean([x['recall'] for x in rr]))} for k in KS if (rr:=[x for x in rows if x['policy']==p and x['K']==k])} for p in sorted({x['policy'] for x in rows})};write_json(R/'results'/f'{ds}_summary.json',summary)
 b.search=real;b.cache_queries=False;b._query_vectors.clear();timings=[]
 for q in qs[:12]:
  for k in [5,20,50]:
   for policy in ['A_fixed_rules','B_learned_query','C_learned_rank','D_combined']:
    st=time.perf_counter();initial=[measured_call(b,q['question'],route,100,'q0',route) for route in ['bm25','dense']];rank=fused_rank(initial,ALPHA[ds]);selected='mix_ranked16'
    if policy in ['B_learned_query','D_combined']:
     exps={p:fb.expand(q['question'],rank,p).get('query') for p in ROUTES};inputs={p:[feat.query(public(q),initial,exps[p],p,kk) for kk in KS] for p in ROUTES};selected=choose(selector,inputs,k);text=exps[selected]
    else:text=fb.expand(q['question'],rank,selected).get('query')
    calls=initial+([measured_call(b,text,route,100,'q1',route) for route in ['bm25','dense']] if text and text.strip()!=q['question'].strip() else []);rank=fused_rank(calls,ALPHA[ds])
    if policy in ['C_learned_rank','D_combined']:rank=blend(rank,feat.doc(public(q),calls,rank),ranker,weight)
    timings.append({'id':q['id'],'K':k,'policy':policy,'ms':(time.perf_counter()-st)*1000,'calls':len(calls),'D':sum(len(c['returned']) for c in calls)})
 write_json(R/'results'/f'{ds}_timing.json',{'rows':timings,'scope':'CPU8 warm uncached end-to-end; includes constructing all11 candidate queries and features for B/D;12 fixed heldout questions x K5/20/50; order fixed, descriptive timing only'})
 write_json(R/f'{ds}_complete.json',{'status':'complete','test_questions':len(qs),'test_records':len(rows),'models':2,'test_trace_sha256':file_sha256(R/'results'/f'{ds}_test.jsonl.gz')});progress('dataset_complete',dataset=ds)

def main(ds):
 import torch
 torch.set_num_threads(8);R.joinpath('results').mkdir(exist_ok=True);R.joinpath('models').mkdir(exist_ok=True);progress('starting',dataset=ds)
 cp=R/'work/data/corpus.jsonl';ip=R/'work/index';b=RetrievalBackend(cp,ip,R/'shared/model',threads=8,device='cpu',cache_queries=True);fb=Feedback(b);feat=Features(b);feat.alpha=ALPHA[ds];gold=Gold(b,ds)
 try:feat.query({'question':'x','gold_titles':['x']},[],None,ROUTES[0],5);raise RuntimeError('label-bearing input accepted')
 except AssertionError:pass
 train=gather(ds,b,fb,feat,gold,'train');val=gather(ds,b,fb,feat,gold,'validation');selector,ranker,w=fit(ds,train,val,gold);del train,val;gc.collect();evaluate(ds,b,fb,feat,gold,selector,ranker,w)
if __name__=='__main__':
 try:main(os.sys.argv[1])
 except Exception:write_json(R/'failed.json',{'dataset':os.sys.argv[1],'traceback':traceback.format_exc()});progress('failed',error=traceback.format_exc());raise
