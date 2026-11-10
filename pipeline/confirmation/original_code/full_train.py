"""QSEL-RANK-P1.1-FULL: all questions except300 validation, bounded shards."""
import os,sys,json,time,random,hashlib,gc,collections,traceback,gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier as Clf,HistGradientBoostingRegressor as Reg
import pilot_core as c
R=Path(__file__).resolve().parents[1];V=Path('/root/autodl-tmp/hybridrag-adaptive-v4-20261107');M=Path('/root/autodl-tmp/hybridrag-multiround-v2-20261107/models/bge');SEED=2026110711;KS=c.KS;ROUTES=c.ROUTES
TRAIN_ROUTES=['no_second_branch','evidence_feedback','mix_ranked16'];VERSION='QSEL-RANK-P1.1-FULL'
def progress(ds,stage,**kw):
    x={'version':VERSION,'dataset':ds,'stage':stage,'unix':time.time(),**kw};c.write_json(R/f'{ds}_status.json',x);print(json.dumps(x),flush=True)
class FastBulk:
    def __init__(self,b):
        import torch
        from sentence_transformers import SentenceTransformer
        assert b.doc_ids==sorted(b.doc_ids)
        for n,d in b._manifest['model_files'].items():assert c.file_sha256(M/n)==d['sha256']
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.b=b;self.model=SentenceTransformer(str(M),device='cuda',local_files_only=True).float().eval();self.model.max_seq_length=512;self.emb=torch.tensor(np.asarray(b.embeddings),device='cuda');self.pool=ThreadPoolExecutor(max_workers=2);self.audited=False
    def search(self,texts,qid):
        import torch
        out=[]
        for st in range(0,len(texts),64):
            chunk=texts[st:st+64];futures=[self.pool.submit(self.b.search,t,'bm25',100) for t in chunk];v=self.model.encode([c.QUERY_PREFIX+t for t in chunk],batch_size=64,normalize_embeddings=True,show_progress_bar=False,convert_to_tensor=True)
            with torch.inference_mode():scores=v@self.emb.T;val,ind=torch.topk(scores,min(200,len(self.b.doc_ids)),dim=1)
            vv=val.cpu().numpy();ii=ind.cpu().numpy()
            for row,text in enumerate(chunk):
                order=np.lexsort((ii[row],-vv[row]));picked=ii[row,order[:100]];pv=vv[row,order[:100]]
                if len(self.b.doc_ids)>200 and pv[-1]==vv[row].min():
                    a=torch.nonzero(scores[row]>=float(pv[-1])).ravel();av=scores[row,a].cpu().numpy();ai=a.cpu().numpy();order=np.lexsort((ai,-av))[:100];picked=ai[order];pv=av[order]
                if not self.audited:
                    assert np.array_equal(picked,c.top(scores[row].cpu().numpy(),100));self.audited=True
                dense=[{'doc_id':self.b.doc_ids[int(i)],'rank':j+1,'raw_score':float(s)} for j,(i,s) in enumerate(zip(picked,pv))];bm=futures[row].result()['results'];out.append([{'query_id':qid,'route':route,'query':text,'returned':found} for route,found in [('bm25',bm),('dense',dense)]])
            del scores,v,val,ind
        return out
    def close(self):
        import torch
        self.pool.shutdown();del self.model,self.emb;torch.cuda.empty_cache()
def inputs(feat,q,initial,exps):
    out={}
    for p in ROUTES:
        base=feat.query(c.public(q),initial,exps[p],p,5);out[p]=[]
        for k in KS:
            v=base.copy();v[len(ROUTES)]=k/50;v[len(ROUTES)+1]=c.math.log1p(k)/4;out[p].append(v)
    return out
def plan_batch(qs,bulk,fb,alpha):
    initial=bulk.search([q['question'] for q in qs],'q0');plans=[];unique={}
    for q,calls in zip(qs,initial):
        ranking=c.fused_rank(calls,alpha);exps={p:fb.expand(q['question'],ranking,p).get('query') for p in ROUTES};plans.append(exps)
        for text in exps.values():
            if text and text.strip()!=q['question'].strip():unique[text]=None
    found=bulk.search(list(unique),'q1') if unique else [];unique.update(zip(unique,found));return initial,plans,unique
def collect_train(ds,qs,b,bulk,fb,feat,gold,preflight=False):
    where=R/'supervision'/ds;where.mkdir(parents=True,exist_ok=True);start=time.time();written=[];total_q=0;doc_rows=0;audited=False
    for st in range(0,len(qs),256):
        target=where/f'train_{st:06d}.npz';receipt=target.with_suffix('.json');batch=qs[st:st+256]
        if receipt.exists():
            meta=json.loads(receipt.read_text());assert meta['ids']==[q['id'] for q in batch] and c.file_sha256(target)==meta['sha256'];written.append(str(target));total_q+=len(batch);doc_rows+=meta['document_rows'];continue
        xx=[];yy=[];rec=[];dx=[];dy=[]
        for sb in range(0,len(batch),64):
            small=batch[sb:sb+64];initials,plans,unique=plan_batch(small,bulk,fb,c.ALPHA[ds])
            for q,initial,exps in zip(small,initials,plans):
                qinputs=inputs(feat,q,initial,exps);g=gold.good(q)
                for p in ROUTES:
                    text=exps[p];calls=initial+(unique[text] if text and text in unique else []);rank=c.fused_rank(calls,c.ALPHA[ds])
                    for j,k in enumerate(KS):
                        score,recall=gold.score(rank,q,k);xx.append(qinputs[p][j]);yy.append(score);rec.append(recall)
                    if p in TRAIN_ROUTES:
                        pos=[j for j,i in enumerate(rank) if i in g][:12];neg=[j for j,i in enumerate(rank) if i not in g];seed=SEED+int(hashlib.sha256((q['id']+p).encode()).hexdigest()[:8],16);sample=pos+neg[:12]+random.Random(seed).sample(neg[12:],min(12,len(neg[12:])));chosen=[rank[j] for j in sample];a=feat.doc(c.public(q),calls,chosen);a[:,0]=[1/(j+1) for j in sample];labels=np.array([int(i in g) for i in chosen],np.uint8)
                        if not audited:
                            assert np.array_equal(a,feat.doc(c.public(q),calls,rank)[sample]);assert all(np.array_equal(np.asarray(qinputs[p][j]),np.asarray(feat.query(c.public(q),initial,text,p,k))) for j,k in enumerate(KS));audited=True
                        dx.append(a);dy.append(labels)
        x=np.asarray(xx,np.float32);y=np.asarray(yy,np.uint8);z=np.asarray(rec,np.float32);d=np.concatenate(dx);e=np.concatenate(dy);assert len(x)==len(batch)*11*6
        if preflight:return {'questions':len(batch),'query_rows':len(x),'document_rows':len(e),'optimized_features_exact':audited,'gpu_topk_exact':bulk.audited}
        with target.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,x=x,y=y,recall=z,dx=d,dy=e)
        target.with_suffix('.tmp').replace(target);meta={'ids':[q['id'] for q in batch],'questions':len(batch),'query_rows':len(x),'document_rows':len(e),'sha256':c.file_sha256(target),'optimized_feature_equivalence_checked':audited};c.write_json(receipt,meta);written.append(str(target));total_q+=len(batch);doc_rows+=len(e);progress(ds,'collect_all_training',questions=total_q,total=len(qs),query_rows=total_q*66,document_rows=doc_rows,seconds=time.time()-start)
        del xx,yy,rec,dx,dy,x,y,z,d,e;gc.collect()
    meta={'questions':total_q,'query_rows':total_q*66,'document_rows':doc_rows,'shards':len(written),'feature_optimization':'same features; reuse K-invariant fields; compute sampled document rows with original rank restored','all_training_questions_used':total_q==len(qs)};c.write_json(where/'training_complete.json',meta);return meta
def validation_records(ds,qs,bulk,fb,feat,gold):
    p=R/'supervision'/ds/'validation.joblib'
    if p.exists():return joblib.load(p)
    rows=[];start=time.time()
    for st in range(0,len(qs),64):
        small=qs[st:st+64];initials,plans,unique=plan_batch(small,bulk,fb,c.ALPHA[ds])
        for q,initial,exps in zip(small,initials,plans):
            metrics={};docsets={};qinputs=inputs(feat,q,initial,exps)
            for p,query in exps.items():
                calls=initial+(unique[query] if query and query in unique else []);rank=c.fused_rank(calls,c.ALPHA[ds]);metrics[p]={str(k):gold.score(rank,q,k) for k in KS};docsets[p]=(rank,feat.doc(c.public(q),calls,rank))
            rows.append({'q':q,'inputs':qinputs,'metrics':metrics,'docs':docsets})
        progress(ds,'collect_validation',questions=len(rows),total=len(qs),seconds=time.time()-start)
    joblib.dump(rows,p,compress=3);return rows
def fit(ds,val,gold):
    out=R/'models'/ds;out.mkdir(parents=True,exist_ok=True);frozen=out/'frozen.json'
    if frozen.exists():
        cfg=json.loads(frozen.read_text());assert c.file_sha256(out/'query_selector.joblib')==cfg['query_selector_sha256'] and c.file_sha256(out/'document_ranker.joblib')==cfg['ranker_sha256'];return joblib.load(out/'query_selector.joblib'),joblib.load(out/'document_ranker.joblib'),cfg['rerank_weight']
    files=sorted((R/'supervision'/ds).glob('train_*.npz'));pieces=[np.load(p) for p in files];x=np.concatenate([a['x'] for a in pieces]);y=np.concatenate([a['y'] for a in pieces]);rec=np.concatenate([a['recall'] for a in pieces]);logs=[];models=[]
    for leaves in [7,15]:
        progress(ds,'fit_query_selector',leaves=leaves,training_rows=len(y));args=dict(max_iter=100,max_leaf_nodes=leaves,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=SEED);model=(Clf(**args).fit(x,y),Reg(**args).fit(x,rec));score=float(np.mean([row['metrics'][c.choose(model,row['inputs'],k)][str(k)][0] for row in val for k in KS]));logs.append({'leaves':leaves,'validation_macro_complete':score});models.append(model)
    chosen=max(range(len(logs)),key=lambda i:logs[i]['validation_macro_complete']);selector=models[chosen];joblib.dump(selector,out/'query_selector.joblib');del x,y,rec,models;gc.collect();dx=np.concatenate([a['dx'] for a in pieces]);dy=np.concatenate([a['dy'] for a in pieces]);progress(ds,'fit_document_ranker',training_rows=len(dy),positives=int(dy.sum()));ranker=Clf(max_iter=100,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=SEED).fit(dx,dy);joblib.dump(ranker,out/'document_ranker.joblib');wl=[]
    for w in [0.,.25,.5]:
        score=float(np.mean([gold.score(c.blend(row['docs']['mix_ranked16'][0],row['docs']['mix_ranked16'][1],ranker,w),row['q'],k)[0] for row in val for k in KS]));wl.append({'weight':w,'validation_macro_complete':score})
    weight=max(wl,key=lambda a:a['validation_macro_complete'])['weight'];cfg={'version':VERSION,'selector_candidates':logs,'selected':logs[chosen],'ranker_rows':len(dy),'ranker_positive_rows':int(dy.sum()),'rerank_candidates':wl,'rerank_weight':weight,'validation_n':len(val),'test_n':0,'evaluation_scope':'validation selected; not independent test','query_selector_sha256':c.file_sha256(out/'query_selector.joblib'),'ranker_sha256':c.file_sha256(out/'document_ranker.joblib')};c.write_json(frozen,cfg);progress(ds,'models_frozen',**{k:v for k,v in cfg.items() if k!='version'});del dx,dy,pieces;gc.collect();return selector,ranker,weight
def evaluate_validation(ds,qs,b,fb,feat,gold,selector,ranker,weight):
    out=R/'results';out.mkdir(exist_ok=True);rows=[];start=time.time();b.warmup();real=b.search;memo={}
    def cached(text,route,depth):
        key=text,route
        if key not in memo:memo[key]=real(text,route,200)
        z=memo[key];return {**z,'results':z['results'][:depth]}
    b.search=cached
    with gzip.open(out/f'{ds}_validation.jsonl.gz','wt',compresslevel=3) as f:
        for qi,q in enumerate(qs):
            memo.clear();initial=[c.measured_call(b,q['question'],route,100,'q0',route) for route in ['bm25','dense']];initialrank=c.fused_rank(initial,c.ALPHA[ds]);exps={p:fb.expand(q['question'],initialrank,p).get('query') for p in ROUTES};qinputs=inputs(feat,q,initial,exps);outputs={};callsets={}
            for p,text in exps.items():
                calls=initial+([c.measured_call(b,text,route,100,'q1',route) for route in ['bm25','dense']] if text and text.strip()!=q['question'].strip() else []);callsets[p]=calls;outputs[p]=c.fused_rank(calls,c.ALPHA[ds])
            outputs['hybrid_rrf']=c.fused_rank([c.measured_call(b,q['question'],route,200,'q0',route) for route in ['bm25','dense']],c.ALPHA[ds])
            for p in c.RERANK:outputs[p+'__q0w70']=c.fuse_weighted(callsets[p],c.ALPHA[ds],.7)
            for k in KS:
                selected=c.choose(selector,qinputs,k);fixed='mix_ranked16';rc=c.blend(outputs[fixed],feat.doc(c.public(q),callsets[fixed],outputs[fixed]),ranker,weight);rd=c.blend(outputs[selected],feat.doc(c.public(q),callsets[selected],outputs[selected]),ranker,weight);ranks={**outputs,'A_fixed_rules':outputs[fixed],'B_learned_query':outputs[selected],'C_learned_rank':rc,'D_combined':rd}
                for p,rank in ranks.items():
                    strict,rec=gold.score(rank,q,k);docs=[b.doc_lookup[i] for i in rank[:k]];record={'id':q['id'],'K':k,'policy':p,'strict':strict,'recall':rec,'selected_query_policy':selected if p in ['B_learned_query','D_combined'] else None,'selected_ids':rank[:k],'body_words':sum(len(d['text'].split()) for d in docs),'text_characters':sum(len(d['title'])+1+len(d['text']) for d in docs),'scope':'validation selected'};rows.append(record);f.write(json.dumps(record)+'\n')
            if (qi+1)%25==0:progress(ds,'cpu_validation',questions=qi+1,total=len(qs),seconds=time.time()-start)
    summary={p:{str(k):{'n':len(rr),'complete':float(np.mean([x['strict'] for x in rr])),'recall':float(np.mean([x['recall'] for x in rr])),'mean_body_words':float(np.mean([x['body_words'] for x in rr]))} for k in KS if (rr:=[x for x in rows if x['policy']==p and x['K']==k])} for p in sorted({x['policy'] for x in rows})};c.write_json(out/f'{ds}_summary.json',summary);b.search=real;b.cache_queries=False;b._query_vectors.clear();timings=[]
    for q in qs[:12]:
        for k in [5,20,50]:
            for p in ['A_fixed_rules','B_learned_query','C_learned_rank','D_combined']:
                st=time.perf_counter();initial=[c.measured_call(b,q['question'],route,100,'q0',route) for route in ['bm25','dense']];rank=c.fused_rank(initial,c.ALPHA[ds]);selected='mix_ranked16'
                if p in ['B_learned_query','D_combined']:
                    exps={route:fb.expand(q['question'],rank,route).get('query') for route in ROUTES};selected=c.choose(selector,inputs(feat,q,initial,exps),k);text=exps[selected]
                else:text=fb.expand(q['question'],rank,selected).get('query')
                calls=initial+([c.measured_call(b,text,route,100,'q1',route) for route in ['bm25','dense']] if text and text.strip()!=q['question'].strip() else []);rank=c.fused_rank(calls,c.ALPHA[ds])
                if p in ['C_learned_rank','D_combined']:rank=c.blend(rank,feat.doc(c.public(q),calls,rank),ranker,weight)
                timings.append({'id':q['id'],'K':k,'policy':p,'ms':1000*(time.perf_counter()-st),'calls':len(calls)})
    c.write_json(out/f'{ds}_timing.json',{'scope':'CPU3 threads per worker, multiple corpora may run concurrently; warm model/document features; no query vector cache; fixed order12 validation questions x K5/20/50; descriptive only, not comparable to isolated P1 CPU8 timing','rows':timings});return len(rows)
def main(ds,preflight=False):
    import torch
    torch.set_num_threads(3);cfg=json.loads((R/'protocol.json').read_text());d=cfg['datasets'][ds];source=V/'data'/ds/'questions.json';assert c.file_sha256(source)==d['source_questions_sha256'];by={q['id']:q for q in json.loads(source.read_text())['queries']};train=[by[i] for i in d['ids']['train']];val=[by[i] for i in d['ids']['validation']];assert len(train)==d['train_n'] and len(val)==300 and not set(d['ids']['train'])&set(d['ids']['validation']);b=c.RetrievalBackend(V/'data'/ds/'corpus.jsonl',V/'indexes'/ds,M,threads=3,device='cpu',cache_queries=True);assert b._manifest['corpus_sha256']==d['corpus_sha256'];fb=c.Feedback(b);feat=c.Features(b);feat.alpha=c.ALPHA[ds];gold=c.Gold(b,ds);bulk=FastBulk(b);progress(ds,'starting',train_questions=len(train),validation_questions=300)
    try:feat.query({'question':'x','gold_titles':['x']},[],None,ROUTES[0],5);raise RuntimeError('gold-bearing input accepted')
    except AssertionError:pass
    if preflight:
        result=collect_train(ds,train[:2],b,bulk,fb,feat,gold,preflight=True);bulk.close();c.write_json(R/f'{ds}_preflight.json',result);print(json.dumps(result));return
    meta=collect_train(ds,train,b,bulk,fb,feat,gold);vr=validation_records(ds,val,bulk,fb,feat,gold);bulk.close();selector,ranker,w=fit(ds,vr,gold);del vr;gc.collect();n=evaluate_validation(ds,val,b,fb,feat,gold,selector,ranker,w);assert n==36000;c.write_json(R/f'{ds}_complete.json',{'version':VERSION,'status':'complete','train_questions':len(train),'validation_questions':300,'test_questions':0,'validation_records':n,'training':meta,'trace_sha256':c.file_sha256(R/'results'/f'{ds}_validation.jsonl.gz'),'scope':'validation selected; no independent test'});progress(ds,'complete',train_questions=len(train),validation_questions=300)
if __name__=='__main__':
    ds=sys.argv[1]
    try:main(ds,'--preflight' in sys.argv)
    except Exception:c.write_json(R/f'{ds}_failed.json',{'traceback':traceback.format_exc()});progress(ds,'failed',error=traceback.format_exc());raise
