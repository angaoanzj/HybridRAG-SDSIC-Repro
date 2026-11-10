"""Public-state feedback scoring and retrieval. Labels live only in Gold."""
from pathlib import Path
import sys,os,json,time,math,collections,hashlib,gzip
import numpy as np

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent/'frozen'))
from feedback import Feedback,ALL,RERANK,fuse_weighted
from query_expansion import words,STOPWORDS,WORD_PATTERN,containing_sentence
from retrieval_backends import RetrievalBackend,QUERY_PREFIX,tokenize,file_sha256,write_json
from run_retrieval import measured_call,fused_rank

OLD=Path('/root/autodl-tmp/hybridrag-feedback-optimization-20261107')
PAPER=Path('/root/autodl-tmp/hybridrag-paper-repair-20261105')
MU=Path('/root/autodl-tmp/hybridrag-musique-extension-20261107')
MODEL=PAPER/'model'
KS=(5,10,20,30,40,50)
BASE=ALL+tuple(p+'__q0w70' for p in RERANK)
SCORES=('bm25sent','bge','ce','bge_mmr','rm3mix')
NEW=tuple(f'mix_{s}{n}' for s in SCORES for n in (8,16))+('bm25_400','rm3_sparse8','rm3_sparse16')
LEARNED=('phrase_gain8','phrase_gain16')

def load(p):return json.loads(Path(p).read_text())
def progress(stage,**kw):
    o={'stage':stage,'unix':time.time(),**kw};write_json(R/'status.json',o);print(json.dumps(o),flush=True)
def inputs(ds):
    if ds.startswith('train_'):
        stem='design' if ds=='train_design' else 'check';p=OLD/'data/train_pool'
        return p/'corpus.jsonl',OLD/'indexes/train_pool',load(p/(stem+'_public.json')),load(p/(stem+'_labels.json')),.5,'paragraph'
    if ds=='musique':return MU/'data/musique/corpus.jsonl',MU/'indexes/musique',load(MU/'data/musique/public.json'),load(MU/'data/musique/labels.json'),.5,'paragraph'
    return PAPER/'data'/ds/'corpus.jsonl',PAPER/'indexes'/ds,load(OLD/'data'/('manuscript_'+ds+'_public.json')),load(OLD/'data'/('manuscript_'+ds+'_labels.json')),.3 if ds=='2wiki' else .7,'sentences'
def ref_name(ds):return 'manuscript_'+ds if ds in ('2wiki','hotpotqa') else ds

class Gold:
    def __init__(self,b,labels,mode):
        self.b=b;self.labels={r['id']:r for r in labels};self.mode=mode;self.cache={};self.bytitle=collections.defaultdict(list)
        for d in b.docs:self.bytitle[d['title']].append(d)
    def units(self,qid):
        if qid not in self.cache:
            q=self.labels[qid]
            if self.mode=='paragraph':self.cache[qid]=set(q['gold_doc_ids'])
            else:
                norm=lambda s:' '.join(s.split())
                self.cache[qid]={d['doc_id']:t for t in q['gold_titles'] for d in self.bytitle[t] if q['supporting_sentences_by_title'][t] and all(norm(s) in norm(d['text']) for s in q['supporting_sentences_by_title'][t])}
        return self.cache[qid]
    def score(self,rank,qid,k):
        unit=self.units(qid);ids=set(rank[:k]);q=self.labels[qid];titles=set(q['gold_titles'])
        if self.mode=='paragraph':hits=ids&unit;f=int(unit<=ids);rec=len(hits)/len(unit)
        else:hits={unit[i] for i in ids if i in unit};f=int(titles<=hits);rec=len(hits&titles)/len(titles)
        return {'complete':f,'recall':rec,'title_complete':int(titles<={self.b.doc_lookup[i]['title'] for i in ids}),'documents':len(ids),'words':sum(len(self.b.doc_lookup[i]['text'].split()) for i in ids)}
    def scores(self,rank,qid):return {str(k):self.score(rank,qid,k) for k in KS}
    def pool(self,rank,qid):return self.score(rank,qid,len(rank))['complete']

class FastBackend(RetrievalBackend):
    """Exact tie-stable top-D without sorting the whole corpus."""
    def top(self,scores,depth):
        depth=min(depth,len(scores));cut=np.partition(scores,len(scores)-depth)[len(scores)-depth]
        inds=np.flatnonzero(scores>=cut)
        order=np.lexsort((np.asarray(self.doc_ids,dtype=object)[inds],-scores[inds]))
        return inds[order[:depth]]
    def search(self,query,route,depth):
        st=time.perf_counter();enc=0.
        if route=='bm25':
            cols=sorted({self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary})
            scores=np.asarray(self.bm25[:,cols].sum(1)).ravel() if cols else np.zeros(len(self.docs),np.float32)
        else:
            self._load_model();t=time.perf_counter();vec=self.vector(query);enc=time.perf_counter()-t;scores=self.embeddings@vec
        inds=self.top(scores,depth)
        results=[{'doc_id':self.doc_ids[j],'rank':r,'raw_score':float(scores[j])} for r,j in enumerate(inds,1)]
        return {'results':results,'timing':{'query_encoding_seconds':enc,'search_seconds':time.perf_counter()-st-enc,'total_seconds':time.perf_counter()-st,'query_cache_hit':False,'query_cache_enabled':self.cache_queries}}
    def vector(self,q):
        self._load_model()
        if self.cache_queries and q in self._query_vectors:return self._query_vectors[q]
        v=np.asarray(self._model.encode([QUERY_PREFIX+q],batch_size=1,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0],np.float32)
        if self.cache_queries:self._query_vectors[q]=v
        return v
    def weighted(self,weights,depth=200):
        cols=[self.vocabulary[t] for t in sorted(weights) if t in self.vocabulary];v=np.asarray([weights[t] for t in sorted(weights) if t in self.vocabulary],np.float32)
        scores=np.asarray(self.bm25[:,cols]@v).ravel() if cols else np.zeros(len(self.docs),np.float32)
        inds=self.top(scores,depth)
        return [{'doc_id':self.doc_ids[i],'rank':r,'raw_score':float(scores[i])} for r,i in enumerate(inds,1)]

class Content(Feedback):
    def __init__(self,b):
        super().__init__(b);self.avg_sent=float(np.mean([len(words(s)) for d in b.docs for s in d.get('sentences',[]) if s.strip()]))
        self.corpus_df=np.asarray(b.bm25.getnnz(axis=0)).ravel();self.sentences={}
    def state(self,q,initial):
        assert isinstance(q,str)
        bridge=self.exp.bridge_query(q,initial,self.backend.doc_lookup);titles=[e['mentioned_title'] for e in bridge['provenance']['entities']]
        qt=set(words(q))-STOPWORDS;excluded=set(words(q))|STOPWORDS|{w for t in titles for w in words(t)};denom=sum(self.idf.get(w,0) for w in qt) or 1.;pool=[]
        for rank,i in enumerate(initial[:5],1):
            d=self.backend.doc_lookup[i];family=' '.join(words(d['title']));so=sum(self.idf.get(w,0) for w in qt&set(words(d['title'])))/denom
            matches=list(WORD_PATTERN.finditer(d['text']))[:180]
            for pos,m in enumerate(matches):
                t=m.group().lower()
                if t in excluded or len(t)<3 or t.isdigit():continue
                found=containing_sentence(d,m.start(),m.end())
                if found is None:continue
                sent=found[0];overlap=sum(self.idf.get(w,0) for w in qt&set(words(sent)))/denom
                tf=collections.Counter(words(sent));length=sum(tf.values());norm=1.5*(.25+.75*length/max(self.avg_sent,1.))
                bm=sum(self.idf.get(w,0)*2.5*tf[w]/(tf[w]+norm) for w in qt if tf[w])/(2.5*denom)
                pool.append({'token':t,'doc_id':i,'source_title':d['title'],'family':family,'rank':rank,'position':pos,'sentence':sent,'sentence_overlap':overlap,'source_overlap':so,'bm25sent':bm,'idf':self.idf.get(t,0),'score':self.idf.get(t,0)*(overlap+.25*so)/(60+rank)})
        return {'question':q,'titles':titles,'residual':bridge['provenance']['residual_keywords'],'bridge_query':bridge['query'],'pool':pool,'initial':initial[:200]}
    def prepare_scores(self,s,emb,ce,qvec):
        s['qvec']=qvec;s['emb']=emb;s['ce']=ce
        counts={}
        for rank,i in enumerate(s['initial'][:5],1):
            toks=words(self.backend.doc_lookup[i]['text'])[:180];tf=collections.Counter(toks);counts[rank]=(tf,len(toks))
        ws={r:1/(60+r) for r in counts};total=sum(ws.values()) or 1
        for x in s['pool']:
            x['bge']=max(0.,float(emb[x['sentence']]@qvec));x['ce']=float(ce[x['sentence']])
            x['rm3mix']=sum(ws[r]/total*tf[x['token']]/max(n,1) for r,(tf,n) in counts.items())
        return s
    def select(self,s,mode,n):
        items=[]
        for x in s['pool']:
            z=x.copy();z['selection_score']=x['rm3mix'] if mode=='rm3mix' else x['idf']*(x['bge' if mode=='bge_mmr' else mode]+.25*x['source_overlap'])/(60+x['rank']);items.append(z)
        items.sort(key=lambda x:(-x['selection_score'],x['rank'],x['position'],x['token']))
        selected=[];seen=set();counts=collections.Counter();maxscore=max([x['selection_score'] for x in items],default=1.) or 1.
        while items and len(selected)<n:
            eligible=[x for x in items if x['token'] not in seen and counts[x['family']]<4]
            if not eligible:break
            if mode=='bge_mmr':
                def val(x):
                    redundancy=max([max(0.,float(s['emb'][x['sentence']]@s['emb'][a['sentence']])) for a in selected],default=0.)
                    return .7*x['selection_score']/maxscore-.3*redundancy
                chosen=max(eligible,key=lambda x:(val(x),-x['rank'],-x['position'],x['token']))
                chosen={**chosen,'mmr_score':val(chosen)}
            else:chosen=eligible[0]
            selected.append(chosen);seen.add(chosen['token']);counts[chosen['family']]+=1
            items=[x for x in items if x['token']!=chosen['token']]
        return selected
    def query(self,s,chosen):
        parts=s['titles']+[x.get('phrase',x.get('token')) for x in chosen]+s['residual']
        return ' '.join(parts).strip()[:100000] if s['titles'] or chosen else None
    def sparse_rm3(self,q,bm25calls,n):
        docs=bm25calls[0]['returned'][:5];weights=np.asarray([x['raw_score'] for x in docs],np.float64);weights=np.maximum(weights,0);weights=weights/weights.sum() if weights.sum() else np.ones(len(docs))/len(docs)
        probs=collections.defaultdict(float)
        for weight,row in zip(weights,docs):
            tf=collections.Counter(t for t in words(self.backend.doc_lookup[row['doc_id']]['title']+'\n'+self.backend.doc_lookup[row['doc_id']]['text']) if t not in STOPWORDS and len(t)>=3 and not t.isdigit());norm=sum(tf.values()) or 1
            for t,v in tf.items():probs[t]+=weight*v/norm
        top=sorted(probs,key=lambda t:(-probs[t],t))[:n];norm=sum(probs[t] for t in top) or 1.;qtf=collections.Counter(t for t in words(q) if t not in STOPWORDS);qn=sum(qtf.values()) or 1
        model={t:.5*v/qn for t,v in qtf.items()}
        for t in top:model[t]=model.get(t,0)+.5*probs[t]/norm
        return model,{'feedback_terms':top,'feedback_probabilities':{t:probs[t]/norm for t in top},'original_query_weight':.5,'document_weights':weights.tolist()}
    def phrases(self,s):
        selected=[];seen=set();excluded={tuple(words(t)) for t in s['titles']}|{tuple(words(s['question']))};qkeys={x[2] for x in self.exp.title_mentions(words(s['question']))}
        for rank,i in enumerate(s['initial'][:5],1):
            d=self.backend.doc_lookup[i];matches=list(WORD_PATTERN.finditer(d['text']))[:180];ts=[m.group().lower() for m in matches]
            for start,end,key,title in self.exp.title_mentions(ts):
                if key in excluded or key in qkeys or key in seen or key==tuple(words(d['title'])):continue
                found=containing_sentence(d,matches[start].start(),matches[end-1].end())
                if found is None:continue
                seen.add(key);selected.append({'phrase':title,'token':title,'sentence':found[0],'rank':rank,'position':start,'family':' '.join(words(d['title'])),'source_title':d['title'],'doc_id':i,'kind':'title_phrase'})
                if len(selected)>=4:break
            if len(selected)>=4:break
        for x in self.select(s,'bge',8):
            if tuple(words(x['token'])) in seen:continue
            selected.append({**x,'phrase':x['token'],'kind':'word'});seen.add(tuple(words(x['token'])))
            if len(selected)>=8:break
        return selected
    def features(self,s,x,k):
        toks=words(x['phrase']);idf=[self.idf.get(w,0) for w in toks];qt=set(words(s['question']))-STOPWORDS;denom=sum(self.idf.get(w,0) for w in qt) or 1.;sent=x['sentence'];ov=sum(self.idf.get(w,0) for w in qt&set(words(sent)))/denom
        return [k/50,math.log1p(k),len(toks)/8,math.log1p(len(x['phrase'])),float(np.mean(idf)) if idf else 0.,max(idf,default=0.),1/x['rank'],1/(1+x['position']),ov,len(qt&set(toks))/max(len(qt),1),float(s['emb'][sent]@s['qvec']),float(s['ce'][sent]),float(x['kind']=='title_phrase'),len(s['titles'])/2,len(s['pool'])/500,len(words(s['question']))/40,len(set(toks)&set(words(x['source_title'])))/max(len(toks),1)]

def bounded_query(content,s,chosen,limit=64):
    query=content.query(s,chosen)
    if query is None:return None
    if len(query.split())>limit:query=' '.join(query.split()[:limit])
    return query

def calls_for(b,q,initial,query,alpha,memo):
    calls=list(initial)
    if query and query.strip()!=q.strip():
        for route in ['bm25','dense']:
            key=(query,route,100)
            if key not in memo:memo[key]=measured_call(b,query,route,100,'q1',route)
            calls.append(memo[key])
    assert len(calls)<=4 and sum(len(c['returned']) for c in calls)<=400
    return calls,fused_rank(calls,alpha)

def aggregate(rows):
    out={}
    for p in sorted({r['policy'] for r in rows}):
        rr=[r for r in rows if r['policy']==p]
        out[p]={'n':len(rr),'at_k':{str(k):{m:float(np.mean([r['at_k'][str(k)][m] for r in rr])) for m in ['complete','recall','title_complete','documents','words']} for k in KS},'pool_complete':float(np.mean([r['pool_complete'] for r in rr]))}
    return out
