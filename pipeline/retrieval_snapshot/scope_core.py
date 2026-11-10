"""Fixed-title feedback scope study. No labels enter candidate selection."""
from pathlib import Path
import sys,os,json,hashlib,collections,time
import numpy as np
from window_rules import positions

ROOT=Path(__file__).resolve().parent
LEGACY=Path(os.environ.get('FBSCOPE_R1_ROOT','/root/autodl-tmp/hybridrag-feedback-relevance-r1-20261107'))
sys.path.insert(0,str(LEGACY/'code'))
import core as c

MS=(3,5,10,20)
WINDOWS=('head180','head180_tail180','tail180','full')
SCORES=('idf','rm')
CAPS=(8,16)
CONDITIONS=tuple((m,w,score,n) for m in MS for w in WINDOWS for score in SCORES for n in CAPS)
KS=(2,5,10,15,20,30,40,50)

def key(m,w,score,n):return f'{score}_m{m}_{w}_w{n}'
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def sha(path):return c.file_sha256(path)
def load(path):return json.loads(Path(path).read_text())
def write(path,value):c.write_json(path,value)
def progress(stage,**values):
    value={'stage':stage,'unix':time.time(),**values};write(ROOT/'status.json',value);print(json.dumps(value),flush=True)

def data_inputs(dataset,role,with_labels=True):
    spec=load(ROOT/'data_request.json')['datasets'][dataset]
    all_public=load(spec['source_public']);public={x['id']:x for x in all_public}
    ids=spec['calibration_and_timing'] if role=='calibration' else spec['effectiveness']
    selected=[{'id':i,'question':public[i]['question'],'split':role} for i in ids]
    labels=load(spec['source_labels']) if with_labels else []
    if with_labels:
        by={x['id']:x for x in labels};labels=[by[i] for i in ids]
    if dataset=='musique':cp=c.MU/'data/musique/corpus.jsonl';ip=c.MU/'indexes/musique';mode='paragraph'
    else:cp=c.PAPER/'data'/dataset/'corpus.jsonl';ip=c.PAPER/'indexes'/dataset;mode='sentences'
    return cp,ip,selected,labels,mode

class ScopeSelector:
    def __init__(self,backend,feedback):
        self.b=backend;self.fb=feedback;self.profiles={}

    def profile(self,doc_id):
        if doc_id in self.profiles:return self.profiles[doc_id]
        d=self.b.doc_lookup[doc_id];text=d['text'];matches=list(c.WORD_PATTERN.finditer(text));tokens=[m.group().lower() for m in matches]
        spans=[];cursor=0
        for sentence in d.get('sentences',[]):
            stripped=sentence.strip()
            if not stripped:continue
            start=text.find(stripped,cursor)
            if start<0:continue
            cursor=start+len(stripped);spans.append((start,cursor,sentence))
        sentence_ids=[];j=0
        for match in matches:
            while j<len(spans) and spans[j][1]<match.end():j+=1
            sentence_ids.append(j if j<len(spans) and spans[j][0]<=match.start() and match.end()<=spans[j][1] else None)
        p={'tokens':tokens,'sentence_ids':sentence_ids,'spans':spans,'family':' '.join(c.words(d['title'])),'title_tokens':set(c.words(d['title'])),'windows':{}}
        self.profiles[doc_id]=p;return p

    def window(self,doc_id,window):
        p=self.profile(doc_id)
        if window not in p['windows']:
            length=len(p['tokens']);equivalent=next((v for w,v in p['windows'].items() if length<=180 or (length<=360 and {w,window}<={'head180_tail180','full'})),None)
            if equivalent is not None:p['windows'][window]=equivalent
            else:
                ii=positions(length,window);p['windows'][window]={'positions':ii,'tf':collections.Counter(p['tokens'][i] for i in ii),'length':len(ii)}
        return p,p['windows'][window]

    def context(self,s):
        question=s['question'];qt=set(c.words(question))-c.STOPWORDS
        excluded=set(c.words(question))|c.STOPWORDS|{t for title in s['titles'] for t in c.words(title)}
        # Existing public-state statistics preserve original default arithmetic.
        sentence_overlap={x['sentence']:x['sentence_overlap'] for x in s.get('pool',[])}
        source_overlap={x['doc_id']:x['source_overlap'] for x in s.get('pool',[])}
        return {'s':s,'qt':qt,'excluded':excluded,'denom':sum(self.fb.idf.get(w,0) for w in qt) or 1.,'sentence_overlap':sentence_overlap,'source_overlap':source_overlap,'pools':{}}

    def pool(self,ctx,m,window):
        cache_key=(m,window)
        if cache_key in ctx['pools']:return ctx['pools'][cache_key]
        s=ctx['s'];items=[];counts={};token_positions=0;families=set()
        old={(x['rank'],x['position']):x for x in s.get('pool',[])}
        for rank,doc_id in enumerate(s['initial'][:m],1):
            p,w=self.window(doc_id,window);counts[rank]=(w['tf'],w['length']);token_positions+=w['length'];families.add(p['family'])
            if doc_id not in ctx['source_overlap']:
                ctx['source_overlap'][doc_id]=sum(self.fb.idf.get(t,0) for t in ctx['qt']&p['title_tokens'])/ctx['denom']
            for pos in w['positions']:
                token=p['tokens'][pos];sid=p['sentence_ids'][pos]
                if token in ctx['excluded'] or len(token)<3 or token.isdigit() or sid is None:continue
                sentence=p['spans'][sid][2]
                if sentence not in ctx['sentence_overlap']:
                    ctx['sentence_overlap'][sentence]=sum(self.fb.idf.get(t,0) for t in ctx['qt']&set(c.words(sentence)))/ctx['denom']
                previous=old.get((rank,pos))
                score=previous['score'] if previous is not None else self.fb.idf.get(token,0)*(ctx['sentence_overlap'][sentence]+.25*ctx['source_overlap'][doc_id])/(60+rank)
                items.append({'token':token,'rank':rank,'position':pos,'family':p['family'],'idf_score':score})
        weights={rank:1/(60+rank) for rank in counts};total=sum(weights.values()) or 1.
        probs={token:sum(weights[rank]/total*tf[token]/max(n,1) for rank,(tf,n) in counts.items()) for token in {x['token'] for x in items}}
        for x in items:x['rm_score']=probs[x['token']]
        meta={'source_documents':len(counts),'source_families':len(families),'candidate_positions':token_positions,'eligible_occurrences':len(items),'eligible_token_types':len(probs)}
        ctx['pools'][cache_key]=(items,meta);return items,meta

    def choose(self,ctx,m,window,score,n):
        pool,meta=self.pool(ctx,m,window);field='idf_score' if score=='idf' else 'rm_score'
        ranked=sorted(pool,key=lambda x:(-x[field],x['rank'],x['position'],x['token']));selected=[];seen=set();family=collections.Counter()
        for x in ranked:
            if x['token'] in seen or family[x['family']]>=4:continue
            selected.append([x['token'],x['rank'],x['position'],float(x[field])]);seen.add(x['token']);family[x['family']]+=1
            if len(selected)==n:break
        s=ctx['s'];query=' '.join(s['titles']+[x[0] for x in selected]+s['residual']) if s['titles'] or selected else None
        if query and len(query.split())>64:query=' '.join(query.split()[:64])
        return query,selected,{**meta,'selected_words':len(selected),'query_words':len(query.split()) if query else 0}

def compact_calls(calls):return {call['route']:[r['doc_id'] for r in call['returned']] for call in calls}
def calls_from_lists(lists,query_id):return [{'route':route,'query_id':query_id,'returned':[{'doc_id':doc_id,'rank':j} for j,doc_id in enumerate(ids,1)]} for route,ids in lists.items()]
def fused_independent(initial,supplement,alpha):
    groups=[initial]+([supplement] if supplement else []);scores=collections.defaultdict(float)
    for group in groups:
        for route,ids in group.items():
            seen=set();weight=(alpha if route=='dense' else 1-alpha)/len(groups)
            for rank,doc_id in enumerate(ids,1):
                if doc_id not in seen:scores[doc_id]+=weight/(60+rank);seen.add(doc_id)
    return sorted(scores,key=lambda doc_id:(-scores[doc_id],doc_id))

def reference_rows(dataset,wanted=None):
    import gzip
    refs={};policies={'hybrid_rrf','no_second_branch','mix_ranked8','mix_ranked16','mix_rm3mix8','mix_rm3mix16'}
    for path in [c.OLD/'results'/c.ref_name(dataset)/'rankings.jsonl.gz',LEGACY/'results'/dataset/'new_rankings.jsonl.gz']:
        with gzip.open(path,'rt') as f:
            for line in f:
                r=json.loads(line)
                if r['policy'] not in policies or (wanted is not None and r['id'] not in wanted):continue
                refs[r['id'],r['policy']]={'ranking':r['ranking'],'query':r['expanded_query'],'at_k':r['at_k'],'source':str(path)}
    return refs

class IndependentGold:
    def __init__(self,backend,labels,mode):
        self.docs=backend.doc_lookup;self.labels={x['id']:x for x in labels};self.mode=mode;self.cache={};self.bytitle=collections.defaultdict(list)
        self.words={i:len(d['text'].split()) for i,d in self.docs.items()}
        for i,d in self.docs.items():self.bytitle[d['title']].append(i)
    def groups(self,qid):
        if qid not in self.cache:
            q=self.labels[qid]
            if self.mode=='paragraph':self.cache[qid]=[set(q['gold_doc_ids'])]
            else:
                norm=lambda t:' '.join(t.split());groups=[]
                for title in q['gold_titles']:
                    required=[norm(t) for t in q['supporting_sentences_by_title'][title]]
                    groups.append({i for i in self.bytitle[title] if required and all(t in norm(self.docs[i]['text']) for t in required)})
                self.cache[qid]=groups
        return self.cache[qid]
    def score(self,rank,qid,k):
        chosen=set(rank[:k]);groups=self.groups(qid);q=self.labels[qid]
        if self.mode=='paragraph':hit=len(chosen&groups[0]);total=len(groups[0])
        else:hit=sum(bool(chosen&g) for g in groups);total=len(groups)
        valid=q.get('strict_annotation_valid',True)
        return {'complete':int(hit==total) if valid else None,'recall':hit/total if valid else None,'title_complete':int(set(q['gold_titles'])<={self.docs[i]['title'] for i in chosen}),'documents':len(chosen),'words':sum(self.words[i] for i in chosen)}
