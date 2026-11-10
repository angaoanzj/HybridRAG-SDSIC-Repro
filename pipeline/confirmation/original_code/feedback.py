"""Public-text-only feedback selection. No answer/support labels accepted."""
import math,collections
import numpy as np
from query_expansion import words,STOPWORDS,WORD_PATTERN,containing_sentence
from expansion_variants import VariantExpander
from order_variants import VariantExpander as OrderExpander
BASE=('hybrid_rrf','one_target','first_two_valid_titles','no_second_branch','bridge_expand','evidence_feedback')
NEW=('fallback_feedback','feedback_ranked24','mix_order8','mix_order16','mix_ranked8','mix_ranked16')
ALL=BASE+NEW
RERANK=('no_second_branch','evidence_feedback','mix_ranked8','mix_ranked16')
class Feedback:
 def __init__(self,backend):
  self.backend=backend;self.exp=VariantExpander(backend.docs,policy='no_second_branch');lookup=self.exp.title_lookup
  self.exps={p:(OrderExpander(policy=p,title_lookup=lookup) if p=='first_two_valid_titles' else VariantExpander(policy=p if p!='hybrid_rrf' else 'no_second_branch',title_lookup=lookup)) for p in BASE}
  df=np.asarray(backend.bm25.getnnz(axis=0)).ravel();n=len(backend.docs);idf=np.log1p((n-df+.5)/(df+.5));self.idf={t:float(idf[i]) for t,i in backend.vocabulary.items()}
 def terms(self,question,ranking,titles,n,mode):
  qt=set(words(question))-STOPWORDS;exclude=set(words(question))|STOPWORDS|{w for t in titles for w in words(t)};denom=sum(self.idf.get(w,0) for w in qt) or 1.;pool=[]
  for rank,i in enumerate(ranking[:5],1):
   d=self.backend.doc_lookup[i];family=' '.join(words(d['title']));matches=list(WORD_PATTERN.finditer(d['text']))[:180]
   source_overlap=sum(self.idf.get(w,0) for w in qt&set(words(d['title'])))/denom
   for pos,m in enumerate(matches):
    token=m.group().lower()
    if token in exclude or len(token)<3 or token.isdigit():continue
    ss=containing_sentence(d,m.start(),m.end())
    if ss is None:continue
    sentence,_=ss;overlap=sum(self.idf.get(w,0) for w in qt&set(words(sentence)))/denom
    relevance=overlap+.25*source_overlap;score=self.idf.get(token,0)*relevance/(60+rank)
    pool.append({'token':token,'doc_id':i,'source_title':d['title'],'family':family,'rank':rank,'position':pos,'sentence':sentence,'sentence_overlap':overlap,'source_overlap':source_overlap,'idf':self.idf.get(token,0),'score':score})
  pool.sort(key=(lambda x:(-x['score'],x['rank'],x['position'],x['token'])) if mode=='ranked' else (lambda x:(x['rank'],x['position'],x['token'])))
  seen=set();counts=collections.Counter();selected=[];cap=6 if n==24 else 4
  for item in pool:
   if item['token'] in seen or counts[item['family']]>=cap:continue
   selected.append(item);seen.add(item['token']);counts[item['family']]+=1
   if len(selected)>=n:break
  return selected
 def expand(self,q,ranking,policy):
  assert isinstance(q,str)
  if policy in BASE:return self.exps[policy].bridge_query(q,ranking,self.backend.doc_lookup)
  bridge=self.exp.bridge_query(q,ranking,self.backend.doc_lookup)
  if policy=='fallback_feedback':
   if bridge['query']:return {**bridge,'provenance':{**bridge['provenance'],'route':'title'}}
   r=self.exps['evidence_feedback'].feedback_query(q,ranking,self.backend.doc_lookup)
   return {**r,'provenance':{**r['provenance'],'route':'feedback_fallback'}}
  mixed=policy.startswith('mix_');titles=[x['mentioned_title'] for x in bridge['provenance']['entities']] if mixed else []
  n=24 if policy=='feedback_ranked24' else 8 if policy.endswith('8') else 16
  selected=self.terms(q,ranking,titles,n,'ranked' if 'ranked' in policy else 'order');residual=bridge['provenance']['residual_keywords'];feedback=[x['token'] for x in selected]
  query=' '.join(titles+feedback+residual) if titles or feedback else None
  if query and len(query.split())>64:query=' '.join(query.split()[:64])
  return {'query':query,'provenance':{'kind':policy,'titles':titles,'feedback_terms':selected,'requested_terms':n,'actual_terms':len(selected),'per_title_family_cap':6 if n==24 else 4,'residual_keywords':residual,'uses_labels':False},'reason':'mixed_or_feedback' if query else 'no_public_expansion'}

def fuse_weighted(calls,alpha,q0_weight=.7):
 scores=collections.defaultdict(float);query_ids={c['query_id'] for c in calls}
 for c in calls:
  w=(alpha if c['route']=='dense' else 1-alpha)*(1 if len(query_ids)==1 else q0_weight if c['query_id']=='q0' else 1-q0_weight);seen=set()
  for r in c['returned']:
   if r['doc_id'] in seen:continue
   scores[r['doc_id']]+=w/(60+r['rank']);seen.add(r['doc_id'])
 return sorted(scores,key=lambda i:(-scores[i],i))
