"""Reconstruct frozen rules, returned-list fusion and support metrics without runner imports."""
from __future__ import annotations
import math,re
from collections import defaultdict

WORD=re.compile(r'[a-z0-9]+',re.I)
STOP=set('a an the and or but if then of in on at to for from by with without is are was were be been being do does did who what which when where how why whose whom both either neither than that this these those it its as has have had can could would should will may might also please tell give name named called two these those each between among'.split())
CUES=set('directed director written writer author authored born birthplace spouse married husband wife father mother son daughter founded founder created creator produced producer composed composer'.split())
TITLE_POLICIES={'one_target','first_two_valid_titles','no_second_branch','bridge_expand','score_two_reverse'}
def require(ok,message):
    if not ok:raise ValueError(message)
def normalized(s):return ' '.join(s.split())
def words(s):return [m.group().lower() for m in WORD.finditer(s)]
def checked(ids,docs):
    require(isinstance(ids,list) and len(ids)==len(set(ids)),'Duplicate selected/returned document IDs')
    require(all(i in docs for i in ids),'Unknown document ID')
    return ids
def fusion(calls,alpha):
    qids={c['query_id'] for c in calls};scores=defaultdict(float)
    for c in calls:
        weight=(alpha if c['route']=='dense' else 1-alpha)/len(qids);seen=set()
        for r in c['returned']:
            if r['doc_id'] not in seen:scores[r['doc_id']]+=weight/(60+r['rank']);seen.add(r['doc_id'])
    return sorted(scores,key=lambda d:(-scores[d],d))

class IndependentRules:
    def __init__(self,docs):
        self.docs=docs;self.dictionary={};self.cache={}
        for doc in docs.values():
            key=tuple(words(doc['title']))
            if 2<=len(key)<=8 and any(t not in STOP for t in key):self.dictionary.setdefault(key,set()).add(doc['title'])
    def matches(self,tokens):
        p=0
        while p<len(tokens):
            for size in range(min(8,len(tokens)-p),1,-1):
                key=tuple(tokens[p:p+size])
                if key in self.dictionary:
                    for title in sorted(self.dictionary[key]):yield p,p+size,key,title
                    p+=size;break
            else:p+=1
    def pool(self,question,initial):
        question_words=words(question);explicit={key for a,b,key,t in self.matches(question_words)}
        excluded={w for key in explicit for w in key};residual=[]
        for w in question_words:
            if w not in STOP and w not in excluded and w not in residual:residual.append(w)
            if len(residual)==24:break
        candidates=[]
        for rank,did in enumerate(initial[:5],1):
            d=self.docs[did];text=d['text'];matches=list(WORD.finditer(text))[:180];tokens=[m.group().lower() for m in matches];source_key=tuple(words(d['title']))
            for a,b,key,title in self.matches(tokens):
                if key==source_key or key in explicit:continue
                start,end=matches[a].start(),matches[b-1].end();cursor=0;sentence=None
                for s in d['sentences']:
                    stripped=s.strip()
                    if not stripped:continue
                    pos=text.find(stripped,cursor)
                    if pos<0:continue
                    cursor=pos+len(stripped)
                    if pos<=start and end<=cursor:
                        prefix=s[:start-pos+len(s)-len(s.lstrip())];sentence=s;break
                if sentence is None:continue
                score=4*float(source_key in explicit)+1.5*float(any(w in CUES for w in words(prefix)[-5:]))+1/rank+0.1/(1+a)+0.01*len(key)
                candidates.append({'key':key,'title':title,'doc_id':did,'source_family':source_key,'rank':rank,'start':start,'score':score,'source_explicit':source_key in explicit})
        return candidates,residual
    def expected(self,policy,question,initial,pool=None):
        candidates,residual=pool if pool is not None else self.pool(question,initial)
        if policy in {'hybrid_rrf','repeat_q0'}:return [], question if policy=='repeat_q0' else None
        if policy=='evidence_feedback':
            excluded=set(words(question))|STOP;tokens=[]
            for did in initial[:5]:
                for token in words(self.docs[did]['text'])[:180]:
                    if token not in excluded and token not in tokens:tokens.append(token)
                    if len(tokens)==24:break
                if len(tokens)==24:break
            query=' '.join(tokens+residual) if tokens else None
            return [],None if query and query.strip()==question.strip() else query
        byorder=lambda c:(c['rank'],c['start'],c['title'])
        byscore=lambda c:(-c['score'],c['rank'],c['start'],c['title'])
        candidates=sorted(candidates,key=byorder if policy=='first_two_valid_titles' else byscore)
        chosen=[];keys=set();families=set();limit=1 if policy=='one_target' else 2
        while len(chosen)<limit:
            possible=[c for c in candidates if c['key'] not in keys]
            if not possible:break
            if policy=='bridge_expand' and chosen:
                alternative=[c for c in possible if c['source_explicit'] and c['source_family'] not in families]
                if alternative:possible=alternative
            c=possible[0];chosen.append(c);keys.add(c['key']);families.add(c['source_family'])
        if policy=='score_two_reverse' and len(chosen)==2:chosen.reverse()
        query=' '.join([c['title'] for c in chosen]+residual) if chosen else None
        return chosen,None if query and query.strip()==question.strip() else query

def audit_group(row,documents,alpha,question,timing=False,rules=None,pool=None):
    require(row.get('status')=='complete','Incomplete group')
    require(row.get('question')==question,'Executed question text differs')
    require(row.get('gpu_disabled') is True and row.get('cache_queries') is False and row.get('warmup_excluded') is True,'Execution policy mismatch')
    calls=row['calls'];require(2<=len(calls)<=4,'Invalid call count');pairs=set();texts={};union=set();dcount=0
    require(len({c['call_id'] for c in calls})==len(calls),'Duplicate physical call ID')
    for c in calls:
        pair=(c['query_id'],c['route']);require(pair not in pairs and pair[0] in {'q0','q1'} and pair[1] in {'bm25','dense'},'Invalid/duplicate query-route pair');pairs.add(pair)
        require(c['round']==(0 if pair[0]=='q0' else 1),'Wrong round')
        checked([x['doc_id'] for x in c['returned']],documents)
        require([x['rank'] for x in c['returned']]==list(range(1,len(c['returned'])+1)),'Non-contiguous ranks')
        require(isinstance(c['depth'],int) and len(c['returned'])==c['depth'],'Requested/returned count mismatch')
        require(all(math.isfinite(float(x['raw_score'])) for x in c['returned']),'Nonfinite backend score')
        require(c['cache_hit'] is False and c['query_cache_enabled'] is False,'Cached query used')
        require(c['query_id'] not in texts or texts[c['query_id']]==c['query'],'Inconsistent logical query text');texts[c['query_id']]=c['query']
        for f in ('query_encode_seconds','search_seconds','total_seconds'):require(math.isfinite(c[f]) and c[f]>=0,'Invalid physical timing')
        union.update(x['doc_id'] for x in c['returned']);dcount+=len(c['returned'])
    require(texts.get('q0')==question and texts.get('q1')==row.get('expanded_query'),'Call/query text mismatch')
    require(dcount<=400 and len(texts)<=2,'Exceeded Q/D')
    policy=row['policy'];basepairs={('q0','bm25'),('q0','dense')}
    if policy=='hybrid_rrf':
        require(pairs==basepairs and all(c['depth']==200 for c in calls),'RRF allocation mismatch')
        initial=fusion([{**c,'returned':c['returned'][:100]} for c in calls],alpha)
    else:
        wanted=basepairs|({('q1','bm25'),('q1','dense')} if row.get('expanded_query') is not None else set())
        require(pairs==wanted and all(c['depth']==100 for c in calls),'Two-round allocation mismatch')
        initial=fusion([c for c in calls if c['query_id']=='q0'],alpha)
    ranked=fusion(calls,alpha)
    require(row['ranking']==ranked and set(ranked)==union,'Independent final RRF mismatch')
    require(row['initial']==initial,'Independent common initial100/route RRF mismatch')
    require(row['initial_selections']['50']==initial[:50],'Initial Top50 mismatch')
    require(set(row['selections'])=={'10','20','30','50'},'Wrong selected K values')
    for k in (10,20,30,50):require(checked(row['selections'][str(k)],documents)==ranked[:k],'Final TopK IDs mismatch')
    require((row['j_count'],row['q_count'],row['d_count'])==(len(calls),len(texts),dcount),'Independent call/D counts differ')
    require(row['q1_provenance']==row['expansion_provenance'],'Provenance aliases differ')
    diag=row['reviewer_diagnostics'];require(diag.get('first_round_F50') is None,'Runner contains an offline-label-derived metric')
    require(diag['initial_selections']['50']==initial[:50],'Diagnostic initial Top50 mismatch')
    prov=row['expansion_provenance'] if isinstance(row['expansion_provenance'],dict) else {};entities=prov.get('entities',[])
    require(diag['selected_count']==len(entities) and diag['selected_titles']==[e['mentioned_title'] for e in entities],'Selected count/title mismatch')
    require(diag['selected_keys']==[e['entity_tokens'] for e in entities],'Selected-key mismatch')
    for e in entities:
        require(e['doc_id'] in initial[:5],'Expansion source outside Top5')
        d=documents[e['doc_id']];a,b=e['source_text_span'];sa,sb=e['sentence_span']
        require(d['title']==e['source_title'] and d['text'][a:b]==e['mention_text'],'Source-text provenance mismatch')
        require(e['sentence'] in d['sentences'] and e['sentence'][sa:sb]==e['mention_text'],'Sentence provenance mismatch')
    if policy=='repeat_q0':
        require(len(calls)==4 and texts['q1']==question and dcount==400,'Repeat did not actually issue all four calls')
        require(prov.get('kind')=='forced_actual_original_query_repeat','Missing repeat provenance')
        for route in ('bm25','dense'):
            a=next(c for c in calls if c['query_id']=='q0' and c['route']==route);b=next(c for c in calls if c['query_id']=='q1' and c['route']==route)
            require(a['returned']==b['returned'],'Same-query repeated returned lists differ')
    if not timing:
        require(rules is not None and pool is not None,'Missing independent pool audit')
        candidates,residual=pool;keys={c['key'] for c in candidates}
        require(diag['candidate_pool_occurrence_count']==len(candidates),'Candidate occurrence count mismatch')
        require(diag['candidate_pool_distinct_count']==len(keys),'Distinct candidate count mismatch')
        require(diag['candidate_pool_distinct_keys']==[list(k) for k in sorted(keys)],'Distinct candidate keys mismatch')
        occurrence_keys={tuple(x['entity_key']) for x in diag['candidate_occurrences']}
        require(occurrence_keys==keys and len(diag['candidate_occurrences'])==len(candidates),'Recorded candidate pool inconsistent')
        chosen,query=rules.expected(policy,question,initial,pool)
        require(query==row['expanded_query'],'Independently reconstructed q1 mismatch')
        require([c['title'] for c in chosen]==[e['mentioned_title'] for e in entities],'Independently reconstructed title selection/order mismatch')
        for c,e in zip(chosen,entities):
            require(c['doc_id']==e['doc_id'] and c['start']==e['source_text_span'][0],'Selected occurrence mismatch')
        if policy=='score_two_reverse':
            control=prov['reviewer_order_control'];ref,refq=rules.expected('no_second_branch',question,initial,pool)
            require(control['reference_query']==refq and control['same_selected_target_set'] is True and control['residual_keywords_unchanged'] is True,'Invalid reversal reference')
            require([x['mentioned_title'] for x in control['reference_entities']]==[x['title'] for x in ref],'Reversal reselected different targets')
            require(control['reversed_two_title_order']==(len(ref)==2),'Wrong reversal count')
    for f in ('total_seconds','selection_seconds','pipeline_plus_selection_seconds'):require(math.isfinite(row[f]) and row[f]>=0,'Invalid group timing')
    require(math.isclose(row['pipeline_plus_selection_seconds'],row['total_seconds']+row['selection_seconds'],abs_tol=1e-12,rel_tol=1e-10),'Timing sum mismatch')
    return {'j_count':len(calls),'q_count':len(texts),'d_count':dcount,'cumulative_ids':sorted(union),'initial_ids':initial[:50]}

def score(selected,label,docs):
    gold=set(label['gold_titles']);require(bool(gold),'Empty gold title set');titles={docs[i]['title'] for i in selected};strict=set()
    for did in selected:
        d=docs[did];t=d['title']
        if t in gold and all(normalized(s) in d['normalized_text'] for s in label['supporting_sentences_by_title'][t]):strict.add(t)
    return {'R':len(gold&titles)/len(gold),'F':int(gold<=titles),'strictF':int(gold<=strict),'hit_titles':sorted(gold&titles),'strict_hit_titles':sorted(strict),'gold_titles':sorted(gold)}

def validate_labels(labels,docs):
    for label in labels.values():
        gold=set(label['gold_titles']);require(gold and len(label['gold_doc_ids'])==len(gold),'Incomplete gold version mapping')
        reached=set()
        for did in label['gold_doc_ids']:
            require(did in docs,'Gold support version absent');d=docs[did];title=d['title'];sentences=label['supporting_sentences_by_title'].get(title,[])
            require(title in gold and sentences and all(normalized(s) and normalized(s) in d['normalized_text'] for s in sentences),'Gold support sentence unreachable')
            reached.add(title)
        require(reached==gold,'Gold versions fail complete reachability')
