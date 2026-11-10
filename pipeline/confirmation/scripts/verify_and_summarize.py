"""Separate implementation for ranking/strict-cutoff checks and paired summaries."""
from pathlib import Path
import collections, csv, datetime, gzip, hashlib, json, sys
import joblib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/root/autodl-tmp/hybridrag-qsel-rank-p1-1-full-20261107')
DATA = Path('/root/autodl-tmp/hybridrag-adaptive-v4-20261107/data')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ratio(a,b):return float(a/b) if b else None
def readrows(p):
    with gzip.open(p,'rt') as f:return list(map(json.loads,f))

def verify(ds):
    contract=json.loads((ROOT/'frozen/protocol.json').read_text())
    assert sha(ROOT/'frozen/protocol.json')==(ROOT/'frozen/protocol.sha256').read_text().strip()
    for f,h in contract['frozen_files'].items():assert sha(ROOT/'frozen'/f)==h
    source=ROOT/(f'inputs/{ds}_questions.json' if ds=='musique' else f'frozen/{ds}_questions.json')
    qs=json.loads(source.read_text());seen=[];checks=collections.Counter()
    model=joblib.load(ROOT/f'models/{ds}/document_ranker.joblib');cfg=json.loads((ROOT/f'models/{ds}/frozen.json').read_text())
    assert sha(ROOT/f'models/{ds}/document_ranker.joblib')==cfg['ranker_sha256']
    if ds!='musique':assert cfg['ranker_sha256']==sha(OLD/f'models/{ds}/document_ranker.joblib') and cfg['rerank_weight']==.5
    else:
        assert cfg['fit_from_scratch'] and not cfg['warm_start'] and not cfg['confirmation_used_for_selection']
        assert cfg['rerank_weight']==max(cfg['rerank_candidates'],key=lambda v:v['development_macro_complete'])['weight']
        assert cfg['ranker_sha256']!=sha(OLD/f'models/{ds}/document_ranker.joblib')
    # Reconstruct qualifying single-version support directly from the corpus.
    required_titles={t for q in qs for t in q['gold_titles']};docs={};by_title=collections.defaultdict(list)
    with (DATA/ds/'corpus.jsonl').open() as f:
        for line in f:
            d=json.loads(line);docs[d['doc_id']]=d
            if d['title'] in required_titles:by_title[d['title']].append(d)
    compact={(r['id'],r['pool']):r for r in readrows(ROOT/f'results/{ds}_per_question.jsonl.gz')}
    assert len(compact)==2000
    for path in sorted((ROOT/f'cache/{ds}').glob('batch_*.joblib')):
        receipt=json.loads(path.with_suffix('.json').read_text());assert receipt['sha256']==sha(path)
        rr=joblib.load(path);assert [r['q']['id'] for r in rr]==receipt['ids']
        assert receipt['model_sha256']==cfg['ranker_sha256'] and receipt['questions_sha256']==sha(source)
        if ds=='musique':assert receipt['utc']>cfg['utc']>contract['frozen_utc']
        else:assert receipt['utc']>contract['frozen_utc']
        for row in rr:
            q=row['q'];seen.append(q['id'])
            if ds=='musique':units={i:[i] for i in q['gold_paragraph_ids']};assert all(i in docs for i in units)
            else:
                units={t:sorted(d['doc_id'] for d in by_title[t] if all(' '.join(s.split()) in ' '.join(d['text'].split()) for s in ss)) for t,ss in q['supporting_sentences_by_title'].items()}
            assert units==row['units'] and all(units.values());checks['support_units_rebuilt']+=len(units)
            for name,p in row['pools'].items():
                calls=p['calls'];assert len(calls) in [2,4] and all(len(x['returned'])==100 for x in calls)
                assert {x['route'] for x in calls}=={'bm25','dense'}
                assert sum(len(x['returned']) for x in calls)==p['returned_records']
                assert (len(calls)==4)==p['q1_executed']
                assert all(x['query']==q['question'] for x in calls if x['query_id']=='q0')
                nq=len({x['query_id'] for x in calls});score=collections.defaultdict(float)
                alpha=contract['retrieval']['alpha'][ds]
                for call in calls:
                    assert [v['rank'] for v in call['returned']]==list(range(1,101))
                    assert len({v['doc_id'] for v in call['returned']})==100
                    weight=(alpha if call['route']=='dense' else 1-alpha)/nq
                    for d in call['returned']:score[d['doc_id']]+=weight/(60+d['rank'])
                rank=sorted(score,key=lambda i:(-score[i],i));assert rank==p['rank'];checks['fusion_rankings_rebuilt']+=1
                prob=model.predict_proba(p['features'])[:,1]
                rrscore=np.array([1/(61+j) for j in range(len(rank))]);rrscore/=rrscore.max()
                s=(1-cfg['rerank_weight'])*rrscore+cfg['rerank_weight']*prob
                rerank=[rank[j] for j in sorted(range(len(rank)),key=lambda j:(-s[j],j))]
                assert rerank==p['reranked'];checks['model_rankings_rebuilt']+=1
                rstar=[]
                for ranking in [rank,rerank]:
                    hit=set();need=set(units);reverse=collections.defaultdict(set)
                    for unit,values in units.items():
                        for d in values:reverse[d].add(unit)
                    first=None
                    for k,d in enumerate(ranking,1):
                        hit.update(reverse[d])
                        if hit==need and first is None:first=k
                    rstar.append(first)
                assert rstar==[p['r_before'],p['r_after']]
                assert (rstar[0] is not None)==p['pool_complete'] and (rstar[1] is not None)==p['pool_complete']
                checks['strict_first_complete_rebuilt']+=2
                r=compact[q['id'],name]
                for k in ['pool_complete','r_before','r_after','q1_executed','returned_records']:assert r[k]==p[k]
                assert r['rank_sha256']==hashlib.sha256(json.dumps(rank).encode()).hexdigest()
                assert r['reranked_sha256']==hashlib.sha256(json.dumps(rerank).encode()).hexdigest()
                for cutoff in range(1,51):
                    a=set(rank[:cutoff]);b=set(rerank[:cutoff])
                    assert all(bool(a&set(v)) for v in units.values())==bool(r['r_before'] is not None and r['r_before']<=cutoff)
                    assert all(bool(b&set(v)) for v in units.values())==bool(r['r_after'] is not None and r['r_after']<=cutoff)
                    checks['strict_cutoff_checks']+=2
    assert seen==[q['id'] for q in qs] and len(set(seen))==1000
    return dict(checks)

def training_verify():
    ids=json.loads((ROOT/'frozen/musique_ids.json').read_text());keep=set(ids['train']);exclude=set(ids['confirmation']+ids['buffer']);dev=set(ids['development'])
    assert len(keep)==20990 and len(exclude)==1003 and len(dev)==300 and not keep & (exclude|dev)
    manifest=json.loads((ROOT/'training/filter_complete.json').read_text());rows=0;seen=set();kept_order=[]
    mapping=readrows(ROOT/'training/original_question_row_map.jsonl.gz')
    mapped=collections.defaultdict(list)
    for r in mapping:mapped[r['source_shard']].append(r)
    for rec in manifest['shards']:
        p=ROOT/rec['path'];assert sha(p)==rec['sha256'];z=np.load(p)
        original=OLD/'supervision/musique'/p.name;assert sha(original)==rec['source_sha256'];old=np.load(original)
        mask=np.zeros(len(old['dy']),bool);expected=[]
        for r in mapped[p.name]:
            assert r['excluded']==(r['id'] in exclude)
            if r['id'] in keep:
                kept_order.append(r['id'])
                for sp in r['spans']:
                    mask[sp['start']:sp['end']]=True;expected += [r['id']]*(sp['end']-sp['start'])
        assert z['question_ids'].tolist()==expected
        assert np.array_equal(z['dx'],old['dx'][mask]) and np.array_equal(z['dy'],old['dy'][mask])
        ss=set(z['question_ids'].tolist());assert ss<=keep and not ss&(exclude|dev);seen|=ss;rows+=len(z['dy'])
    assert seen==keep and kept_order==ids['train']
    groups=json.loads((ROOT/'frozen/musique_selected_groups.json').read_text())
    assert set(i for g in groups for i in g['train_canonical_ids'])==set(ids['confirmation'])
    assert all(not g['dev_canonical_ids'] and not set(g['canonical_ids'])&keep for g in groups)
    return {'retained_questions':len(seen),'retained_document_rows':rows,'excluded_confirmation':1000,'excluded_buffer':3,
            'confirmation_groups':len(groups),'all_filtered_features_bit_exact_to_retained_original_rows':True,'zero_excluded_training_rows':True}

def stats(rr,k):
    n=len(rr);p=sum(r['pool_complete'] for r in rr)
    a=[r['r_before'] is not None and r['r_before']<=k for r in rr]
    b=[r['r_after'] is not None and r['r_after']<=k for r in rr]
    x=sum(a);y=sum(b);w=sum(bb and not aa for aa,bb in zip(a,b));l=sum(aa and not bb for aa,bb in zip(a,b))
    assert y-x==w-l and max(x,y)<=p
    return {'n':n,'pool_complete':p,'K':k,'before':x,'after':y,'wins':w,'losses':l,
        'pool_rate':ratio(p,n),'before_rate':ratio(x,n),'after_rate':ratio(y,n),'delta_rate':ratio(y-x,n),
        'before_retention':ratio(x,p),'after_retention':ratio(y,p),'rescue_rate':ratio(w,p-x),
        'harm_rate':ratio(l,x),'net_conditional_gain':ratio(w-l,p)}

def confidence(rr,ds):
    # Metrics conditional on this frozen model/split; no training-seed uncertainty.
    rng=np.random.default_rng(2026111002);v=[]
    for r in rr:
        a=int(r['r_before'] is not None and r['r_before']<=5);b=int(r['r_after'] is not None and r['r_after']<=5)
        v.append([1,int(r['pool_complete']),a,b,int(b and not a),int(a and not b)])
    v=np.asarray(v,float)
    if ds=='musique':
        groups=sorted({r['group_id'] for r in rr});ix={g:j for j,g in enumerate(groups)};s=np.zeros((len(groups),6))
        for row,r in zip(v,rr):s[ix[r['group_id']]]+=row
        draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)@s
    else:
        strata=collections.defaultdict(list)
        for j,r in enumerate(rr):strata[r['support_count'],r['type']].append(j)
        draws=np.zeros((2000,6))
        for key in sorted(strata):
            js=strata[key];draws += rng.multinomial(len(js),np.full(len(js),1/len(js)),size=2000)@v[js]
    n,p,a,b,w,l=draws.T
    series={'pool_rate':p/n,'before_rate':a/n,'after_rate':b/n,'delta_rate':(b-a)/n,
            'before_retention':a/p,'after_retention':b/p,'rescue_rate':w/(p-a),'harm_rate':l/a,'net_conditional_gain':(w-l)/p}
    return {key:np.quantile(x[np.isfinite(x)],[.025,.975]).tolist() for key,x in series.items()}

def summarize():
    result={};flat=[]
    for ds in ['2wiki','hotpotqa','musique']:
        rr=readrows(ROOT/f'results/{ds}_per_question.jsonl.gz');result[ds]={}
        for name in ['S2','A']:
            sub=[r for r in rr if r['pool']==name];assert len(sub)==1000
            grid=[stats(sub,k) for k in range(1,51)]
            strata={'all':stats(sub,5),'executed':stats([r for r in sub if r['q1_executed']],5),
                    'skipped':stats([r for r in sub if not r['q1_executed']],5)}
            depths=[]
            for lo,hi in [(6,10),(11,20),(21,50),(51,1000000)]:
                failed=[r for r in sub if r['r_before'] is not None and lo<=r['r_before']<=hi]
                x=stats(failed,5);depths.append({'min_original_rank':lo,'max_original_rank':hi,**x})
            support={str(n):stats([r for r in sub if r['support_count']==n],5) for n in sorted({r['support_count'] for r in sub})}
            result[ds][name]={'K5':grid[4],'K5_bootstrap_95':confidence(sub,ds),'K_grid':grid,'execution_strata':strata,
                'failure_depth_strata':depths,'support_count_strata':support,
                'gamma':sub[0]['gamma'],'median_unique_documents':float(np.median([r['unique_documents'] for r in sub])),
                'mean_returned_records':float(np.mean([r['returned_records'] for r in sub]))}
            flat += [{'dataset':ds,'pool':name,**x} for x in grid]
    write('results/summary.json',result)
    with (ROOT/'results/K_grid.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps({ds:{p:x['K5'] for p,x in v.items()} for ds,v in result.items()},indent=2))

if __name__=='__main__':
    checks={'training':training_verify(),'datasets':{}}
    for ds in ['2wiki','hotpotqa','musique']:
        checks['datasets'][ds]=verify(ds)
        print(json.dumps({'verified':ds,**checks['datasets'][ds]}),flush=True)
    write('verification/independent_recomputation.json',{'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**checks})
    summarize()
