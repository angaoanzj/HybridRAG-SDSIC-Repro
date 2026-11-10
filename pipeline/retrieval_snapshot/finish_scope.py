"""Frozen-manifest statistics, CPU/GPU audit, aggregate-only final receipt."""
from pathlib import Path
import sys,json,gzip,time,hashlib,traceback,collections,types
import numpy as np
import scope_core as sc
import run_scope as run

R=sc.ROOT

def path_for(ds,role,index,qid):
    parts=1 if role=='calibration' else {'2wiki':6,'hotpotqa':4,'musique':2}[ds]
    return R/'parts'/ds/role/f'part{index%parts}'/(sc.digest(qid)[:24]+'.json.gz')

def condition_pairs():
    index={x:i for i,x in enumerate(sc.CONDITIONS)};pairs=[]
    for m in sc.MS:
        for w in sc.WINDOWS:
            for n in sc.CAPS:pairs.append((index[m,w,'rm',n],index[m,w,'idf',n],'scoring'))
    for m in [3,10,20]:
        for w in sc.WINDOWS:
            for score in sc.SCORES:
                for n in sc.CAPS:pairs.append((index[m,w,score,n],index[5,w,score,n],'source_count'))
    for m in sc.MS:
        for w in sc.WINDOWS[1:]:
            for score in sc.SCORES:
                for n in sc.CAPS:pairs.append((index[m,w,score,n],index[m,'head180',score,n],'window'))
    for i in range(64):pairs.extend([(i,64,'bridge_reference'),(i,65,'RRF_reference')])
    assert len(pairs)==len(set((a,b) for a,b,_ in pairs))==256
    return pairs

def effectiveness_pairs(ds):
    _,_,pub,labels,_=sc.data_inputs(ds,'effectiveness');lab={x['id']:x for x in labels};array=np.full((len(pub),66,len(sc.KS)),-1,np.int8)
    for j,q in enumerate(pub):
        with gzip.open(path_for(ds,'effectiveness',j,q['id']),'rt') as f:r=json.load(f)
        assert r['id']==q['id']
        for i,qi,si,pi in r['rows']:
            for ki,row in enumerate(r['queries'][qi]['at_k']):
                if row[0] is not None:array[j,i,ki]=int(row[0])
    valid=np.asarray([lab[q['id']].get('strict_annotation_valid',True) for q in pub]);assert np.all(array[valid]>=0) and np.all(array[~valid]==-1);array=array[valid]
    comparisons={};seed=sc.load(R/'protocol.json')['paired_bootstrap_seed'];B=2000
    for ai,bi,kind in condition_pairs():
        method=run.ALL_METHODS[ai];base=run.ALL_METHODS[bi];result={}
        for ki,k in enumerate(sc.KS):
            delta=array[:,ai,ki]-array[:,bi,ki];wins=int((delta>0).sum());losses=int((delta<0).sum());n=len(delta);ties=n-wins-losses
            rng=np.random.default_rng(int(sc.digest([seed,ds,method,base,k])[:16],16));draw=rng.multinomial(n,[wins/n,losses/n,ties/n],size=B);boot=(draw[:,0]-draw[:,1])/n*100
            result[str(k)]={'n':n,'delta_pp':float(delta.mean()*100),'wins':wins,'losses':losses,'ties':ties,'CI95_pp':np.percentile(boot,[2.5,97.5]).tolist()}
        comparisons[method+' minus '+base]={'kind':kind,'at_k':result}
    (R/'comparisons').mkdir(exist_ok=True)
    sc.write(R/'comparisons'/f'{ds}.json',{'comparisons':comparisons,'n_strict':int(valid.sum()),'n_all':len(pub),'scope':'2000 marginal paired-question bootstrap draws using equivalent multinomial win/loss/tie counts for binary complete differences; nominal pointwise95%; no multiplicity adjustment; no support-component or constituent clustering; historically exposed exploratory evaluation'})
    return {'questions':len(pub),'strict_valid':int(valid.sum()),'pairs':256,'K':len(sc.KS)}

def cpu_quality_audit(ds):
    cp,ip,pub,labels,mode=sc.data_inputs(ds,'calibration');docs={}
    with cp.open() as f:
        for line in f:
            d=json.loads(line);docs[d['doc_id']]=d
    docids=sc.load(ip/'docids.json');ig=sc.IndependentGold(types.SimpleNamespace(doc_lookup=docs),labels,mode);results={}
    for i,name in enumerate(run.ALL_METHODS):results[name]={'requests':0,'query_differences':0,'top50_differences':0,'metric_differences':{str(k):dict.fromkeys(run.FIELDS,0) for k in sc.KS},'repeat_output_differences':0}
    first={}
    for j,q in enumerate(pub):
        with gzip.open(path_for(ds,'calibration',j,q['id']),'rt') as f:record=json.load(f)
        reference={i:record['queries'][qi]['at_k'] for i,qi,si,pi in record['rows']}
        for rep in range(2):
            path=R/'timing/raw'/ds/(f'rep{rep}_'+sc.digest(q['id'])[:24]+'.json.gz')
            with gzip.open(path,'rt') as f:block=json.load(f)
            assert len(block)==66 and {r['method_index'] for r in block}==set(range(66))
            for row in block:
                i=row['method_index'];dst=results[run.ALL_METHODS[i]];dst['requests']+=1;dst['query_differences']+=not row['GPU_quality_query_equal'];dst['top50_differences']+=not row['GPU_quality_top50_equal']
                rank=[docids[n] for n in row['rank50']];assert len(rank)==len(set(rank))==50
                state=(row['query_sha256'],row['rank50'])
                if rep==0:first[q['id'],i]=state
                else:dst['repeat_output_differences']+=state!=first[q['id'],i]
                for ki,k in enumerate(sc.KS):
                    v=ig.score(rank,q['id'],k)
                    for fi,field in enumerate(run.FIELDS):
                        expected=reference[i][ki][fi];actual=v[field]
                        equal=(actual is None and expected is None) or (actual is not None and expected is not None and abs(actual-expected)<1e-12)
                        dst['metric_differences'][str(k)][field]+=not equal
    assert all(v['requests']==200 for v in results.values())
    sc.write(R/'aggregates'/f'{ds}_CPU_quality_audit.json',{'methods':results,'same100_calibration_and_timing':True,'scope':'CPU8 independently executed timed outputs vs GPUFP32 offline outputs on the frozen calibration/timing100; labels loaded only during this post-timing audit; no tuning'})
    return {'requests':sum(v['requests'] for v in results.values()),'query_differences':sum(v['query_differences'] for v in results.values()),'top50_differences':sum(v['top50_differences'] for v in results.values()),'strict_complete50_differences':sum(v['metric_differences']['50']['complete'] for v in results.values()),'repeat_output_differences':sum(v['repeat_output_differences'] for v in results.values())}

def inventory():
    out={}
    for ds in ['2wiki','hotpotqa','musique']:
        files=sorted((R/'parts'/ds).glob('*/*/*.json.gz'));h=hashlib.sha256();size=0
        for p in files:size+=p.stat().st_size;h.update(str(p.relative_to(R)).encode()+b'\0'+sc.sha(p).encode()+b'\n')
        db=R/'cache'/f'{ds}.sqlite';out[ds]={'question_trace_files':len(files),'trace_bytes':size,'trace_directory':str(R/'parts'/ds),'trace_tree_sha256':h.hexdigest(),'query_cache_path':str(db),'query_cache_bytes':db.stat().st_size,'query_cache_sha256':sc.sha(db),'trace_format':'method indices, unique selections, metrics, and references to compressed int32 route rankings; corpus docid dictionary remains in original index'}
    sc.write(R/'aggregates/raw_retained_inventory.json',{'scope':'raw new trajectories remain on original server; only this aggregate inventory is returned','datasets':out});return out

def main():
    sc.write(R/'statistics_sources.json',{'unix':time.time(),'source_sha256':sc.sha(Path(__file__)),'planned_pairs':256,'protocol_sha256':sc.sha(R/'protocol.json')})
    while not ((R/'quality_complete.json').exists() and (R/'timing_complete.json').exists()):time.sleep(5)
    run.check_sources();f=sc.load(R/'scientific_frozen.json');assert sc.sha(R/'calibration_frozen.json')==f['calibration_frozen_sha256'] and sc.sha(R/'sampling_certificate.json')==f['sampling_certificate_sha256']
    certificate=sc.load(R/'sampling_certificate.json');request=sc.load(R/'data_request.json')
    for ds,spec in request['datasets'].items():
        assert sc.sha(Path(spec['source_public']))==certificate['datasets'][ds]['source_public_sha256']
        assert sc.sha(Path(spec['source_labels']))==certificate['datasets'][ds]['source_labels_sha256']
    comparisons={};cpu={}
    for ds in ['2wiki','hotpotqa','musique']:
        comparisons[ds]=effectiveness_pairs(ds);cpu[ds]=cpu_quality_audit(ds);print(json.dumps({'statistics_dataset':ds,'pairs':256,'CPU_audit':cpu[ds]}),flush=True)
    raw=inventory();quality=sc.load(R/'quality_complete.json');timing=sc.load(R/'timing_complete.json')
    assert timing['requests']==39600
    out={'status':'complete','version':'FBSCOPE-R2','main_conditions':64,'reference_conditions':2,'m':[3,5,10,20],'windows':list(sc.WINDOWS),'feedback_words':[8,16],'scores':['idf','rm'],'K':list(sc.KS),'effectiveness':comparisons,'CPU_timing_audit':cpu,'timing_requests':39600,'quality':quality,'calibration':sc.load(R/'calibration_frozen.json'),'sampling_certificate_sha256':sc.sha(R/'sampling_certificate.json'),'scientific_frozen_sha256':sc.sha(R/'scientific_frozen.json'),'raw_retained_bytes':sum(x['trace_bytes']+x['query_cache_bytes'] for x in raw.values()),'scope':'historically exposed questions, unified frozen IDs, strict validity shared across methods, calibration/timing100 excluded from effectiveness; no reader or training; all methods/all grid cells retained','unix':time.time()}
    sc.write(R/'complete.json',out);sc.progress('complete',main_conditions=64,timing_requests=39600)

if __name__=='__main__':
    try:main()
    except Exception:sc.write(R/'statistics_failed.json',{'traceback':traceback.format_exc(),'unix':time.time()});raise
