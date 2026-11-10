"""Unified sampling, backbone calibration, and compact 64-condition evaluation."""
from pathlib import Path
import sys,json,time,os,gzip,sqlite3,zlib,hashlib,collections,subprocess,traceback,unicodedata
import numpy as np
import scope_core as sc
from window_rules import check_rules

R=sc.ROOT;c=sc.c
FIELDS=('complete','recall','title_complete','documents','words')
ALL_METHODS=[sc.key(*x) for x in sc.CONDITIONS]+['no_second_branch','hybrid_rrf']
SCIENCE_FILES=['window_rules.py','scope_core.py','run_scope.py','data_request.json','seed_registration.json','sampling_manifest.json','protocol.json']

def check_sources():
    old=sc.load(sc.LEGACY/'execution_frozen.json');assert all(sc.sha(sc.LEGACY/n)==v for n,v in old['source_sha256'].items())
    p=R/'scientific_frozen.json'
    if p.exists():assert all(sc.sha(R/n)==v for n,v in sc.load(p)['files'].items())

def setup():
    check_sources();req=sc.load(R/'data_request.json');manifest=sc.load(R/'sampling_manifest.json');out={'status':'pass','datasets':{},'raw_data_transferred':False}
    for ds,spec in req['datasets'].items():
        public=sc.load(spec['source_public']);labels=sc.load(spec['source_labels']);by={q['id']:q for q in public};lab={q['id']:q for q in labels}
        assert len(by)==len(public) and len(lab)==len(labels)
        cal=spec['calibration_and_timing'];eff=spec['effectiveness'];assert len(cal)==100 and len(set(cal))==100 and not set(cal)&set(eff)
        assert len(eff)==len(set(eff)) and set(cal+eff)==set(spec.get('canonical_source_order',by))
        if ds=='musique':
            assert spec['source_order']==[q['id'] for q in public]
            assert hashlib.sha256(json.dumps(spec['source_order']).encode()).hexdigest()==spec['expected_original_order_hash']
            normalized=[' '.join(unicodedata.normalize('NFKC',q['question']).casefold().split()) for q in public]
            seen=set();canonical=[]
            for row,norm in zip(public,normalized):
                if norm not in seen:seen.add(norm);canonical.append(row['id'])
            assert canonical==spec['canonical_source_order']
            assert sc.sha(Path(spec['source_public']))==spec['source_public_sha256']
        else:
            assert all(hashlib.sha256(by[i]['question'].encode()).hexdigest()==h for i,h in spec['question_text_sha256'].items())
        assert all(i in lab for i in cal+eff)
        cp,ip,_,_,mode=sc.data_inputs(ds,'calibration');im=sc.load(ip/'index_manifest.json');assert im['corpus_sha256']==sc.sha(cp)
        out['datasets'][ds]={'source_public_sha256':sc.sha(Path(spec['source_public'])),'source_labels_sha256':sc.sha(Path(spec['source_labels'])),'corpus_sha256':im['corpus_sha256'],'index_manifest_sha256':sc.sha(ip/'index_manifest.json'),'calibration_timing_n':100,'effectiveness_n':len(eff),'calibration_timing_ids_sha256':sc.digest(cal),'effectiveness_ids_sha256':sc.digest(eff),'effectiveness_strict_valid_n':sum(lab[i].get('strict_annotation_valid',True) for i in eff),'calibration_strict_valid_n':sum(lab[i].get('strict_annotation_valid',True) for i in cal),'roles_intersection':0,'source_mode':mode}
    out['sampling_manifest_sha256']=sc.sha(R/'sampling_manifest.json');out['seed_registration_sha256']=sc.sha(R/'seed_registration.json')
    sc.write(R/'sampling_certificate.json',out)
    if (R/'setup_failed.json').exists():
        dst=R/'attempts/setup_failed_first.json';dst.parent.mkdir(exist_ok=True);(R/'setup_failed.json').replace(dst)
    sc.progress('sampling_verified',datasets={k:v['effectiveness_n'] for k,v in out['datasets'].items()})

def make_backend(ds,device,threads=8):
    cp,ip,_,_,_=sc.data_inputs(ds,'calibration',False)
    b=c.FastBackend(cp,ip,c.MODEL,threads=threads,device=device,cache_queries=True);b.warmup();return b,c.Content(b)

def title_metric(rank,label,docs):
    selected={docs[i]['title'] for i in rank[:50]};gold=set(label['gold_titles']);return len(selected&gold)/len(gold),int(gold<=selected)

def calibrate():
    assert (R/'sampling_certificate.json').exists();check_sources();out={'status':'complete','sampling_certificate_sha256':sc.sha(R/'sampling_certificate.json'),'dataset_results':{},'grid':[.3,.5,.7,.9],'objective':'mean recall@50, then complete@50, closest0.5, smaller alpha; q0 each route200; CPU8','data_roles':'exact frozen100 used again for every timing method; no effectiveness input'}
    for ds in ['2wiki','hotpotqa','musique']:
        _,_,pub,labels,mode=sc.data_inputs(ds,'calibration');b,fb=make_backend(ds,'cpu');gold=sc.IndependentGold(b,labels,mode);by={l['id']:l for l in labels};rows={a:[] for a in out['grid']}
        for i,q in enumerate(pub):
            calls=[c.measured_call(b,q['question'],route,200,'q0',route) for route in ['bm25','dense']]
            for a in out['grid']:
                rank=c.fused_rank(calls,a)
                if ds=='musique':v=gold.score(rank,q['id'],50);value=(v['recall'],v['complete'])
                else:value=title_metric(rank,by[q['id']],b.doc_lookup)
                rows[a].append(value)
            b._query_vectors.clear()
            if (i+1)%25==0:sc.progress('calibration',dataset=ds,questions=i+1,total=100)
        scores=[{'alpha':a,'mean_recall':float(np.mean([x[0] for x in rr])),'mean_complete':float(np.mean([x[1] for x in rr]))} for a,rr in rows.items()]
        best=max(scores,key=lambda x:(x['mean_recall'],x['mean_complete'],-abs(x['alpha']-.5),-x['alpha']))
        out['dataset_results'][ds]={'alpha':best['alpha'],'trials':scores,'n':100,'ordered_ID_sha256':sc.digest([q['id'] for q in pub]),'objective_metric':'native support paragraph recall' if ds=='musique' else 'support-title recall'}
        del b,fb,gold
    out['alpha']={ds:v['alpha'] for ds,v in out['dataset_results'].items()};sc.write(R/'calibration_frozen.json',out);sc.progress('calibration_complete',alpha=out['alpha'])

def primary_score(gold,rank,qid,k,labels):
    value=gold.score(rank,qid,k)
    if not labels[qid].get('strict_annotation_valid',True):value['complete']=None;value['recall']=None
    return value

def preflight():
    assert (R/'calibration_frozen.json').exists();check_sources();rules=check_rules();alpha=sc.load(R/'calibration_frozen.json')['alpha'];counts={'default_constructor_checks':0,'sentence_span_checks':0,'CPU_GPU_query_checks':0,'CPU_GPU_top50_differences':0,'independent_metric_checks':0}
    for ds in ['2wiki','hotpotqa','musique']:
        _,_,pub,labels,mode=sc.data_inputs(ds,'calibration');b,fb=make_backend(ds,'cpu');sel=sc.ScopeSelector(b,fb);gold=c.Gold(b,labels,mode);ig=sc.IndependentGold(b,labels,mode);labs={x['id']:x for x in labels}
        for q in [pub[0],pub[50],pub[-1]]:
            calls=[c.measured_call(b,q['question'],route,100,'q0',route) for route in ['bm25','dense']];rank=c.fused_rank(calls,alpha[ds]);state=fb.state(q['question'],rank);ctx=sel.context(state)
            initial=sc.compact_calls(calls);assert sc.fused_independent(initial,None,alpha[ds])==rank
            for n in sc.CAPS:
                query,chosen,_=sel.choose(ctx,5,'head180','idf',n);expected=fb.expand(q['question'],rank,f'mix_ranked{n}')['query'];assert query==expected;counts['default_constructor_checks']+=1
                class ZeroEmb:
                    def __getitem__(self,key):return np.zeros(1,np.float32)
                original=fb.state(q['question'],rank);fb.prepare_scores(original,ZeroEmb(),collections.defaultdict(float),np.zeros(1,np.float32));rm=fb.select(original,'rm3mix',n)
                assert sel.choose(ctx,5,'head180','rm',n)[0]==c.bounded_query(fb,original,rm);counts['default_constructor_checks']+=1
            for m,w,score,n in sc.CONDITIONS:
                text,chosen,meta=sel.choose(ctx,m,w,score,n);assert len(chosen)<=n and (not text or len(text.split())<=64)
                for token,r,pos,val in chosen:
                    profile,win=sel.window(state['initial'][r-1],w);assert pos in win['positions'] and profile['tokens'][pos]==token
            for doc_id in rank[:20]:
                profile=sel.profile(doc_id);matches=list(c.WORD_PATTERN.finditer(b.doc_lookup[doc_id]['text']))
                for j in {0,len(matches)//2,len(matches)-1}:
                    if not matches or j<0:continue
                    old=c.containing_sentence(b.doc_lookup[doc_id],matches[j].start(),matches[j].end());sid=profile['sentence_ids'][j]
                    assert (old[0] if old else None)==(profile['spans'][sid][2] if sid is not None else None);counts['sentence_span_checks']+=1
            for k in sc.KS:
                assert primary_score(gold,rank,q['id'],k,labs)==ig.score(rank,q['id'],k);counts['independent_metric_checks']+=5
            # Numerical device check uses calibration IDs only, no parameter selection.
            baseline=rank[:50];b._model.to('cuda');b.device='cuda';b._query_vectors.clear()
            gpu_calls=[c.measured_call(b,q['question'],route,100,'q0',route) for route in ['bm25','dense']];gpu_rank=c.fused_rank(gpu_calls,alpha[ds])
            counts['CPU_GPU_query_checks']+=1;counts['CPU_GPU_top50_differences']+=gpu_rank[:50]!=baseline
            b._model.to('cpu');b.device='cpu';b._query_vectors.clear()
        del b,fb,sel,gold,ig
    sc.write(R/'preflight_complete.json',{'status':'pass','window_rules':rules,**counts,'device_note':'GPU FP32 quality and CPU8 timing are separately recorded; complete100 timing will audit outputs for all conditions'})
    sc.write(R/'scientific_frozen.json',{'version':'FBSCOPE-R2','unix':time.time(),'files':{n:sc.sha(R/n) for n in SCIENCE_FILES},'calibration_frozen_sha256':sc.sha(R/'calibration_frozen.json'),'sampling_certificate_sha256':sc.sha(R/'sampling_certificate.json'),'legacy_source_manifest_sha256':sc.sha(sc.LEGACY/'execution_frozen.json')})

class CompactCache:
    def __init__(self,ds,backend):
        self.ds=ds;self.b=backend;self.index={i:j for j,i in enumerate(backend.doc_ids)};directory=R/'cache';directory.mkdir(exist_ok=True)
        self.db=sqlite3.connect(directory/(ds+'.sqlite'),timeout=90,isolation_level=None);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=NORMAL');self.db.execute('PRAGMA busy_timeout=90000');self.db.execute('PRAGMA wal_autocheckpoint=2000')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, query TEXT, depth INTEGER, payload BLOB)');self.actual_route_calls=0;self.hits=0
    def get(self,query,depth):
        key=sc.digest([query,depth]);row=self.db.execute('SELECT query,depth,payload FROM calls WHERE key=?',(key,)).fetchone()
        if row:
            assert row[0]==query and row[1]==depth;self.hits+=1;return key,np.frombuffer(zlib.decompress(row[2]),dtype='<i4').reshape(2,depth).copy()
        arrays=[]
        for route in ['bm25','dense']:
            found=self.b.search(query,route,depth)['results'];arrays.append([self.index[x['doc_id']] for x in found]);self.actual_route_calls+=1
        arr=np.asarray(arrays,dtype='<i4');assert arr.shape==(2,depth)
        payload=zlib.compress(arr.tobytes(),1);self.db.execute('INSERT OR IGNORE INTO calls VALUES (?,?,?,?)',(key,query,depth,payload));saved=self.db.execute('SELECT payload FROM calls WHERE key=?',(key,)).fetchone()[0]
        assert zlib.decompress(saved)==arr.tobytes(),'Conflicting deterministic query cache';self.b._query_vectors.clear();return key,arr
    def read(self,key):
        row=self.db.execute('SELECT query,depth,payload FROM calls WHERE key=?',(key,)).fetchone();assert row is not None
        return row[0],np.frombuffer(zlib.decompress(row[2]),dtype='<i4').reshape(2,row[1]).copy()
    def close(self):self.db.close()

def list_calls(arr,b,query_id):
    return [{'route':route,'query_id':query_id,'returned':[{'doc_id':b.doc_ids[int(i)],'rank':j} for j,i in enumerate(arr[z],1)]} for z,route in enumerate(['bm25','dense'])]
def fused(arrays,b,alpha):
    calls=[]
    for j,arr in enumerate(arrays):calls+=list_calls(arr,b,'q'+str(j))
    return c.fused_rank(calls,alpha)
def rank_sha(rank,cache):return hashlib.sha256(np.asarray([cache.index[i] for i in rank],dtype='<i4').tobytes()).hexdigest()

def make_question(q,b,fb,selector,cache,gold,labs,alpha):
    q0key,full=cache.get(q['question'],200);initial=full[:,:100];rank=fused([initial],b,alpha);s=fb.state(q['question'],rank);ctx=selector.context(s)
    plans=[]
    for i,(m,w,score,n) in enumerate(sc.CONDITIONS):
        text,selected,meta=selector.choose(ctx,m,w,score,n);plans.append((i,text,[[z[0],z[1],z[2]] for z in selected],meta))
    plans += [(64,s['bridge_query'],[],{}),(65,None,[],{})]
    queries=[];qlookup={};selections=[];slookup={};scopes=[];scope_lookup={};rows=[]
    for i,text,selected,meta in plans:
        marker=('rrf200',None) if i==65 else ('supplement',text if text and text.strip()!=q['question'].strip() else None)
        if marker not in qlookup:
            if i==65:extra=None;final=fused([full],b,alpha);qkey=None
            elif marker[1] is None:extra=None;final=rank;qkey=None
            else:qkey,extra=cache.get(text,100);final=fused([initial,extra],b,alpha)
            values=[]
            for k in sc.KS:
                metric=primary_score(gold,final,q['id'],k,labs);values.append([metric[f] for f in FIELDS])
            pool=primary_score(gold,final,q['id'],len(final),labs)['complete']
            qlookup[marker]=len(queries);queries.append({'cache_key':qkey,'kind':'rrf200' if i==65 else 'supplement' if extra is not None else 'initial100','rank_sha256':rank_sha(final,cache),'at_k':values,'pool_complete':pool,'query_words':len(text.split()) if text else 0})
        skey=sc.digest(selected)
        if skey not in slookup:slookup[skey]=len(selections);selections.append(selected)
        if i<64:
            vals=[meta[f] for f in ['source_documents','source_families','candidate_positions','eligible_occurrences','eligible_token_types']];scope_key=(sc.CONDITIONS[i][0],sc.CONDITIONS[i][1])
            if scope_key not in scope_lookup:scope_lookup[scope_key]=len(scopes);scopes.append(vals)
            scope_index=scope_lookup[scope_key]
        else:scope_index=None
        rows.append([i,qlookup[marker],slookup[skey],scope_index])
    return {'version':'FBSCOPE-R2','id':q['id'],'q0_cache_key':q0key,'rows':rows,'queries':queries,'selections':selections,'scopes':scopes}

def verify_record(record,q,b,fb,sel,cache,ig,alpha):
    qtext,full=cache.read(record['q0_cache_key']);assert qtext==q['question'];initial=full[:,:100];checks=0;budgets=0
    state=fb.state(q['question'],fused([initial],b,alpha));ranks=[];ctx=sel.context(state);qtexts={}
    for qr in record['queries']:
        if qr['kind']=='rrf200':arr=full;extra=None
        else:
            arr=initial
            if qr['cache_key']:
                text,extra=cache.read(qr['cache_key']);qtexts[qr['cache_key']]=text
            else:extra=None
        il={route:[b.doc_ids[int(i)] for i in arr[j]] for j,route in enumerate(['bm25','dense'])};el={route:[b.doc_ids[int(i)] for i in extra[j]] for j,route in enumerate(['bm25','dense'])} if extra is not None else None
        rank=sc.fused_independent(il,el,alpha);assert rank_sha(rank,cache)==qr['rank_sha256'];ranks.append(rank)
        for ki,k in enumerate(sc.KS):
            v=ig.score(rank,q['id'],k)
            for j,f in enumerate(FIELDS):
                expected=qr['at_k'][ki][j];assert (v[f] is None and expected is None) or (v[f] is not None and expected is not None and abs(v[f]-expected)<1e-12);checks+=1
        assert ig.score(rank,q['id'],len(rank))['complete']==qr['pool_complete'];checks+=1
    assert [row[0] for row in record['rows']]==list(range(66))
    for i,qi,si,pi in record['rows']:
        if i>=64:continue
        m,w,score,n=sc.CONDITIONS[i];chosen=record['selections'][si];tokens=set();family=collections.Counter()
        for token,source_rank,pos in chosen:
            assert source_rank<=m;profile,window=sel.window(state['initial'][source_rank-1],w)
            assert pos in window['positions'] and profile['tokens'][pos]==token and token not in tokens and token not in ctx['excluded'] and not token.isdigit() and len(token)>=3
            assert profile['sentence_ids'][pos] is not None;tokens.add(token);family[profile['family']]+=1;budgets+=1
        assert len(tokens)<=n and max(family.values(),default=0)<=4
        query=record['queries'][qi]
        if query['cache_key']:
            text=qtexts[query['cache_key']];expected=' '.join(state['titles']+[x[0] for x in chosen]+state['residual']);expected=' '.join(expected.split()[:64]) if len(expected.split())>64 else expected
            assert text==expected and len(text.split())<=64
        budgets+=1
    return checks,budgets

def empty_sums():return {m:{'n':0,'at_k':{str(k):{f:{'sum':0.,'n':0} for f in FIELDS} for k in sc.KS},'stats':{}} for m in ALL_METHODS}
def add_sum(sums,record):
    for i,qi,si,pi in record['rows']:
        m=sums[ALL_METHODS[i]];m['n']+=1;qr=record['queries'][qi]
        for ki,k in enumerate(sc.KS):
            for j,f in enumerate(FIELDS):
                value=qr['at_k'][ki][j]
                if value is not None:m['at_k'][str(k)][f]['sum']+=value;m['at_k'][str(k)][f]['n']+=1
        stats={'pool_complete':qr['pool_complete'],'selected_words':len(record['selections'][si]),'query_words':qr['query_words'],'physical_calls_if_independent':2 if qr['kind'] in ['initial100','rrf200'] else 4,'returned_records_if_independent':400 if qr['kind']!='initial100' else 200}
        if pi is not None:stats.update(dict(zip(['source_documents','source_families','candidate_positions','eligible_occurrences','eligible_token_types'],record['scopes'][pi])))
        for f,v in stats.items():
            if v is None:continue
            dst=m['stats'].setdefault(f,{'sum':0.,'n':0});dst['sum']+=v;dst['n']+=1

def worker(dataset,role,part,parts):
    check_sources();assert (R/'scientific_frozen.json').exists();out=R/'parts'/dataset/role/f'part{part}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    _,_,allpub,labels,mode=sc.data_inputs(dataset,role);pub=[q for i,q in enumerate(allpub) if i%parts==part];wanted={q['id'] for q in pub};labels=[q for q in labels if q['id'] in wanted]
    b,fb=make_backend(dataset,'cuda');sel=sc.ScopeSelector(b,fb);cache=CompactCache(dataset,b);gold=c.Gold(b,labels,mode);ig=sc.IndependentGold(b,labels,mode);labs={q['id']:q for q in labels};alpha=sc.load(R/'calibration_frozen.json')['alpha'][dataset]
    sc.write(out/'ordered_ids.json',[q['id'] for q in pub]);start=time.time()
    for j,q in enumerate(pub):
        p=out/(sc.digest(q['id'])[:24]+'.json.gz')
        if not p.exists():
            if os.statvfs(R).f_bavail*os.statvfs(R).f_frsize<55*2**20:raise RuntimeError('Owned output disk reserve reached; checkpoints retained')
            value=make_question(q,b,fb,sel,cache,gold,labs,alpha);tmp=p.with_suffix('.partial')
            with gzip.open(tmp,'wt',compresslevel=3) as f:json.dump(value,f,separators=(',',':'),ensure_ascii=False)
            tmp.replace(p)
        if (j+1)%10==0 or j+1==len(pub):sc.write(out/'status.json',{'stage':'quality','questions':j+1,'total':len(pub),'dataset':dataset,'role':role,'part':part,'seconds':time.time()-start,'new_route_calls':cache.actual_route_calls,'cache_hits':cache.hits,'free_MiB':os.statvfs(R).f_bavail*os.statvfs(R).f_frsize/2**20})
    sums=empty_sums();checks=0;budgets=0;output_bytes=0
    for j,q in enumerate(pub):
        p=out/(sc.digest(q['id'])[:24]+'.json.gz')
        with gzip.open(p,'rt') as f:record=json.load(f)
        assert record['id']==q['id'];a,z=verify_record(record,q,b,fb,sel,cache,ig,alpha);checks+=a;budgets+=z;add_sum(sums,record);output_bytes+=p.stat().st_size
        if (j+1)%50==0:sc.write(out/'status.json',{'stage':'saved_output_audit','questions':j+1,'total':len(pub),'dataset':dataset,'role':role,'part':part})
    sc.write(out/'summary_sums.json',sums);sc.write(out/'complete.json',{'status':'complete','dataset':dataset,'role':role,'part':part,'parts':parts,'questions':len(pub),'conditions':66,'independent_numeric_checks':checks,'budget_checks':budgets,'output_bytes':output_bytes,'new_route_calls':cache.actual_route_calls,'cache_hits':cache.hits,'seconds':time.time()-start,'ordered_ID_sha256':sc.digest([q['id'] for q in pub])});cache.close()

def quality():
    check_sources();assert (R/'scientific_frozen.json').exists();jobs=[(ds,'calibration',0,1) for ds in ['2wiki','hotpotqa','musique']]
    jobs += [(ds,'effectiveness',j,n) for ds,n in [('2wiki',6),('hotpotqa',4),('musique',2)] for j in range(n)]
    active=[];done=[]
    while jobs or active:
        while jobs and len(active)<2:
            ds,role,j,n=jobs.pop(0);name=f'{ds}_{role}_{j}';p=R/'parts'/ds/role/f'part{j}'/'complete.json'
            if p.exists():done.append(name);continue
            log=(R/(name+'.log')).open('ab',buffering=0);proc=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'worker',ds,role,str(j),str(n)],cwd=R,stdout=log,stderr=subprocess.STDOUT);active.append((proc,log,name))
        for item in active.copy():
            proc,log,name=item;rc=proc.poll()
            if rc is None:continue
            log.close();active.remove(item)
            if rc!=0:
                for other,other_log,other_name in active:other.terminate();other_log.close()
                raise RuntimeError(('quality worker failed; only owned sibling workers terminated',name,rc))
            done.append(name)
        sc.write(R/'quality_status.json',{'active':[x[2] for x in active],'finished':done,'waiting':jobs,'unix':time.time()});time.sleep(2)
    (R/'aggregates').mkdir(exist_ok=True);receipts={}
    for ds in ['2wiki','hotpotqa','musique']:
        for role in ['calibration','effectiveness']:
            _,_,pub,labels,mode=sc.data_inputs(ds,role);roots=sorted((R/'parts'/ds/role).glob('part*'));ids=[];combined=empty_sums();checks=0;budgets=0
            for root in roots:
                rr=sc.load(root/'complete.json');assert rr['status']=='complete';ids+=sc.load(root/'ordered_ids.json');checks+=rr['independent_numeric_checks'];budgets+=rr['budget_checks'];ss=sc.load(root/'summary_sums.json')
                for m,v in ss.items():
                    dst=combined[m];dst['n']+=v['n']
                    for k,fields in v['at_k'].items():
                        for f,z in fields.items():dst['at_k'][k][f]['sum']+=z['sum'];dst['at_k'][k][f]['n']+=z['n']
                    for f,z in v['stats'].items():d=dst['stats'].setdefault(f,{'sum':0.,'n':0});d['sum']+=z['sum'];d['n']+=z['n']
            assert len(ids)==len(set(ids))==len(pub) and set(ids)=={q['id'] for q in pub}
            summary={m:{'n':v['n'],'at_k':{k:{f:(z['sum']/z['n'] if z['n'] else None) for f,z in ff.items()} for k,ff in v['at_k'].items()},'valid_n':{f:v['at_k']['50'][f]['n'] for f in FIELDS},'stats':{f:z['sum']/z['n'] for f,z in v['stats'].items()}} for m,v in combined.items()}
            sc.write(R/'aggregates'/f'{ds}_{role}.json',{'summary':summary,'questions':len(pub),'conditions':66,'ordered_ID_sha256':sc.digest([q['id'] for q in pub]),'independent_numeric_checks':checks,'budget_checks':budgets})
            receipts[ds+'_'+role]={'questions':len(pub),'conditions':66,'independent_numeric_checks':checks,'budget_checks':budgets}
    for ds in ['2wiki','hotpotqa','musique']:
        db=sqlite3.connect(R/'cache'/f'{ds}.sqlite',timeout=90);db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    sc.write(R/'quality_complete.json',{'status':'complete','version':'FBSCOPE-R2','datasets':receipts,'new_main_conditions':64,'reference_conditions':2,'same_effect_manifests_all_methods':True,'quality_device':'GPU FP32; query batch1; TF32off; route search CPU8','K':sc.KS});sc.progress('quality_complete')

if __name__=='__main__':
    import torch
    torch.set_num_threads(8);torch.manual_seed(20261107);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    stage=sys.argv[1]
    try:
        if stage=='setup':setup()
        elif stage=='calibrate':calibrate()
        elif stage=='preflight':preflight()
        elif stage=='quality':quality()
        elif stage=='worker':worker(sys.argv[2],sys.argv[3],int(sys.argv[4]),int(sys.argv[5]))
    except Exception:sc.write(R/(stage+'_failed.json'),{'traceback':traceback.format_exc(),'unix':time.time()});raise
