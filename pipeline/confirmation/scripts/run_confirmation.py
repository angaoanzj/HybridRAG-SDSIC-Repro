"""Frozen confirmation; read original artifacts, write only this isolated run."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
from pathlib import Path
import collections, datetime, gc, gzip, hashlib, json, random, shutil, sys, time, traceback
import joblib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/root/autodl-tmp/hybridrag-qsel-rank-p1-1-full-20261107')
DATA = Path('/root/autodl-tmp/hybridrag-adaptive-v4-20261107')
MODEL = Path('/root/autodl-tmp/hybridrag-multiround-v2-20261107/models/bge')
sys.path.insert(0, str(ROOT / 'original_code'))
import pilot_core as c
import full_train as f
POLICIES = ['no_second_branch', 'mix_ranked16']
ALL_K = list(range(1, 51))

def sha(p): return c.file_sha256(p)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p, obj):
    p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    c.write_json(p, obj)
def log(stage, **kw):
    x = {'stage': stage, 'utc': now(), **kw}
    put('status.json', x)
    print(json.dumps(x), flush=True)
def backend(ds):
    b = c.RetrievalBackend(DATA / f'data/{ds}/corpus.jsonl', DATA / f'indexes/{ds}', MODEL, threads=3, device='cpu', cache_queries=True)
    old = json.loads((OLD / 'protocol.json').read_text())
    assert b._manifest['corpus_sha256'] == old['datasets'][ds]['corpus_sha256']
    feat = c.Features(b); feat.alpha = c.ALPHA[ds]
    return b, c.Feedback(b), feat, c.Gold(b, ds)

def preflight():
    contract = json.loads((ROOT / 'frozen/protocol.json').read_text())
    assert sha(ROOT / 'frozen/protocol.json') == (ROOT / 'frozen/protocol.sha256').read_text().strip()
    for name, digest in contract['frozen_files'].items(): assert sha(ROOT / 'frozen' / name) == digest
    assert sha(OLD / 'protocol.json') == contract['original_protocol_sha256']
    old = json.loads((OLD / 'protocol.json').read_text())
    snapshots = {}
    for ds in ['2wiki', 'hotpotqa', 'musique']:
        source = DATA / f'data/{ds}/questions.json'
        assert sha(source) == old['datasets'][ds]['source_questions_sha256']
        snapshots[ds] = {'source_questions_sha256': sha(source),
            'corpus_sha256': sha(DATA / f'data/{ds}/corpus.jsonl'),
            'index_manifest_sha256': sha(DATA / f'indexes/{ds}/index_manifest.json'),
            'old_ranker_sha256': sha(OLD / f'models/{ds}/document_ranker.joblib'),
            'old_frozen_sha256': sha(OLD / f'models/{ds}/frozen.json')}
        assert snapshots[ds]['corpus_sha256'] == old['datasets'][ds]['corpus_sha256']
        frozen = json.loads((OLD / f'models/{ds}/frozen.json').read_text())
        assert snapshots[ds]['old_ranker_sha256'] == frozen['ranker_sha256']
        if ds != 'musique':
            dest = ROOT / f'models/{ds}'; dest.mkdir(parents=True, exist_ok=True)
            for name in ['document_ranker.joblib', 'frozen.json']:
                target = dest / name
                if target.exists(): assert sha(target) == sha(OLD / f'models/{ds}/{name}')
                else: shutil.copy2(OLD / f'models/{ds}/{name}', target)
    ids = json.loads((ROOT / 'frozen/musique_ids.json').read_text())
    by = {q['id']: q for q in json.loads((DATA / 'data/musique/questions.json').read_text())['queries']}
    groups = json.loads((ROOT / 'frozen/musique_selected_groups.json').read_text())
    group_map = {i: g['group_id'] for g in groups for i in g['train_canonical_ids']}
    qs = [{**by[i], 'split': 'confirmation', 'group_id': group_map[i], 'support_count': len(by[i]['gold_paragraph_ids'])} for i in ids['confirmation']]
    dest = ROOT / 'inputs/musique_questions.json'; dest.parent.mkdir(exist_ok=True)
    text = json.dumps(qs, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if dest.exists(): assert dest.read_text() == text
    else: dest.write_text(text)
    put('verification/preflight.json', {'status': 'PASS', 'utc': now(), 'protocol_sha256': sha(ROOT / 'frozen/protocol.json'),
        'sources': snapshots, 'musique_confirmation_questions_sha256': sha(dest),
        'original_code_sha256': {p.name: sha(p) for p in sorted((ROOT / 'original_code').glob('*.py'))}})
    return by, ids

def shard_segments(path, by):
    meta = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == meta['sha256']
    z = np.load(path); dx, dy = z['dx'], z['dy']; x = z['x']
    assert len(x) == len(meta['ids']) * 66
    assert int((dy == 0).sum()) == len(meta['ids']) * 72
    assert set(np.unique(dy)) <= {0, 1}
    offset, segments = 0, []
    for j, qid in enumerate(meta['ids']):
        q = by[qid]
        assert len(q['gold_paragraph_ids']) <= 4
        # This field belongs to each original question, before any row filtering.
        assert np.all(x[j*66:(j+1)*66, len(c.ROUTES)+2] == np.float32(len(c.words(q['question']))/60))
        spans = []
        for policy in f.TRAIN_ROUTES:
            start = offset
            while offset < len(dy) and dy[offset] == 1: offset += 1
            npos = offset - start
            assert npos <= len(q['gold_paragraph_ids']) and np.all(dy[offset:offset+24] == 0)
            stop = offset + 24
            ranks = np.rint(1. / dx[start:stop, 0]).astype(int)
            assert np.array_equal(dx[start:stop, 0], np.array([1./r for r in ranks], np.float32))
            assert len(set(ranks.tolist())) == len(ranks)
            pos_ranks = set(ranks[:npos].tolist())
            expected_hard = [r for r in range(1, 17) if r not in pos_ranks][:12]
            assert ranks[npos:npos+12].tolist() == expected_hard
            assert all(r > expected_hard[-1] for r in ranks[npos+12:])
            spans.append({'policy': policy, 'start': start, 'end': stop, 'positives': npos})
            offset = stop
        segments.append({'id': qid, 'spans': spans})
    assert offset == len(dy) == meta['document_rows']
    return meta, dx, dy, segments

def filter_training(by, ids):
    done = ROOT / 'training/filter_complete.json'
    if done.exists():
        d = json.loads(done.read_text())
        for r in d['shards']: assert sha(ROOT / r['path']) == r['sha256']
        return
    exclusion = set(ids['confirmation']) | set(ids['buffer'])
    seen, kept, dropped, receipts = [], [], [], []
    root = ROOT / 'training'; root.mkdir(exist_ok=True)
    with gzip.open(root / 'original_question_row_map.jsonl.gz', 'wt') as mapfile:
        for path in sorted((OLD / 'supervision/musique').glob('train_*.npz')):
            meta, dx, dy, segments = shard_segments(path, by)
            seen += meta['ids']; keep = np.zeros(len(dy), bool); row_qids=[]
            kept_ids = []
            for seg in segments:
                excluded = seg['id'] in exclusion
                mapfile.write(json.dumps({'source_shard': path.name, **seg, 'excluded': excluded}) + '\n')
                if excluded: dropped.append(seg['id'])
                else:
                    kept.append(seg['id']); kept_ids.append(seg['id'])
                    for span in seg['spans']:
                        keep[span['start']:span['end']] = True
                        row_qids += [seg['id']] * (span['end'] - span['start'])
            target = root / path.name
            np.savez_compressed(target, dx=dx[keep], dy=dy[keep], question_ids=np.array(row_qids))
            r = {'path': str(target.relative_to(ROOT)), 'sha256': sha(target), 'source_sha256': meta['sha256'],
                 'source_question_n': len(meta['ids']), 'retained_ids': kept_ids, 'retained_rows': int(keep.sum()), 'excluded_rows': int((~keep).sum())}
            put(str(target.with_suffix('.json').relative_to(ROOT)), r); receipts.append(r)
    oldids = json.loads((OLD / 'protocol.json').read_text())['datasets']['musique']['ids']['train']
    assert seen == oldids and kept == ids['train'] and set(dropped) == exclusion
    assert not set(kept) & (exclusion | set(ids['development']))
    put('training/filter_complete.json', {'status': 'PASS', 'utc': now(), 'old_questions': len(seen), 'retained_questions': len(kept),
        'excluded_questions': len(dropped), 'excluded_confirmation': len(ids['confirmation']), 'excluded_buffer': len(ids['buffer']),
        'retained_rows': sum(r['retained_rows'] for r in receipts), 'excluded_rows': sum(r['excluded_rows'] for r in receipts),
        'mapping_proof': 'Original ID order + 3 routes per question + positive prefix <=4 + exactly24 negatives per route; all rows consumed; query-length feature and hard-negative rank sequence checked for every question/route',
        'row_map_sha256': sha(root / 'original_question_row_map.jsonl.gz'), 'shards': receipts})
    log('training_filtered', retained_questions=len(kept), excluded_questions=len(dropped))

def audit_sampler(b, fb, feat, gold, by):
    target = ROOT / 'verification/sampler_regeneration.json'
    if target.exists(): assert json.loads(target.read_text())['status'] == 'PASS'; return
    path = sorted((OLD / 'supervision/musique').glob('train_*.npz'))[0]
    meta, dx, dy, segments = shard_segments(path, by)
    qs = [by[i] for i in meta['ids'][:64]]
    bulk = f.FastBulk(b)
    initials, plans, unique = f.plan_batch(qs, bulk, fb, c.ALPHA['musique'])
    checks, exact, maximum = 0, 0, 0.
    for q, initial, exps, seg in zip(qs, initials, plans, segments):
        good = gold.good(q)
        for span in seg['spans']:
            p = span['policy']; text = exps[p]
            calls = initial + (unique[text] if text and text in unique else [])
            rank = c.fused_rank(calls, c.ALPHA['musique'])
            pos = [j for j, i in enumerate(rank) if i in good][:12]
            neg = [j for j, i in enumerate(rank) if i not in good]
            seed = f.SEED + int(hashlib.sha256((q['id']+p).encode()).hexdigest()[:8], 16)
            sample = pos + neg[:12] + random.Random(seed).sample(neg[12:], min(12, len(neg[12:])))
            a = feat.doc(c.public(q), calls, [rank[j] for j in sample]); a[:, 0] = [1/(j+1) for j in sample]
            y = np.array([int(rank[j] in good) for j in sample], np.uint8)
            ref = dx[span['start']:span['end']]
            assert np.array_equal(y, dy[span['start']:span['end']]) and a.shape == ref.shape
            assert np.array_equal(a[:,0], ref[:,0]), 'sampled rank mismatch'
            diff = float(np.max(np.abs(a-ref))); maximum = max(maximum, diff)
            assert np.allclose(a, ref, atol=5e-6, rtol=0), ('feature mismatch', q['id'], p, diff)
            exact += int(np.array_equal(a, ref)); checks += 1
    bulk.close()
    put('verification/sampler_regeneration.json', {'status':'PASS', 'utc':now(), 'questions':64, 'policy_blocks':checks,
        'bit_exact_blocks':exact, 'max_absolute_feature_difference':maximum, 'source_shard_sha256':sha(path),
        'scope':'Sampler/index mapping validation on fixed first original64; original cached features retained for fitting'})
    log('sampler_verified', blocks=checks, maximum_feature_difference=maximum)

def train_musique(gold, ids):
    dest = ROOT / 'models/musique'; dest.mkdir(parents=True, exist_ok=True)
    frozen = dest / 'frozen.json'
    if frozen.exists():
        cfg = json.loads(frozen.read_text()); assert sha(dest / 'document_ranker.joblib') == cfg['ranker_sha256']; return
    assert json.loads((ROOT / 'verification/sampler_regeneration.json').read_text())['status'] == 'PASS'
    pieces = [np.load(p) for p in sorted((ROOT / 'training').glob('train_*.npz'))]
    allowed = set(ids['train']); forbidden = set(ids['confirmation']+ids['buffer']+ids['development'])
    observed = set()
    for p in pieces:
        ss = set(p['question_ids'].tolist()); assert ss <= allowed and not ss & forbidden; observed |= ss
    assert observed == allowed
    dx = np.concatenate([p['dx'] for p in pieces]); dy = np.concatenate([p['dy'] for p in pieces])
    for p in pieces: p.close()
    log('fit_fresh_musique_ranker', training_questions=len(allowed), document_rows=len(dy), positives=int(dy.sum()))
    ranker = f.Clf(max_iter=100, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=10, early_stopping=False, random_state=f.SEED).fit(dx, dy)
    joblib.dump(ranker, dest / 'document_ranker.joblib')
    val = joblib.load(OLD / 'supervision/musique/validation.joblib')
    assert [r['q']['id'] for r in val] == ids['development']
    grid = []
    for w in [0., .25, .5]:
        value = np.mean([gold.score(c.blend(*r['docs']['mix_ranked16'], ranker, w), r['q'], k)[0] for r in val for k in c.KS])
        grid.append({'weight':w, 'development_macro_complete':float(value)})
    w = max(grid, key=lambda x:x['development_macro_complete'])['weight']
    cfg = {'utc':now(), 'version':'CONFIRM-20261110-v1', 'fit_from_scratch':True, 'warm_start':False,
           'ranker_sha256':sha(dest / 'document_ranker.joblib'), 'rerank_weight':w, 'rerank_candidates':grid,
           'training_questions':len(allowed), 'ranker_rows':len(dy), 'ranker_positive_rows':int(dy.sum()),
           'development_n':len(val), 'confirmation_used_for_selection':False,
           'training_filter_receipt_sha256':sha(ROOT / 'training/filter_complete.json')}
    put('models/musique/frozen.json', cfg)
    log('musique_model_frozen', gamma=w, ranker_sha256=cfg['ranker_sha256'])
    del dx, dy, ranker, val; gc.collect()

def support_units(q, b, gold, ds):
    good = gold.good(q)
    if ds == 'musique': return {i:[i] for i in q['gold_paragraph_ids']}
    return {t:sorted(i for i in good if b.doc_lookup[i]['title'] == t) for t in q['gold_titles']}

def first_complete(rank, units):
    position = {i:j+1 for j,i in enumerate(rank)}
    best = [min((position[i] for i in vals if i in position), default=None) for vals in units.values()]
    return None if any(x is None for x in best) else max(best)

def evaluate(ds, b, fb, feat, gold):
    cfg = json.loads((ROOT / f'models/{ds}/frozen.json').read_text())
    model = ROOT / f'models/{ds}/document_ranker.joblib'
    assert sha(model) == cfg['ranker_sha256']
    ranker = joblib.load(model); weight=cfg['rerank_weight']
    source = ROOT / (f'inputs/{ds}_questions.json' if ds == 'musique' else f'frozen/{ds}_questions.json')
    qs = json.loads(source.read_text()); assert len(qs)==1000
    cache = ROOT / f'cache/{ds}'; cache.mkdir(parents=True, exist_ok=True)
    start = time.time(); bulk=f.FastBulk(b); rows=[]
    for st in range(0, len(qs), 64):
        small=qs[st:st+64]; target=cache/f'batch_{st:06d}.joblib'; receipt=target.with_suffix('.json')
        if receipt.exists():
            meta=json.loads(receipt.read_text()); assert meta['sha256']==sha(target) and meta['ids']==[q['id'] for q in small]
            rr=joblib.load(target); rows+=rr; continue
        initial=bulk.search([q['question'] for q in small], 'q0'); plans=[]; unique={}
        for q,calls in zip(small,initial):
            r=c.fused_rank(calls,c.ALPHA[ds]); plan={p:fb.expand(q['question'],r,p) for p in POLICIES}; plans.append(plan)
            for entry in plan.values():
                text=entry.get('query')
                if text and text.strip()!=q['question'].strip():unique[text]=None
        found=bulk.search(list(unique),'q1') if unique else []; unique.update(zip(unique,found)); rr=[]
        for q, init, plan in zip(small,initial,plans):
            units=support_units(q,b,gold,ds); assert units and all(units.values()), 'fixed corpus reachability inconsistent'
            row={'q':q,'units':units,'pools':{}}
            for policy, label in zip(POLICIES, ['S2','A']):
                exp=plan[policy]; text=exp.get('query'); calls=init+(unique[text] if text and text in unique else [])
                assert len(calls) in [2,4] and all(len(ca['returned'])==100 for ca in calls)
                rank=c.fused_rank(calls,c.ALPHA[ds]); x=feat.doc(c.public(q),calls,rank); out=c.blend(rank,x,ranker,weight)
                assert len(rank)==len(set(rank)) and set(rank)==set(out)
                r0=first_complete(rank,units); r1=first_complete(out,units)
                assert (r0 is None)==(r1 is None)
                for k in ALL_K:
                    assert gold.score(rank,q,k)[0]==int(r0 is not None and r0<=k)
                    assert gold.score(out,q,k)[0]==int(r1 is not None and r1<=k)
                row['pools'][label]={'policy':policy,'calls':calls,'expansion':exp,'rank':rank,'features':x,
                    'reranked':out,'pool_complete':r0 is not None,'r_before':r0,'r_after':r1,
                    'q1_executed':len(calls)==4,'returned_records':sum(len(ca['returned']) for ca in calls)}
            rr.append(row)
        joblib.dump(rr,target,compress=3); put(str(receipt.relative_to(ROOT)),{'sha256':sha(target),'ids':[q['id'] for q in small],
            'questions_sha256':sha(source),'model_sha256':sha(model),'gamma':weight,'utc':now()})
        rows+=rr
        log('confirmation_retrieval',dataset=ds,completed=len(rows),total=1000,seconds=round(time.time()-start,1))
    bulk.close()
    assert [r['q']['id'] for r in rows]==[q['id'] for q in qs]
    out=ROOT / f'results/{ds}_per_question.jsonl.gz'
    with gzip.open(out,'wt') as h:
        for row in rows:
            q=row['q']
            for label,p in row['pools'].items():
                record={'id':q['id'],'group_id':q['group_id'],'type':q.get('type'),
                    'support_count':len(row['units']),'pool':label,
                    **{k:p[k] for k in ['pool_complete','r_before','r_after','q1_executed','returned_records']},
                    'unique_documents':len(p['rank']), 'gamma':weight,
                    'rank_sha256':hashlib.sha256(json.dumps(p['rank']).encode()).hexdigest(),
                    'reranked_sha256':hashlib.sha256(json.dumps(p['reranked']).encode()).hexdigest()}
                h.write(json.dumps(record)+'\n')
    put(f'results/{ds}_complete.json',{'status':'complete','utc':now(),'questions':len(rows),'conditions':4,
        'per_question_sha256':sha(out),'ranker_sha256':sha(model),'gamma':weight,'seconds':time.time()-start})
    log('dataset_complete',dataset=ds,questions=len(rows))

def main():
    import torch
    torch.set_num_threads(3)
    by, ids = preflight()
    filter_training(by, ids)
    for ds in ['musique','2wiki','hotpotqa']:
        if (ROOT/f'results/{ds}_complete.json').exists():continue
        log('load_backend',dataset=ds)
        b,fb,feat,gold=backend(ds)
        if ds=='musique':
            audit_sampler(b,fb,feat,gold,by)
            train_musique(gold,ids)
        evaluate(ds,b,fb,feat,gold)
        del b,fb,feat,gold;gc.collect()
    log('all_confirmation_runs_complete',questions=3000)

if __name__=='__main__':
    try:main()
    except Exception:
        put('failed.json',{'utc':now(),'traceback':traceback.format_exc()})
        log('failed',error=traceback.format_exc());raise
