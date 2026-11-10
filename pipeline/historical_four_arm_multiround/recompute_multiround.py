"""Read-only, post-hoc audit of frozen retrieval records. No new retrieval.

Run from any directory: python3 /absolute/path/recompute_multiround.py
Outputs only into this script's directory. Standard library only.
"""
from pathlib import Path
from collections import Counter, defaultdict
import gzip, hashlib, json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCES = {}

def rows(path):
    p = ROOT / path
    SOURCES[path] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
    with gzip.open(p, 'rt') as f:
        for line in f:
            yield json.loads(line)

def stats(a, b=None):
    n = len(a)
    out = {'n': n, 'complete': sum(x['strict'] for x in a), 'coverage_percent': 100 * sum(x['strict'] for x in a) / n if n else None,
           'supplementary_count_distribution': dict(sorted(Counter(x['supp'] for x in a).items())),
           'q2_or_later_n': sum(x['supp'] >= 2 for x in a),
           'mean_total_queries': sum(x['supp'] + 1 for x in a) / n if n else None,
           'mean_nominal_requested_records': 200 * sum(x['supp'] + 1 for x in a) / n if n else None}
    if b is not None:
        assert len(a) == len(b)
        assert all(x['id'] == y['id'] for x, y in zip(a, b))
        diffs = [x['strict']-y['strict'] for x,y in zip(a,b)]
        out.update({'win':sum(d > 0 for d in diffs), 'loss':sum(d < 0 for d in diffs), 'tie':sum(d == 0 for d in diffs),
                    'delta_pp':100*sum(diffs)/n if n else None,
                    'same_nominal_budget_n':sum(x['supp']==y['supp'] for x,y in zip(a,b)),
                    'greater_nominal_budget_n':sum(x['supp']>y['supp'] for x,y in zip(a,b)),
                    'lower_nominal_budget_n':sum(x['supp']<y['supp'] for x,y in zip(a,b))})
    return out

def small_versions():
    result = {}
    jobs = [
      ('v1','multiround_controller_20261107/results/remote/test_results.jsonl.gz','fixed1'),
      ('v2','multiround_controller_v2_20261107/results/remote/confirmation_results.jsonl.gz','single'),
      *[('kgrid_'+ds,'kgrid_scaling_20261107/results/remote/evaluation_'+ds+'.jsonl.gz','single') for ds in ['2wiki','hotpotqa','musique']]]
    for version,path,base in jobs:
        groups = defaultdict(dict)
        for r in rows(path):
            supp = r['queries']-1
            assert supp == len(r['history'])
            assert 0 <= supp <= 3
            assert len(r['history']) == len(set(r['history'])), 'Repeated literal query'
            metrics = r.get('metrics')
            if isinstance(metrics, list):
                values = {str(m['k']):m['strict'] for m in metrics}
            elif metrics is not None:
                values = {k:m['strict'] for k,m in metrics.items()}
            else:
                values = {'50':r['strict']}
            for k, val in values.items():
                groups[r['dataset'],k,r['policy']][r['id']] = {'id':r['id'],'strict':val,'supp':supp,'hop':r.get('hop'),'history':r['history']}
        report = {}
        for (ds,k,policy), data in sorted(groups.items()):
            baselines = groups[ds,k,base]
            assert data.keys() == baselines.keys()
            ids = sorted(data)
            a,b = [data[i] for i in ids],[baselines[i] for i in ids]
            x = stats(a,b)
            x['q2_triggered_subset'] = stats([v for v in a if v['supp']>=2],[baselines[v['id']] for v in a if v['supp']>=2])
            x['without_q2_subset'] = stats([v for v in a if v['supp']<2],[baselines[v['id']] for v in a if v['supp']<2])
            if ds == 'musique':
                x['by_annotated_hop'] = {str(h):stats([v for v in a if v['hop']==h],[baselines[v['id']] for v in a if v['hop']==h]) for h in [2,3,4]}
            report['/'.join([ds,k,policy])] = x
        result[version] = report
    return result

def v4():
    output = {}
    keep = {'adaptive_v4','once_rrf','best_shared_rule','learned_words_fixed_budget','learned_stop_first_action'}
    for ds in ['2wiki','hotpotqa','musique']:
        grouped = defaultdict(dict)
        new_targets = Counter()
        case = None
        for r in rows(f'adaptive_controller_v4_20261107/retrieved/results/{ds}/evaluation_rows.jsonl.gz'):
            if r['name'] not in keep: continue
            q = [t for t in r['trace'] if t['decision']=='QUERY']
            assert len(q) == r['queries']
            x = {key:r[key] for key in ['id','strict','origin','gold_units','pool_strict','mode']}
            x['supp'] = r['queries']
            x['queries'] = [t['query'] for t in q]
            grouped[r['k'],r['name']][r['id']] = x
            if r['name']=='adaptive_v4' and len(q)>=2:
                new = set(t for a in q[1:] for t in a['targets']) - set(q[0]['targets'])
                new_targets[r['k'],'q2_cases'] += 1
                new_targets[r['k'],'later_new_target_cases'] += bool(new)
                if case is None and new:
                    case = {**x, 'k':r['k'], 'trace':q}
        report = {}
        for (k,name),data in sorted(grouped.items()):
            base = grouped[k,'once_rrf']; ids=sorted(data)
            a,b=[data[i] for i in ids],[base[i] for i in ids]
            z=stats(a,b)
            z['by_support_units']={str(h):stats([v for v in a if v['gold_units']==h],[base[v['id']] for v in a if v['gold_units']==h]) for h in sorted(set(v['gold_units'] for v in a))}
            if name=='adaptive_v4':
                other=grouped[k,'best_shared_rule']
                z['vs_best_shared_rule']=stats(a,[other[i] for i in ids])
                z['q2_triggered_vs_once_rrf']=stats([v for v in a if v['supp']>=2],[base[v['id']] for v in a if v['supp']>=2])
                z['q2_triggered_vs_shared']=stats([v for v in a if v['supp']>=2],[other[v['id']] for v in a if v['supp']>=2])
                z['later_new_target_cases']=new_targets[k,'later_new_target_cases']
                z['same_actual_query_count_vs_shared_descriptive']=stats([v for v in a if v['supp']==other[v['id']]['supp']],[other[v['id']] for v in a if v['supp']==other[v['id']]['supp']])
            report[f'{k}/{name}']=z
        output[ds]={'summary':report,'example_later_new_targets':case}
    return output

def main():
    out={'analysis_date':'2026-11-07','status':'new offline descriptive reanalysis of already completed frozen runs; zero retrieval calls; zero training',
      'budget_definition':'Every logical query requests BM25 top-100 and BGE top-100. Nominal requested records = 200*(1+supplementary queries); actual returned unique document count differs.',
      'selection_warning':'Subsets conditioned on realized query count are post-treatment descriptive groups, not randomized matched-budget causal effects.',
      'fixed_budget_multiround_comparison_found':False,
      'small_versions':small_versions(),'v4':v4(),'sources':SOURCES}
    (OUT/'recomputed_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT/'recomputed_results.json'),'source_files':len(SOURCES),'bytes':(OUT/'recomputed_results.json').stat().st_size}))

if __name__=='__main__':main()
