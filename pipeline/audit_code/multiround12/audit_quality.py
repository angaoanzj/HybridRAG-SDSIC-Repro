from pathlib import Path
import collections, gzip, hashlib, json, math, random

root=Path('/path/to/local/home/Downloads/hybridrag-musique-multiround-results-20261109T074918Z-151828')
out=Path('/tmp/hybridrag_multiround_audit_20261110')
protocol=json.loads((root/'protocol.json').read_text())
summary=json.loads((root/'summary.json').read_text())
complete=json.loads((root/'complete.json').read_text())
with gzip.open(root/'quality_rows.jsonl.gz','rt') as f:
    rows=[json.loads(line) for line in f if line.strip()]
errors=[]; checks=collections.Counter()
def check(ok,name,detail=None):
    checks[name]+=1
    if not ok: errors.append({'check':name,'detail':detail})
def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict): return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return a==b
hashes={}
for name,key in [('protocol.json','protocol_sha256'),('quality_rows.jsonl.gz','quality_sha256'),('summary.json','summary_sha256'),('raw_rankings_and_calls.jsonl.gz','raw_sha256')]:
    h=hashlib.sha256()
    with (root/name).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    hashes[name]=h.hexdigest(); check(h.hexdigest()==complete[key],'file_hash_matches_completion',name)
ids=protocol['question_ids']; arms=protocol['arms']; ks=protocol['K']; n=len(ids)
check(n==2417 and len(set(ids))==2417,'protocol_unique_ids')
check(len(arms)==9 and len(rows)==21753,'expected_dimensions')
by={(r['id'],r['arm']):r for r in rows}
check(len(by)==len(rows),'unique_query_arm_rows')
check(set(by)=={(qid,arm) for qid in ids for arm in arms},'exact_cartesian_product')
check(complete['status']=='COMPLETED' and summary['status']=='COMPLETED','completion_status')
check((complete['questions'],complete['arms'],complete['rows'])==(n,len(arms),len(rows)),'completion_dimensions')
check(ks==[10,20,30,50] and protocol['D_cap']==400,'cutoffs_and_budget_cap')
for qi,qid in enumerate(ids):
    group=rows[qi*9:(qi+1)*9]
    expected_order=list(arms); random.Random(2026110907+qi).shuffle(expected_order)
    check([r['id'] for r in group]==[qid]*9,'ordered_question_groups',qid)
    check([r['arm'] for r in group]==expected_order,'seeded_arm_order',qid)
    check(len({r['hops'] for r in group})==1,'same_hops_across_arms',qid)
    check(len({tuple(sorted(r['gold_ranks'])) for r in group})==1,'same_gold_ids_across_arms',qid)
for r in rows:
    name=f"{r['id']}:{r['arm']}"; steps=r['executed_query_instances']; spec=arms[r['arm']]
    check(1<=steps<=len(spec['depths']),'executed_round_range',name)
    check(r['supplementary_queries']==steps-1 and r['physical_route_calls']==2*steps,'round_call_arithmetic',name)
    check(r['requested_records']==2*sum(spec['depths'][:steps]),'requested_depth_arithmetic',name)
    check(0<=r['returned_records']<=r['requested_records']<=400,'budget_bound',name)
    check(all(len(r[x])==steps for x in ['round_titles','new_title_counts','trajectory_digests','metrics_after_round']),'round_array_alignment',name)
    seen=set()
    for targets,new_n in zip(r['round_titles'],r['new_title_counts']):
        check(new_n==len(set(targets)-seen),'new_title_counts',name); seen.update(targets)
    check(r['at_k']==r['metrics_after_round'][-1]['at_k'] and r['gold_ranks']==r['metrics_after_round'][-1]['gold_ranks'],'final_matches_last_round',name)
    for metrics in r['metrics_after_round']:
        ranks=list(metrics['gold_ranks'].values())
        check(len(ranks)==r['hops'] and len(ranks)>0,'gold_count_matches_hops',name)
        for k in ks:
            m=metrics['at_k'][str(k)]; hits=sum(x is not None and x<=k for x in ranks)
            check(m['FP']==int(hits==len(ranks)) and math.isclose(m['paragraph_recall'],hits/len(ranks),abs_tol=1e-12),'FP_recall_from_gold_ranks',name)
            check(m['FP'] in (0,1) and m['FT'] in (0,1) and m['FP']<=m['FT'],'binary_metric_and_FP_le_FT',name)
        for metric in ['FP','FT','paragraph_recall']:
            values=[metrics['at_k'][str(k)][metric] for k in ks]
            check(values==sorted(values),'metrics_monotonic_over_K',name)
arm_summary={}; detail={}
for arm in arms:
    selection=[by[qid,arm] for qid in ids]
    check([r['id'] for r in rows if r['arm']==arm]==ids,'ordered_ids_per_arm',arm)
    arm_summary[arm]={'n':n,'supplementary_distribution':dict(collections.Counter(str(r['supplementary_queries']) for r in selection)),
        'mean_physical_route_calls':sum(r['physical_route_calls'] for r in selection)/n,
        'mean_returned_records':sum(r['returned_records'] for r in selection)/n,
        'at_k':{str(k):{m:sum(r['at_k'][str(k)][m] for r in selection)/n for m in ['FP','FT','paragraph_recall']} for k in ks}}
    check(equal(arm_summary[arm],summary['arms'][arm]),'recomputed_arm_summary',arm)
    detail[arm]={'FP_counts':{str(k):sum(r['at_k'][str(k)]['FP'] for r in selection) for k in ks},
        'FT_counts':{str(k):sum(r['at_k'][str(k)]['FT'] for r in selection) for k in ks},
        'stop_reasons':dict(collections.Counter(r['stop_reason'] for r in selection)),
        'requested_record_distribution':dict(collections.Counter(str(r['requested_records']) for r in selection)),
        'returned_equals_requested_n':sum(r['returned_records']==r['requested_records'] for r in selection),
        'hops':{str(h):{'n':sum(r['hops']==h for r in selection),'FP_counts':{str(k):sum(r['at_k'][str(k)]['FP'] for r in selection if r['hops']==h) for k in ks}} for h in sorted(set(r['hops'] for r in selection))}}
paired={}
for comparison,interpretation in protocol['comparisons'].items():
    a,b=comparison.split('-minus-')
    paired[comparison]={'interpretation':interpretation,'same_realized_D_n':sum(by[q,a]['returned_records']==by[q,b]['returned_records'] for q in ids),'at_k':{}}
    for k in ks:
        diffs=[by[q,a]['at_k'][str(k)]['FP']-by[q,b]['at_k'][str(k)]['FP'] for q in ids]
        paired[comparison]['at_k'][str(k)]={'wins':sum(x>0 for x in diffs),'losses':sum(x<0 for x in diffs),'delta_pp':100*sum(diffs)/n}
    check(equal(paired[comparison],summary['paired'][comparison]),'recomputed_paired_summary',comparison)
prefix_pairs=[('iter2_100_50_50','stop100_50'),('repeat2_100_50_50','stop100_50'),('stop50_50_50','stop50_50'),('iter3_50_50_50_50','stop50_50_50'),('repeat3_50_50_50_50','stop50_50_50')]
prefix_results={}
for a,b in prefix_pairs:
    extension_ids=[]
    for q in ids:
        ra,rb=by[q,a],by[q,b]; m=len(rb['trajectory_digests'])
        check(ra['trajectory_digests'][:m]==rb['trajectory_digests'],'paired_trajectory_prefix',f'{q}:{a}:{b}')
        check(ra['metrics_after_round'][:m]==rb['metrics_after_round'],'paired_metric_prefix',f'{q}:{a}:{b}')
        check(ra['round_titles'][:m]==rb['round_titles'],'paired_title_prefix',f'{q}:{a}:{b}')
        if ra['executed_query_instances']>rb['executed_query_instances']: extension_ids.append(q)
        else: check(ra['at_k']==rb['at_k'] and ra['returned_records']==rb['returned_records'],'no_continuation_unchanged',f'{q}:{a}:{b}')
    prefix_results[a+'-minus-'+b]={'continued_n':len(extension_ids),'continued_FP50_wins':sum(by[q,a]['at_k']['50']['FP']>by[q,b]['at_k']['50']['FP'] for q in extension_ids),'continued_FP50_losses':sum(by[q,a]['at_k']['50']['FP']<by[q,b]['at_k']['50']['FP'] for q in extension_ids)}
anchors={a:detail[a]['FP_counts']['50'] for a in ['RRF200','single100']}
check(anchors=={'RRF200':1130,'single100':1358}==summary['anchor_counts'],'anchor_reproduction')
original_path=Path('/Volumes/algorithm/HybridRAG_workspace/rag返修2/musique_extension_20261107/data/musique/public.json')
original_data=original_path.read_bytes()
original_ids=[q['id'] for q in json.loads(original_data)]
original_public_verification={'path':str(original_path),'sha256':hashlib.sha256(original_data).hexdigest(),'matches_protocol_original_public_sha256':hashlib.sha256(original_data).hexdigest()==protocol['original_manifest']['public_sha256'],'ids_and_order_match':original_ids==ids,'n':len(original_ids)}
check(original_public_verification['matches_protocol_original_public_sha256'] and original_public_verification['ids_and_order_match'],'original_public_input_identity')
result={'status':'PASS_SAVED_QUALITY_AUDIT' if not errors else 'FAIL','input_directory':str(root),'scope':'Saved-output hash, structure, arithmetic and summary audit only. No retrieval/model execution. FT checked for internal consistency and aggregation; semantic correctness requires raw rankings and corpus/title mapping.','original_public_verification':original_public_verification,'n_questions':n,'n_arms':len(arms),'n_rows':len(rows),'hashes':hashes,'checks':dict(checks),'total_checks':sum(checks.values()),'failures':errors,'recomputed_arms':arm_summary,'arm_detail':detail,'recomputed_paired':paired,'prefix_checks':prefix_results,'anchor_counts':anchors,'single_minus_RRF_FP50_pp':100*(anchors['single100']-anchors['RRF200'])/n,'caveats':['protocol.json status remains RUNNING as an immutable launch record; complete.json and summary.json record COMPLETED.','No inferential significance claim: protocol defines descriptive fixed public-development comparisons.','D<=400 is a common cap, not equal actual record or CPU cost.','Current pasted/summary rounded percentages can differ by 0.01 pp from exact count-based differences.']}
(out/'quality_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=[result['status'],f'{n} questions x {len(arms)} arms = {len(rows)} unique rows; {sum(checks.values())} checks, {len(errors)} failures.',f'Anchors: {anchors}; single-minus-RRF FP50 = {result["single_minus_RRF_FP50_pp"]:.8f} pp.']
for arm,d in arm_summary.items():
    lines.append(f'{arm}: FP50 {detail[arm]["FP_counts"]["50"]}/{n}={100*d["at_k"]["50"]["FP"]:.6f}%; calls={d["mean_physical_route_calls"]:.6f}; records={d["mean_returned_records"]:.6f}; supplementary={d["supplementary_distribution"]}; stop={detail[arm]["stop_reasons"]}')
for name,d in paired.items():
    lines.append(f'{name}: FP50 {d["at_k"]["50"]}; same_D={d["same_realized_D_n"]}')
lines+=result['caveats']
lines.append('Original public input: '+json.dumps(original_public_verification,ensure_ascii=False))
(out/'findings.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
if errors: print(json.dumps(errors[:10],ensure_ascii=False))
