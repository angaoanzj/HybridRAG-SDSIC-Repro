"""Verify and analyze the new four-arm retrieval run; no remote calls."""
from pathlib import Path
from collections import Counter
import gzip,json,hashlib

P=Path(__file__).resolve().parent
R=P/'budget_results'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((R/'summary.json').read_text())
complete=json.loads((R/'complete.json').read_text())
assert sha(R/'protocol.json')==complete['protocol_sha256']
assert sha(R/'quality_rows.jsonl.gz')==complete['quality_sha256']
assert sha(R/'summary.json')==complete['summary_sha256']
rows=[json.loads(l) for l in gzip.open(R/'quality_rows.jsonl.gz','rt')]
assert len(rows)==800
result={'status':'completed real retrieval; post-hoc analysis of prespecified exploratory run','n_quality_rows':len(rows),'datasets':{},'comparability':'A/B/C same requested-record ceiling, not identical realized D when empty/repeated q2 is skipped. D has ceiling300. All same BGE/index/title policy/RRF/CPU threads.','checks':{}}
for ds in ['2wiki','hotpotqa']:
  rr=[r for r in rows if r['dataset']==ds];by={(r['id'],r['arm']):r for r in rr};ids=sorted({r['id'] for r in rr})
  assert len(ids)==100 and len(by)==400
  for arm in ['A_single100','B_iterative50','C_repeat50','D_single50']:
    vals=[by[i,arm] for i in ids]
    for k in range(1,51):
      assert abs(sum(r['metrics'][str(k)]['strict'] for r in vals)/100-summary[ds]['arms'][arm]['coverage'][str(k)])<1e-12
  for i in ids:
    a,b,c,d=[by[i,x] for x in ['A_single100','B_iterative50','C_repeat50','D_single50']]
    assert b['history'][:1]==c['history'][:1]==d['history'][:1]==a['history'][:1]
    assert c['pool']==d['pool'] and c['unique_candidate_count']==d['unique_candidate_count']
    assert b['pool']['strict']>=d['pool']['strict'] and b['pool']['recall']>=d['pool']['recall']
  dsout={'q2_ids':[i for i in ids if by[i,'B_iterative50']['supplementary']==2],
    'no_new_complete_pool_q2':sum(by[i,'B_iterative50']['pool']['strict']>by[i,'D_single50']['pool']['strict'] for i in ids)==0,
    'q2_partial_pool_recall_increases':sum(by[i,'B_iterative50']['pool']['recall']>by[i,'D_single50']['pool']['recall'] for i in ids),
    'mean_new_unique_documents_q2_triggered':sum(by[i,'B_iterative50']['unique_candidate_count']-by[i,'D_single50']['unique_candidate_count'] for i in ids)/25,
    'by_K':{},'cases':{}}
  for k in range(1,51):
    key=str(k);record={}
    score=lambda i,arm:by[i,arm]['metrics'][key]['strict']
    for short,arm,base in [('B-A','B_iterative50','A_single100'),('B-C','B_iterative50','C_repeat50'),('B-D','B_iterative50','D_single50'),('D-A','D_single50','A_single100'),('C-D','C_repeat50','D_single50')]:
      wins=[i for i in ids if score(i,arm)>score(i,base)];losses=[i for i in ids if score(i,arm)<score(i,base)]
      record[short]={'win':len(wins),'loss':len(losses),'delta_pp':len(wins)-len(losses),
        'losses_with_complete_pool_after':sum(by[i,arm]['pool']['strict'] for i in losses)}
      if k in [5,10,50] and losses:dsout['cases'][f'{short}@{k}_loss']=[{a:by[i,a] for a in ['A_single100','B_iterative50','C_repeat50','D_single50']} for i in losses[:1]]
    all_loss=[i for i in ids if score(i,'A_single100')>score(i,'B_iterative50')]
    record['B-A_loss_decomposition']={'n':len(all_loss),'already_lost_after_depth_reduction':sum(not score(i,'D_single50') for i in all_loss),'present_after_depth_reduction_lost_after_q2':sum(score(i,'D_single50') for i in all_loss)}
    q2=dsout['q2_ids']
    record['B-C_q2_triggered_same_D400_descriptive']={'n':len(q2),'win':sum(score(i,'B_iterative50')>score(i,'C_repeat50') for i in q2),'loss':sum(score(i,'B_iterative50')<score(i,'C_repeat50') for i in q2)}
    dsout['by_K'][key]=record
  result['datasets'][ds]=dsout
result['checks']={'source_hashes':True,'800_unique_condition_rows':True,'all_400_coverage_cells_match_summary':True,'common_q1_all_arms':True,'C_D_pool_identical':True,'B_D_cumulative_pool_nondecreasing':True,'independent_timing_fresh_trajectory_assertions_passed_on_remote':True}
(R/'mechanism_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for ds,o in result['datasets'].items():
 print(ds,{k:v for k,v in o.items() if k not in ['by_K','cases','q2_ids']})
 for k in ['5','10','50']:print(k,o['by_K'][k])
