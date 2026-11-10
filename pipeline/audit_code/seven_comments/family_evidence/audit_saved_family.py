"""Read archived 600-question family/score records; no retrieval or inference."""
from pathlib import Path
import collections
import csv
import datetime
import gzip
import hashlib
import json
import math
import shutil

OUT = Path(__file__).resolve().parent
BASE = Path('/path/to/local/home/.codex/visualizations/2026/11/07/01a1b6f1-912d-7e41-9fce-a1cec97f186a')
CURRENT = BASE / 'HybridRAG_排除人工词表_20261109'
READER = CURRENT / 'reader_18条件_去人工缓存'
CODE = BASE / 'HybridRAG_原题集_固定索引_修正复跑_20261108/code'
OLD = Path('/Volumes/algorithm/HybridRAG_workspace/rag返修2/reviewer_resolution_20261107/order_family')
OLD_PROTOCOL = Path('/path/to/local/home/Downloads/返修版本/studies/full_nontraining_20261106/collected/protocol.json')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def exact(w, l):
    n = w + l
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w,l)+1)) / 2**n) if n else 1.0

sources = {
    'current/coverage_summary_18.json': CURRENT / 'analysis/coverage_summary_18.json',
    'current/coverage_view_receipt.json': CURRENT / 'analysis/coverage_view_receipt.json',
    'current/answer_summary.json': READER / 'analysis/scored_answers/summary.json',
    'current/paired_EM_contrasts.json': READER / 'analysis/scored_answers/paired_contrasts.json',
    'current/retrieval_protocol.json': READER / 'source_archive/previous_all19_retrieval_protocol.json',
    'current/reader_analysis_plan.json': BASE / 'HybridRAG_获取与排序分解_五页稿_20261109/配套材料/reader_analysis_plan.json',
    'current/cue_config.json': BASE / 'HybridRAG_获取与排序分解_五页稿_20261109/配套材料/cue_config.json',
    'current/manual_cue_exclusion.md': CURRENT / '排除人工词表_处理与结果说明.md',
    'historical_manual/full_historical_results.json': OLD / 'full_historical_results.json',
    'historical_manual/mechanism_summary.json': OLD / 'full_family_mechanism/summary.json',
    'historical_manual/complete.json': OLD / 'full_family_mechanism/complete.json',
    'historical_manual/protocol.json': OLD_PROTOCOL,
    'historical_manual/frozen_independent_rules.py': OLD / 'full_family_mechanism/frozen_independent_rules.py',
    'historical_manual/analyze_remote.py': OLD / 'full_family_mechanism/analyze_remote.py',
}
for name in ['expansion_variants.py','cue_runtime.py','frozen_entity_variants.py','frozen_order_variants.py']:
    sources['current/code/' + name] = CODE / name
manifest = {}
for name, p in sources.items():
    dest = OUT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert sha(dest) == sha(p), f'Existing evidence differs: {name}'
    else:
        shutil.copy2(p, dest)
    manifest[name] = {'source': str(p), 'sha256':sha(p), 'bytes':p.stat().st_size, 'copy_sha256_matches':sha(dest)==sha(p)}

protocol = read(sources['current/retrieval_protocol.json'])
coverage_summary = read(sources['current/coverage_summary_18.json'])
assert sha(sources['current/retrieval_protocol.json']) == coverage_summary['protocol_sha256']
for name in ['expansion_variants.py','cue_runtime.py','frozen_entity_variants.py','frozen_order_variants.py']:
    assert sha(CODE/name) == protocol['code_sha256'][name]
assert sha(sources['current/cue_config.json']) == protocol['cue_config_sha256']

paths = {
    'coverage': CURRENT / 'analysis/coverage_per_question_18.jsonl.gz',
    'answers': READER / 'analysis/scored_answers/per_question_answers.jsonl.gz',
}
maps = {}
for label, p in paths.items():
    keep = {}; total=0
    with gzip.open(p, 'rt') as stream:
        for line in stream:
            row=json.loads(line);total+=1
            if row['policy'] not in ['family_u24','score2_u24']:
                continue
            key=row['dataset'],row['id'],row['K'],row['policy']
            assert key not in keep
            keep[key]=row
    assert total==43200 and len(keep)==4800
    maps[label]=keep
    manifest['source_only/'+label]={'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'rows':total,'retained_rows':len(keep),'copied':False}
cov,ans=maps['coverage'],maps['answers'];assert cov.keys()==ans.keys()
em_contrasts=read(sources['current/paired_EM_contrasts.json'])
records=[];summaries=[]
for ds in ['2wiki','hotpotqa']:
    ids=sorted({qid for d,qid,k,p in cov if d==ds});assert len(ids)==300
    for k in [10,20,30,50]:
        c=collections.Counter();cross=collections.Counter();gain_answer=collections.Counter()
        for qid in ids:
            s=cov[ds,qid,k,'score2_u24'];f=cov[ds,qid,k,'family_u24'];sa=ans[ds,qid,k,'score2_u24'];fa=ans[ds,qid,k,'family_u24']
            query=s['expanded_query']!=f['expanded_query'];first=s['selected_titles'][:1]!=f['selected_titles'][:1]
            c['n']+=1;c['query_changed']+=query;c['first_title_changed']+=first;c['candidate_distinct_count_changed']+=s['candidate_pool_distinct_count']!=f['candidate_pool_distinct_count']
            c['selected_title_count_changed']+=len(s['selected_titles'])!=len(f['selected_titles'])
            c['selected_title_sequence_changed']+=s['selected_titles']!=f['selected_titles']
            c['selected_document_list_changed']+=s['selected_ids']!=f['selected_ids']
            c['input_key_changed']+=sa['input_key']!=fa['input_key'];c['prediction_text_changed']+=sa['prediction']!=fa['prediction']
            c['logical_calls_changed']+=s['j_count']!=f['j_count'];c['returned_record_budget_changed']+=s['d_count']!=f['d_count']
            outcomes={}
            for label,a,b in [('FT',s['F'],f['F']),('strictFT',s['strictF'],f['strictF']),('pool_FT',s['acquired']['F'],f['acquired']['F']),('pool_strictFT',s['acquired']['strictF'],f['acquired']['strictF']),('EM',sa['EM'],fa['EM'])]:
                outcome='wins' if b>a else 'losses' if b<a else 'ties'
                outcomes[label]=outcome;c[label+'_score_successes']+=a;c[label+'_family_successes']+=b;c[label+'_'+outcome]+=1
                if query:c['query_changed_'+label+'_'+outcome]+=1
            cross[outcomes['FT']+'/'+outcomes['EM']]+=1
            if outcomes['FT']=='wins':
                c['FT_wins_reference_pool_missing']+=not s['acquired']['F']
                c['FT_wins_reference_pool_complete']+=bool(s['acquired']['F'])
                gain_answer[str(int(sa['EM']))+'->'+str(int(fa['EM']))]+=1
            if outcomes['FT']=='losses':c['FT_losses_family_pool_complete']+=bool(f['acquired']['F'])
            if not query:
                assert sa['input_key']==fa['input_key'] and sa['prediction']==fa['prediction']
                assert s['selected_ids']==f['selected_ids'] and s['F']==f['F'] and s['strictF']==f['strictF']
            records.append({'dataset':ds,'id':qid,'K':k,'candidate_count_score':s['candidate_pool_distinct_count'],'candidate_count_family':f['candidate_pool_distinct_count'],'first_title_changed':first,'selected_title_count_changed':len(s['selected_titles'])!=len(f['selected_titles']),'selected_titles_changed':s['selected_titles']!=f['selected_titles'],'query_changed':query,'selected_documents_changed':s['selected_ids']!=f['selected_ids'],'logical_calls_score':s['j_count'],'logical_calls_family':f['j_count'],'returned_records_score':s['d_count'],'returned_records_family':f['d_count'],'score_pool_FT':s['acquired']['F'],'family_pool_FT':f['acquired']['F'],'score_pool_strictFT':s['acquired']['strictF'],'family_pool_strictFT':f['acquired']['strictF'],'score_FT':s['F'],'family_FT':f['F'],'score_strictFT':s['strictF'],'family_strictFT':f['strictF'],'score_EM':sa['EM'],'family_EM':fa['EM'],'score_F1':sa['F1'],'family_F1':fa['F1'],'score_input_key':sa['input_key'],'family_input_key':fa['input_key'],'prediction_text_changed':sa['prediction']!=fa['prediction']})
        saved=next(v for v in em_contrasts if v['dataset']==ds and v['K']==k and v['contrast']=='family_vs_score')
        em=saved['metrics']['EM']
        assert all(c['EM_'+v]==em[v] for v in ['wins','losses','ties'])
        assert exact(c['EM_wins'],c['EM_losses'])==saved['EM_exact_mcnemar_p']
        orig_cov=None
        if k==50:
            oc=next(v for v in coverage_summary['contrasts'] if v['dataset']==ds and v['contrast']=='family_vs_score')
            for label,metric in [('FT','F'),('strictFT','strictF')]:
                assert all(c[label+'_'+v]==oc['metrics'][metric][v] for v in ['wins','losses','ties'])
            assert c['query_changed']==oc['query_changed'] and c['selected_document_list_changed']==oc['top50_changed']
            assert exact(c['FT_wins'],c['FT_losses'])==oc['exact_mcnemar_p']
            orig_cov={'raw_exact_mcnemar_p':oc['exact_mcnemar_p'],'holm_adjusted_p':oc['holm_adjusted_p'],'holm_family_size':oc['holm_family_size'],'scope':'Two pre-existing historical K50 family-versus-score coverage comparisons; separate from the four main title-count/order tests and the 48 EM tests.'}
        metrics={}
        for label in ['FT','strictFT','pool_FT','pool_strictFT','EM']:
            metrics[label]={'score_successes':c[label+'_score_successes'],'family_successes':c[label+'_family_successes'],'n':300,'wins':c[label+'_wins'],'losses':c[label+'_losses'],'ties':c[label+'_ties'],'delta_pp':(c[label+'_wins']-c[label+'_losses'])/3,'recomputed_raw_exact_mcnemar_p':exact(c[label+'_wins'],c[label+'_losses'])}
        summaries.append({'dataset':ds,'K':k,'metrics':metrics,'mechanism_counts':{key:c[key] for key in ['query_changed','first_title_changed','candidate_distinct_count_changed','selected_title_count_changed','selected_title_sequence_changed','selected_document_list_changed','input_key_changed','prediction_text_changed','logical_calls_changed','returned_record_budget_changed']},'baseline_pool_decomposition':{'FT_wins_reference_pool_missing':c['FT_wins_reference_pool_missing'],'FT_wins_reference_pool_complete':c['FT_wins_reference_pool_complete'],'FT_losses_family_pool_complete':c['FT_losses_family_pool_complete'],'interpretation':'Pool-missing wins require support access beyond the reference pool; pool-complete wins locate a retention difference but do not isolate a pure reranking mechanism.'},'coverage_EM_cross_counts':dict(cross),'coverage_gains_EM_transition_counts':dict(gain_answer),'original_K50_coverage_inference':orig_cov,'original_EM_inference':{'raw_exact_mcnemar_p':saved['EM_exact_mcnemar_p'],'bonferroni_adjusted_p':saved['EM_bonferroni_adjusted_p'],'family_size':56,'question_ci95_delta_pp':em['question_ci95_points'],'support_component_ci95_delta_pp':em['cluster_ci95_points']},'query_unchanged_pairs_have_identical_selected_documents_reader_input_and_predictions':True})

csvpath=OUT/'current/family_score_paired_minimal.csv'
with csvpath.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

oldreceipt=read(sources['historical_manual/complete.json']);oldsummary=read(sources['historical_manual/mechanism_summary.json']);oldfull=read(sources['historical_manual/full_historical_results.json'])
for p,key in [(sources['historical_manual/mechanism_summary.json'],'summary_sha256'),(sources['historical_manual/analyze_remote.py'],'script_sha256'),(sources['historical_manual/frozen_independent_rules.py'],'rules_sha256')]:assert sha(p)==oldreceipt[key]
assert sha(OLD_PROTOCOL)==oldsummary['source_protocol_sha256']
old_source_hashes=[]
for item in oldfull['sources']:
    assert sha(Path(item['path']))==item['sha256']
    old_source_hashes.append({**item,'current_hash_matches':True})
oldchecks=[]
for ds,d in oldsummary['datasets'].items():
    for k in oldsummary['K']:
        w=l=acq=ret=0
        for row in d['changed_cases']:
            ra,rb=row['score_complete_rank'],row['family_complete_rank']
            a=ra is not None and ra<=k;b=rb is not None and rb<=k
            w+=b and not a;l+=a and not b;acq+=b and not a and ra is None;ret+=b and not a and ra is not None
        saved=next(v for v in d['paired_by_group'] if v['group']=='all' and v['K']==k)
        assert w==saved.get('wins',0) and l==saved.get('losses',0)
        oldchecks.append({'dataset':ds,'K':k,'n':saved['n'],'wins':w,'losses':l,'ties':saved['n']-w-l,'reference_pool_missing_wins':acq,'reference_pool_complete_wins':ret,'matches_existing_summary':True})

manifest['generated/current/family_score_paired_minimal.csv']={'sha256':sha(csvpath),'rows':len(records),'scope':'Only paired IDs, scalar outcomes, change flags and input hashes; no question, document or prediction text.'}
manifest['generated/audit_saved_family.py']={'sha256':sha(Path(__file__))}
write(OUT/'MANIFEST.json',manifest)
result={
    'status':'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'review_claim_assessed':'No targeted family-preference control, and only aggregates prevent paired testing.',
    'verdict':'Incorrect as a statement about the saved evidence. A targeted FewRel family_u24 versus score2_u24 switch, per-question coverage and answers, existing paired tests and frozen implementation are available. Publication currently gives the EM contrast but little coverage/trigger evidence.',
    'operations':{'new_retrieval':0,'new_generation':0,'training':0,'network':0,'manuscript_edits':0,'paired_ids_per_dataset':300,'K_values':[10,20,30,50],'coverage_source_rows':43200,'answer_source_rows':43200,'retained_rows_per_source':4800,'minimal_paired_rows':2400,'method':'Read stored labels/scores and join by dataset,id,K,policy; recompute paired counts and exact McNemar; compare all eight EM pairs and original K50 coverage pairs with saved aggregates. No re-scoring of text or new predictions.'},
    'current_protocol':{'status':'Historically exposed 20261105 fixed paired cohorts; not independent confirmation','question_n':{'2wiki':300,'hotpotqa':300},'alpha':protocol['alpha'],'D':400,'corpus_documents':protocol['corpus_documents'],'retrieval_protocol_sha256':sha(sources['current/retrieval_protocol.json']),'cue_config_sha256':protocol['cue_config_sha256'],'cue_mode':'FewRel24 exact token matching in both arms, not the old manually assembled 24 cues','reference':'score2_u24','variant':'family_u24','exact_intervention':'Both use the same scored title-mention candidate extraction, source-priority/cue/rank/position/length score, at most two selected titles, one supplementary query and residual keywords. Family enables prefer_second_branch: after first selection, select the next distinct target from an unused question-explicit source-title family if a candidate is available; otherwise use the scored list. Score2 leaves that preference disabled.','parameters_held_fixed':['Historical question IDs and order','Validation-context-union corpus and index','BM25 and BGE model','RRF dense weights 0.3/0.7 and D=400','Five source passages and first 180 words','Two-to-eight-token title eligibility and same-sentence matching','FewRel24 exact cue list and five-word cue window','Score coefficients and score/tie order','At most two target titles and 24 residual keywords','Reader model, decoding and context formatting'],'observed_invariants':'Across all 600 question pairs: first selected title, selected-title count, candidate distinct-count, recorded logical calls and returned-record budget match; query changes occur on 20/300 and 8/300. Unchanged queries have identical saved selected-document sequences, reader input keys and predictions at all four K.'},
    'current_results':summaries,
    'inference_boundaries':['New raw p values for non-K50 coverage/pool cells are post-hoc descriptive recalculations, not a newly prespecified family; original K50 Holm values and all 48-test EM Bonferroni values are retained.','Candidate distinct-count equality does not prove candidate identities equal. Available per-question coverage exports omit full candidate mention pools, source-title provenance, initial ranked IDs and raw route-return lists. The exact source-family availability funnel cannot be rebuilt from these reduced records alone.','The controlled switch is verified in frozen code and recorded outputs; current exported records support second-title/query/input/coverage/EM transitions, not attribution to a retriever internal mechanism or a source-family frequency across systems.','Pool decomposition uses saved per-question acquired FT flags; it is not a fresh audit of full raw candidate pools.','FT and strictFT happen to have the same paired wins/losses here, but remain distinct metrics.','Historical exposures, one reader and one configuration remain. Positive coverage change is not proof of improved answers or a generally superior family rule.'],
    'historical_manual_support':{'use':'Separate historical supporting audit only; never merge with current 12376/7205 primary cohorts or the FewRel 300-question family control','n':{'2wiki':12476,'hotpotqa':7305},'strict_hotpotqa_n':7304,'alpha':{'2wiki':0.3,'hotpotqa':0.7},'cue_mode':'Manually assembled 24 cues in frozen_independent_rules.py:CUES','policies':'bridge_expand minus no_second_branch under this old manual-cue protocol','protocol_sha256':oldsummary['source_protocol_sha256'],'mechanism_summary_sha256':sha(sources['historical_manual/mechanism_summary.json']),'original_completion':oldreceipt,'current_receipt_hashes_match':True,'source_metric_files_currently_present_and_hash_match':old_source_hashes,'candidate_reconstruction':oldsummary['candidate_reconstruction'],'trigger_counts':{ds:v['counts'] for ds,v in oldsummary['datasets'].items()},'lightweight_changed_case_recheck':oldchecks,'scope':'Rechecked compact completion bindings, source-metric file hashes and 366 saved changed-mention cases. Did not reread 199 remote raw shards or rerun the full 19781-question reconstruction. The original completed audit records that work.'},
    'suggested_current_paper_text':'The source-family switch changes the supplementary query for 20/300 2Wiki and 8/300 HotpotQA questions. At K=50, complete title coverage changes from 252 to 257 and remains 281, respectively (five wins/no losses versus no discordances; original two-test Holm p=0.125 and 1). All five 2Wiki coverage gains require support absent from the Score2 pool, but only one changes from an EM non-match to a match. Family-versus-Score2 answer transitions are three wins/two losses and one win/one loss, with Bonferroni-adjusted p=1 in both corpora. This targeted historical control shows limited query changes and uneven answer conversion in the recorded pipeline.',
    'recommended_small_addition':'Add Family K50 absolute scores to the existing reader table and one short trigger/coverage paragraph; keep all four K EM contrasts already in Panel B. Do not substitute old manual full-population results for current FewRel evidence.',
    'evidence_manifest':str(OUT/'MANIFEST.json'),
}
write(OUT.parent/'family_audit.json',result)
print(json.dumps({'status':'PASS','current_paired_rows':len(records),'current_summary_cells':len(summaries),'old_changed_case_checks':len(oldchecks),'output':str(OUT.parent/'family_audit.json')},ensure_ascii=False))
