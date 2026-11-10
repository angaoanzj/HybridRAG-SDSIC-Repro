"""Reaggregate this bundle using only Python's standard library and local files.

Run: python3 /path/to/evidence_bundle/verify_bundle.py
No old machine path, network, model, index or corpus is required.
"""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
KS = [10, 20, 30, 50]

def read_csv(relative):
    with (ROOT/relative).open(newline='') as f:
        return list(csv.DictReader(f))

def number(row, key):
    return float(row[key])

def check(condition, message):
    if not condition:
        raise AssertionError(message)

def main():
    index = json.loads((ROOT/'evidence_index.json').read_text())
    checksums = []
    for relative, expected in index['artifact_sha256'].items():
        actual = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        check(actual == expected, 'Checksum: '+relative)
        checksums.append({'file':relative,'sha256':actual})
    rows = read_csv('delivery/coverage_answer_pairs.csv')
    check(len(rows) == 2400, 'Delivery row count')
    keys = [(r['dataset'], r['id'], int(r['K'])) for r in rows]
    check(len(set(keys)) == len(keys), 'Delivery duplicate keys')
    groups = Counter((r['dataset'],int(r['K'])) for r in rows)
    check(groups == Counter({(d,k):300 for d in ['2wiki','hotpotqa'] for k in KS}), 'Delivery grid')
    delivery = {}
    expected_render = {'2wiki':{'s1':(215,197),'s2':(252,232)},'hotpotqa':{'s1':(277,269),'s2':(281,273)}}
    for dataset in ['2wiki','hotpotqa']:
        group = [r for r in rows if r['dataset']==dataset and int(r['K'])==50]
        counts = {}
        for policy in ['s1','s2']:
            values = (sum(int(r[policy+'_raw_strict_F']) for r in group),sum(int(r[policy+'_reader_strict_F']) for r in group))
            check(values == expected_render[dataset][policy], f'Delivery count {dataset} {policy}')
            counts[policy] = {'retrieved_strict':values[0],'rendered_strict':values[1],'denominator':len(group)}
        delivery[dataset] = counts
    group = [r for r in rows if r['dataset']=='2wiki' and int(r['K'])==50]
    gain = [r for r in group if int(r['s1_title_F'])==0 and int(r['s2_title_F'])==1]
    repaired = [r for r in gain if number(r,'s1_EM')==0 and number(r,'s2_EM')==1]
    already = [r for r in gain if number(r,'s1_EM')==1 and number(r,'s2_EM')==1]
    persistent = [r for r in gain if number(r,'s1_EM')==0 and number(r,'s2_EM')==0]
    complete = [r for r in persistent if int(r['s2_reader_strict_F'])==1]
    lost = [r for r in persistent if int(r['s2_reader_strict_F'])==0]
    check(all(int(r['s2_body_truncation_loses_complete_support'])==1 for r in lost), 'Two cases must have saved truncation flags')
    split = {'title_coverage_gains':len(gain),'EM_repairs':len(repaired),'already_correct':len(already),'persistent_nonmatches':len(persistent),'persistent_complete_rendered_support':len(complete),'persistent_support_lost_in_rendering':len(lost),'within_complete_prediction_changes':sum(r['s1_prediction']!=r['s2_prediction'] for r in complete),'within_complete_Score2_positive_F1':sum(number(r,'s2_F1')>0 for r in complete)}
    check(list(split.values()) == [39,6,4,29,27,2,11,3], 'Nested conversion counts')
    check(len(gain)==len(repaired)+len(already)+len(persistent), '39 partition')
    check(len(persistent)==len(complete)+len(lost), '29 partition')
    intersections = json.loads((ROOT/'delivery/coverage_answer_intersections.json').read_text())['2wiki']
    for a,b in [('title_coverage_gains','new_coverage_n'),('EM_repairs','repaired_n'),('persistent_nonmatches','persistent_wrong_n'),('persistent_complete_rendered_support','persistent_wrong_complete_reader_support_n'),('persistent_support_lost_in_rendering','persistent_wrong_body_truncation_n')]:
        check(split[a]==intersections[b], 'Archived intersection: '+b)
    summary = json.loads((ROOT/'delivery/coverage_answer_summary.json').read_text())
    for dataset in ['2wiki','hotpotqa']:
        item = next(r for r in summary['summaries'] if r['dataset']==dataset and r['K']==50 and r['coverage_transition']=='all')
        for policy in ['s1','s2']:
            check(delivery[dataset][policy]['rendered_strict']==item[policy+'_reader_strict_F_count'], 'Archived summary')
    cohort = {'s1_EM_count':int(sum(number(r,'s1_EM') for r in group)),'s2_EM_count':int(sum(number(r,'s2_EM') for r in group)),'EM_wins':sum(number(r,'s2_EM')>number(r,'s1_EM') for r in group),'EM_losses':sum(number(r,'s2_EM')<number(r,'s1_EM') for r in group),'denominator':300}
    check(list(cohort.values())==[74,87,20,7,300], 'Whole-cohort EM values')

    queries = read_csv('synonyms/paired_queries_600.csv')
    pairs = read_csv('synonyms/paired_reader_records_2400.csv')
    check(len(queries)==600 and len(pairs)==2400, 'Synonym row counts')
    qmap = {(r['dataset'],r['id']):r for r in queries}
    pmap = {(r['dataset'],r['id'],int(r['K'])):r for r in pairs}
    check(len(qmap)==600 and len(pmap)==2400, 'Synonym unique keys')
    check(Counter(r['dataset'] for r in queries)==Counter({'2wiki':300,'hotpotqa':300}), 'Synonym cohorts')
    audit = json.loads((ROOT/'synonyms/synonym_check.json').read_text())
    synonyms = []
    fields = ['ordered_ids_sha256','input_key','prompt_sha256','raw_context_sha256','prediction','EM','F1']
    old_keys = {'ordered_ids_sha256':'selected_ids_changed','input_key':'input_key_changed','prompt_sha256':'prompt_sha256_changed','raw_context_sha256':'raw_context_sha256_changed','prediction':'prediction_changed','EM':'EM_changed','F1':'F1_changed'}
    identities = []
    for key, q in sorted(qmap.items()):
        relevant = [pmap[(*key,k)] for k in KS]
        identities.append({'dataset':key[0],'id':key[1],'query_changed':q['morph_query']!=q['syn_query'],'ordered_documents_changed_K':[k for k,r in zip(KS,relevant) if r['morph_ordered_ids_sha256']!=r['syn_ordered_ids_sha256']],'reader_inputs_changed_K':[k for k,r in zip(KS,relevant) if any(r['morph_'+f]!=r['syn_'+f] for f in ['input_key','prompt_sha256','raw_context_sha256'])],'predictions_changed_K':[k for k,r in zip(KS,relevant) if r['morph_prediction']!=r['syn_prediction']],'EM_changed_K':[k for k,r in zip(KS,relevant) if float(r['morph_EM'])!=float(r['syn_EM'])],'F1_changed_K':[k for k,r in zip(KS,relevant) if float(r['morph_F1'])!=float(r['syn_F1'])]})
    archived_ids = {(r['dataset'],r['id']):r for r in json.loads((ROOT/'synonyms/question_identity_checks_600.json').read_text())}
    for r in identities:
        old = archived_ids[(r['dataset'],r['id'])]
        for f,value in r.items():check(value==old[f], '600-case identity '+f)
    for dataset in ['2wiki','hotpotqa']:
        query_changes = sum(q['morph_query']!=q['syn_query'] for q in queries if q['dataset']==dataset)
        check(query_changes==(0 if dataset=='2wiki' else 1), 'Synonym query changes')
        for k in KS:
            subset = [r for r in pairs if r['dataset']==dataset and int(r['K'])==k]
            check(len(subset)==300, 'Synonym K group count')
            item = {'dataset':dataset,'K':k,'denominator':300,'query_changes':query_changes}
            old = next(r for r in audit['summary_by_dataset_K'] if r['dataset']==dataset and r['K']==k)
            for field in fields:
                changes = sum((float(r['morph_'+field])!=float(r['syn_'+field])) if field in ['EM','F1'] else (r['morph_'+field]!=r['syn_'+field]) for r in subset)
                item[field+'_changes']=changes
                check(changes==old[old_keys[field]], 'Synonym archive '+field)
            synonyms.append(item)
    check(audit['configuration']['accepted_synonym_keys']==[['boy','n']], 'Accepted synonym inventory')
    result = {'status':'PASS_SAVED_RECORD_REAGGREGATION','portable':True,'all_paths_resolved_relative_to_script':True,'checksums_verified':checksums,'delivery_rows':len(rows),'delivery_grid':[{ 'dataset':d,'K':k,'n':n} for (d,k),n in sorted(groups.items())],'delivery_strict_support_K50':delivery,'delivery_nested_2wiki_K50':split,'delivery_whole_2wiki_K50':cohort,'synonym_queries':len(queries),'synonym_paired_K_records':len(pairs),'synonym_results':synonyms,'changed_query_ids':[{'dataset':r['dataset'],'id':r['id']} for r in identities if r['query_changed']],'all_600_identities_match_prior_audit':True,'limits':index['boundaries'],'new_training_retrieval_or_generation':False}
    output = ROOT/'verification_result.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'delivery_nested':split,'synonym_queries_changed':{'2wiki':0,'hotpotqa':1},'all_2400_predictions_unchanged':all(r['prediction_changes']==0 for r in synonyms),'report':str(output)},ensure_ascii=False))

if __name__=='__main__':
    main()
