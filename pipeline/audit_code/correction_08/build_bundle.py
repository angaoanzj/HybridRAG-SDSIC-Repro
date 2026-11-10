"""One-time extraction of stored evidence; no retrieval, training or generation.

Only this extraction script uses original local provenance paths. The separately
provided verify_bundle.py works from bundled relative paths on another computer.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
AUDIT = Path('/path/to/local/home/Documents/待审核论文/HybridRAG_论文审核_20261110/independent_reaudit_07')
DELIVERY_SOURCE = Path('/path/to/local/home/.codex/visualizations/2026/11/07/01a1b6f1-912d-7e41-9fce-a1cec97f186a/HybridRAG_获取与排序分解_五页稿_20261109/配套材料')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def selected_records(path, policies):
    result = {}
    with gzip.open(path, 'rt') as f:
        for line in f:
            row = json.loads(line)
            if row['policy'] not in policies:
                continue
            key = (row['dataset'], row['id'], int(row['K']), row['policy'])
            assert key not in result, key
            result[key] = row
    return result

def main():
    for name in ['delivery', 'synonyms']:
        (ROOT/name).mkdir(exist_ok=True)
    copies = []
    def copy(source, relative):
        target = ROOT/relative
        shutil.copyfile(source, target)
        assert sha(source) == sha(target)
        copies.append({'source_path': str(source), 'bundle_path': relative,
                       'source_sha256': sha(source), 'copy_sha256': sha(target)})
    for name in ['coverage_answer_pairs.csv', 'coverage_answer_summary.json', 'coverage_answer_intersections.json']:
        copy(DELIVERY_SOURCE/name, 'delivery/'+name)
    copy(AUDIT/'synonym_check.json', 'synonyms/synonym_check.json')
    audit = json.loads((AUDIT/'synonym_check.json').read_text())
    cue_path = Path(audit['sources']['cue_config']['path'])
    copy(cue_path, 'synonyms/cue_config.json')
    policies = ('score2_morph', 'no_second_branch')
    coverage_path = Path(audit['sources']['coverage_records']['path'])
    answers_path = Path(audit['sources']['answer_records']['path'])
    for key, path in [('coverage_records', coverage_path), ('answer_records', answers_path)]:
        assert sha(path) == audit['sources'][key]['sha256'], key
    coverage = selected_records(coverage_path, policies)
    answers = selected_records(answers_path, policies)
    assert set(coverage) == set(answers)
    question_pairs = sorted({key[:2] for key in coverage})
    assert len(question_pairs) == 600
    queries = []
    pairs = []
    for dataset, question_id in question_pairs:
        qrow = {'dataset': dataset, 'id': question_id}
        for label, policy in zip(('morph', 'syn'), policies):
            records = [coverage[(dataset, question_id, k, policy)] for k in [10,20,30,50]]
            assert len({r['expanded_query'] for r in records}) == 1
            qrow[label+'_query'] = records[0]['expanded_query']
            qrow[label+'_selected_titles'] = json.dumps(records[0]['selected_titles'], ensure_ascii=False)
            qrow[label+'_selected_cues'] = json.dumps(records[0]['selected_cues'], ensure_ascii=False)
        queries.append(qrow)
        for k in [10,20,30,50]:
            row = {'dataset':dataset, 'id':question_id, 'K':k}
            for label, policy in zip(('morph', 'syn'), policies):
                c = coverage[(dataset, question_id, k, policy)]
                a = answers[(dataset, question_id, k, policy)]
                row[label+'_ordered_ids_sha256'] = digest_ids(c['selected_ids'])
                for field in ['input_key','prompt_sha256','raw_context_sha256','prediction','EM','F1']:
                    row[label+'_'+field] = a[field]
            pairs.append(row)
    write_csv(ROOT/'synonyms/paired_queries_600.csv', queries)
    write_csv(ROOT/'synonyms/paired_reader_records_2400.csv', pairs)
    write_json(ROOT/'synonyms/question_identity_checks_600.json', audit['all_600_question_identity_checks'])
    write_json(ROOT/'synonyms/changed_query_case.json', audit['changed_query_cases'])
    index = {
        'schema_version':1,
        'purpose':'Portable evidence supplement for correction_08; deterministic reuse of stored records only.',
        'copied_sources':copies,
        'extracted_sources':[
            {'source_path':str(coverage_path),'source_sha256':sha(coverage_path),
             'outputs':['synonyms/paired_queries_600.csv','synonyms/paired_reader_records_2400.csv'],
             'selection':'policies score2_morph and no_second_branch; both corpora, all300 IDs each, K10/20/30/50'},
            {'source_path':str(answers_path),'source_sha256':sha(answers_path),
             'outputs':['synonyms/paired_reader_records_2400.csv'],
             'selection':'Same dataset,id,K,policy join; predictions and input hashes retained verbatim'}],
        'claims': {
            'delivery':'Section6.4; historical FewRel Score1 versus Score2, K50; N300 per corpus. Other K are retained in originalCSV.',
            'synonyms':'Section6.1 and Table5B Syn-minus-Morph; historical FewRel, N300 per corpus, four saved K.',
            'differences':'Delivery compares Score1/Score2. Synonym analysis compares score2_morph/no_second_branch. Never merge their denominators or controls.'},
        'boundaries':[
            'No new retrieval, training, reader generation, tokenization or prompt reconstruction.',
            'Delivered-support flags and answer scores are inherited saved measurements. Verification reaggregates them, not reruns the original measurement.',
            'Ordered-document SHA256 is derived from the exact saved ordered ID list using UTF8 canonical JSON, ensure_ascii=False, separators=(comma,colon). Full unchanged ID lists are not duplicated.',
            'Input and prediction hashes/values demonstrate saved equality or difference; this is not a fresh model execution.',
            'Selected-title cue lists do not cover discarded candidate windows. Zero changed queries does not establish zero candidate synonym matches.',
            'The original execution/config/code binding audit is preserved, but its historical source paths are provenance only; portable verifier does not depend on them.',
            '11 prediction changes and3 positiveF1 outcomes are properties of the same27-case subset, not disjoint categories or EM repairs.',
            'No statistical-significance, cross-system, multi-round, or confirmation-reader claim is introduced.'],
        'run_portably':'python3 verify_bundle.py (Python standard library only; any current working directory)',
        'artifact_sha256':{}
    }
    for folder in [ROOT/'delivery', ROOT/'synonyms']:
        for file in sorted(folder.rglob('*')):
            if file.is_file():index['artifact_sha256'][str(file.relative_to(ROOT))] = sha(file)
    write_json(ROOT/'evidence_index.json', index)
    print(json.dumps({'copied_files':len(copies),'query_rows':len(queries),'paired_reader_rows':len(pairs),'status':'EXTRACTED'},ensure_ascii=False))

if __name__ == '__main__':
    main()
