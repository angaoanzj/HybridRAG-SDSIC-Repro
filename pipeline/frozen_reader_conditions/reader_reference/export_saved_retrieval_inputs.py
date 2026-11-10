from pathlib import Path
import gzip
import hashlib
import json
import datetime

source = Path('/root/autodl-tmp/hybridrag_original_cohort_correction_20261108')
out = Path('/root/autodl-tmp/hybridrag_reader_input_audit_20261108')
out.mkdir(exist_ok=True)
dest = out / 'current_retrieval_groups.jsonl.gz'
assert not dest.exists(), dest
needed = {}
questions = {}
counts = {}
with gzip.open(dest, 'wt', encoding='utf-8') as writer:
    for dataset in ['2wiki', 'hotpotqa']:
        needed[dataset] = set()
        seen = set()
        counts[dataset] = 0
        with gzip.open(source / 'runs' / dataset / 'results.jsonl.gz', 'rt') as reader:
            for line in reader:
                row = json.loads(line)
                assert row['status'] == 'complete', row['status']
                key = (row['id'], row['policy'])
                assert key not in seen
                seen.add(key)
                assert set(row['selections']) == {'10', '20', '30', '50'}
                selected = row['selections']['50']
                assert len(selected) == 50
                needed[dataset].update(selected)
                qkey = dataset + '/' + row['id']
                if qkey in questions:
                    assert questions[qkey] == row['question']
                questions[qkey] = row['question']
                for cap in [10, 20, 30, 50]:
                    assert row['selections'][str(cap)] == selected[:cap]
                writer.write(json.dumps({
                    'dataset': dataset,
                    'id': row['id'],
                    'policy': row['policy'],
                    'question': row['question'],
                    'selected_doc_ids_50': selected,
                }, ensure_ascii=False) + '\n')
                counts[dataset] += 1
        assert counts[dataset] == 5700
docs_path = out / 'selected_documents.jsonl.gz'
assert not docs_path.exists()
found = {dataset: set() for dataset in needed}
with gzip.open(docs_path, 'wt', encoding='utf-8') as writer:
    for dataset in needed:
        with (source / 'data' / dataset / 'corpus.jsonl').open() as reader:
            for line in reader:
                row = json.loads(line)
                if row['doc_id'] not in needed[dataset]:
                    continue
                found[dataset].add(row['doc_id'])
                writer.write(json.dumps({'dataset': dataset, 'doc_id': row['doc_id'], 'title': row['title'], 'text': row['text']}, ensure_ascii=False) + '\n')
        assert found[dataset] == needed[dataset]
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
receipt = {
    'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'source': str(source),
    'scope': 'All 19 frozen conditions and all K=10,20,30,50. No retrieval, corpus, or index mutation.',
    'source_protocol_sha256': sha(source / 'protocol.json'),
    'source_results_sha256': {dataset: sha(source / 'runs' / dataset / 'results.jsonl.gz') for dataset in counts},
    'source_corpus_sha256': {dataset: sha(source / 'data' / dataset / 'corpus.jsonl') for dataset in counts},
    'method_question_groups': counts,
    'expanded_K_groups': sum(counts.values()) * 4,
    'question_count': len(questions),
    'selected_document_count': {dataset: len(ids) for dataset, ids in needed.items()},
    'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in [dest, docs_path]},
    'answer_or_support_fields_exported': False,
}
(out / 'retrieval_export_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
print(json.dumps(receipt, ensure_ascii=False, indent=2))
