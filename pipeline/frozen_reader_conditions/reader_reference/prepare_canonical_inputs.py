"""Reconstruct reader inputs and audit historical reuse without answer labels."""
from pathlib import Path
from collections import Counter, defaultdict
import concurrent.futures
import datetime
import gzip
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import urllib.request

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
sys.path.insert(0, '/root/reader-k-pilot-20261107/code')
import reader_worker as reference

ROOT = Path('/root/autodl-tmp/hybridrag_reader_all19_allK_20261108')
OUT = ROOT / 'inputs'
SOURCE = OUT / 'source'
assert not (OUT / 'input_preparation_receipt.json').exists()
OUT.mkdir(exist_ok=True)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def write_gzip(path, rows):
    assert not path.exists(), path
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(reference.MODEL, local_files_only=True)
model_dir = Path(reference.MODEL)
reader_config = {
    'model': reference.MODEL,
    'model_sha256': file_hash(model_dir / 'model.safetensors'),
    'config_sha256': file_hash(model_dir / 'config.json'),
    'tokenizer_sha256': file_hash(model_dir / 'tokenizer.json'),
    'tokenizer_config_sha256': file_hash(model_dir / 'tokenizer_config.json'),
    'chat_template_sha256': file_hash(model_dir / 'chat_template.jinja'),
    'reference_worker_sha256': file_hash(Path(reference.__file__)),
    'system_prompt': reference.SYSTEM,
    'params': reference.PARAMS,
    'body_token_cap': 192,
    'title_token_cap': 32,
    'prompt_token_cap': 24512,
    'software': {name: importlib.metadata.version(name) for name in ['vllm', 'transformers', 'tokenizers', 'torch']},
}
assert reader_config['reference_worker_sha256'] == 'a358cbc3d24c33f426c1c32abb12a743408b54b21231fb497e654a8746fa7c48'
reader_config_hash = digest(reader_config)
print(json.dumps({'phase': 'model_frozen', 'reader_config_hash': reader_config_hash}), flush=True)

export_receipt = json.loads((SOURCE / 'retrieval_export_receipt.json').read_text())
for name, expected in export_receipt['files'].items():
    assert file_hash(SOURCE / name) == expected['sha256']
groups = load_gzip(SOURCE / 'current_retrieval_groups.jsonl.gz')
assert len(groups) == 11400
current_queries = {}
for row in groups:
    key = (row['dataset'], row['id'])
    if key in current_queries:
        assert current_queries[key] == row['question']
    current_queries[key] = row['question']
assert len(current_queries) == 600

formatted_documents = {}
documents_by_id = {}


def prepare_document(row):
    title_hash = hashlib.sha256(row['title'].encode()).hexdigest()
    text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
    key = (row['doc_id'], title_hash, text_hash)
    if key not in formatted_documents:
        body_ids = tokenizer.encode(row['text'], add_special_tokens=False)
        title_ids = tokenizer.encode(row['title'], add_special_tokens=False)
        body = tokenizer.decode(body_ids[:192], skip_special_tokens=True)
        title = tokenizer.decode(title_ids[:32], skip_special_tokens=True)
        formatted_documents[key] = {
            'doc_id': row['doc_id'], 'title': row['title'],
            'title_sha256': title_hash, 'full_body_sha256': text_hash,
            'body_original_tokens': len(body_ids),
            'body_used_tokens': min(192, len(body_ids)),
            'body_truncated': len(body_ids) > 192,
            'body_sha256': hashlib.sha256(body.encode()).hexdigest(),
            'formatted_title': title, 'formatted_body': body,
        }
    return formatted_documents[key]


for row in load_gzip(SOURCE / 'selected_documents.jsonl.gz'):
    documents_by_id[(row['dataset'], row['doc_id'])] = prepare_document(row)
print(json.dumps({'phase': 'documents_formatted', 'count': len(documents_by_id)}), flush=True)

prompt_cache = {}


def prepare_context(dataset, qid, question, docs):
    parts = [f"Document {index}\nTitle: {doc['formatted_title']}\n{doc['formatted_body']}" for index, doc in enumerate(docs, 1)]
    messages = [{'role': 'system', 'content': reference.SYSTEM}, {'role': 'user', 'content': '\n\n'.join(parts) + '\n\nQuestion: ' + question + '\nShort answer:'}]
    prompt_hash = digest(messages)
    raw_context = [{'doc_id': doc['doc_id'], 'title_sha256': doc['title_sha256'], 'full_body_sha256': doc['full_body_sha256']} for doc in docs]
    context_hash = digest(raw_context)
    input_key = digest({'dataset': dataset, 'id': qid, 'question': question, 'raw_context_sha256': context_hash, 'prompt_sha256': prompt_hash, 'reader_config_sha256': reader_config_hash})
    if prompt_hash not in prompt_cache:
        prompt_cache[prompt_hash] = {'messages': messages}
    return {
        'input_key': input_key, 'dataset': dataset, 'id': qid, 'question': question,
        'prompt_sha256': prompt_hash, 'raw_context_sha256': context_hash,
        'reader_config_sha256': reader_config_hash,
        'selected_doc_ids': [doc['doc_id'] for doc in docs],
        'meta': {
            'retained_doc_count': len(docs), 'reader_doc_count': len(docs),
            'budget_removed_doc_count': 0,
            'body_truncated_doc_count': sum(doc['body_truncated'] for doc in docs),
            'documents': [{key: doc[key] for key in ['doc_id', 'title', 'body_original_tokens', 'body_used_tokens', 'body_truncated', 'body_sha256']} for doc in docs],
        },
    }


unique_inputs = {}
group_map = []
for index, row in enumerate(sorted(groups, key=lambda x: (x['dataset'], x['id'], x['policy']))):
    all_docs = [documents_by_id[(row['dataset'], doc_id)] for doc_id in row['selected_doc_ids_50']]
    for cap in [10, 20, 30, 50]:
        prepared = prepare_context(row['dataset'], row['id'], row['question'], all_docs[:cap])
        unique_inputs.setdefault(prepared['input_key'], prepared)
        group_map.append({'dataset': row['dataset'], 'id': row['id'], 'policy': row['policy'], 'K': cap, 'input_key': prepared['input_key'], 'prompt_sha256': prepared['prompt_sha256'], 'raw_context_sha256': prepared['raw_context_sha256'], 'selected_doc_ids': prepared['selected_doc_ids']})
    if (index + 1) % 1500 == 0:
        print(json.dumps({'phase': 'current_contexts', 'groups': (index + 1) * 4, 'unique_inputs': len(unique_inputs)}), flush=True)
assert len(group_map) == 45600

history_specs = [
    ('matched1800', Path('/root/autodl-tmp/reader_matched_20261105_side'), 'quality.jsonl', 'protocol.json'),
    ('pilot3600', Path('/root/reader-k-pilot-20261107'), 'results/quality.jsonl', 'run_freeze.json'),
    ('AIR1200', Path('/root/autodl-tmp/reader_air_20261105_side'), 'quality.jsonl', 'protocol.json'),
]
history_by_key = defaultdict(list)
history_candidates = []
history_info = []
history_reconstruction_checks = 0
for priority, (name, directory, predictions_name, protocol_name) in enumerate(history_specs):
    freeze = json.loads((directory / protocol_name).read_text())
    assert freeze.get('model_sha256', freeze.get('model_weight_sha256')) == reader_config['model_sha256']
    assert freeze['tokenizer_sha256'] == reader_config['tokenizer_sha256']
    assert freeze['chat_template_sha256'] == reader_config['chat_template_sha256']
    assert freeze.get('reader_worker_sha256', freeze.get('worker_sha256')) == reader_config['reference_worker_sha256']
    if name == 'pilot3600':
        assert freeze['params'] == reference.PARAMS and freeze['system_prompt'] == reference.SYSTEM
    rows = {(row['dataset'], row['id']): row for row in load_gzip(directory / 'reader_inputs.jsonl.gz')}
    counts = Counter()
    with (directory / predictions_name).open() as stream:
        for line_number, line in enumerate(stream, 1):
            old = json.loads(line)
            counts['total_rows'] += 1
            qkey = (old['dataset'], old['id'])
            if qkey not in current_queries:
                continue
            counts['current_question_rows'] += 1
            assert not old.get('failed')
            row = rows[qkey]
            assert row['question'] == current_queries[qkey]
            if old['K'] not in [10, 20, 30, 50]:
                counts['unsupported_current_K_rows'] += 1
                continue
            docs = [prepare_document(doc) for doc in row['policies'][old['policy']]['docs'][:old['K']]]
            prepared = prepare_context(old['dataset'], old['id'], row['question'], docs)
            assert prepared['prompt_sha256'] == old['prompt_sha256']
            assert prepared['meta']['documents'] == old['documents']
            assert old['usage']['prompt_tokens'] == old['tokenizer_input_tokens']
            assert old.get('budget_removed_doc_count', 0) == 0
            history_reconstruction_checks += 1
            candidate = {
                **prepared, 'history_source': name, 'source_priority': priority,
                'source_file': str(directory / predictions_name), 'source_line': line_number,
                'source_policy': old['policy'], 'source_K': old['K'],
                'prediction': old['prediction'], 'usage': old['usage'],
                'finish_reason': old['finish_reason'], 'model': old['model'],
                'tokenizer_input_tokens': old['tokenizer_input_tokens'],
                'local_rendered_input_tokens': old['local_rendered_input_tokens'],
            }
            history_candidates.append(candidate)
            history_by_key[prepared['input_key']].append(candidate)
            counts['eligible_reconstructed_rows'] += 1
    history_info.append({'name': name, 'directory': str(directory), 'files': {file: file_hash(directory / file) for file in [predictions_name, protocol_name, 'reader_inputs.jsonl.gz']}, 'counts': dict(counts)})

print(json.dumps({'phase': 'history_reconstructed', 'counts': history_info, 'unique_prompts_to_count': len(prompt_cache)}), flush=True)


TOKENIZER_LOCK = threading.Lock()

def count_prompt(item):
    prompt_hash, data = item
    with TOKENIZER_LOCK:
        local_tokens = len(reference.chat_ids(tokenizer, data['messages']))
    service_tokens = reference.service_token_count(data['messages'])
    assert 50 < service_tokens <= 24512
    return prompt_hash, local_tokens, service_tokens


# All requested K are <=50, and every historical candidate reconstructed here
# also has K<=50. The frozen budget can be enforced without any document removal.
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for index, result in enumerate(pool.map(count_prompt, prompt_cache.items()), 1):
        prompt_hash, local_tokens, service_tokens = result
        prompt_cache[prompt_hash]['local_rendered_input_tokens'] = local_tokens
        prompt_cache[prompt_hash]['tokenizer_input_tokens'] = service_tokens
        if index % 500 == 0:
            print(json.dumps({'phase': 'same_reader_token_counts', 'completed': index, 'total': len(prompt_cache)}), flush=True)

for candidate in history_candidates:
    counted = prompt_cache[candidate['prompt_sha256']]
    assert candidate['tokenizer_input_tokens'] == counted['tokenizer_input_tokens']
    assert candidate['local_rendered_input_tokens'] == counted['local_rendered_input_tokens']

selected_history = []
conflicts = []
reuse_by_input = {}
for key, row in unique_inputs.items():
    counted = prompt_cache[row['prompt_sha256']]
    row['messages'] = counted['messages']
    row['meta'].update({name: counted[name] for name in ['tokenizer_input_tokens', 'local_rendered_input_tokens']})
    row['meta']['prompt_sha256'] = row['prompt_sha256']
    row['meta']['token_count_source'] = 'same live reader /tokenize endpoint, checked before generation'
    candidates = sorted(history_by_key.get(key, []), key=lambda x: (x['source_priority'], x['source_line']))
    if not candidates:
        continue
    selected = candidates[0]
    selected_history.append(selected)
    reuse_by_input[key] = selected
    if len({candidate['prediction'] for candidate in candidates}) > 1:
        conflicts.append({'input_key': key, 'candidates': [{field: candidate[field] for field in ['history_source', 'source_file', 'source_line', 'prediction']} for candidate in candidates], 'selected_source_file': selected['source_file'], 'selected_source_line': selected['source_line']})

matrix = defaultdict(Counter)
for row in group_map:
    history = reuse_by_input.get(row['input_key'])
    row['historical_match'] = history is not None
    if history:
        row['history_source'] = history['history_source']
    matrix[(row['dataset'], row['policy'], row['K'])]['total'] += 1
    matrix[(row['dataset'], row['policy'], row['K'])]['reusable' if history else 'missing'] += 1

write_gzip(OUT / 'unique_inputs.jsonl.gz', sorted(unique_inputs.values(), key=lambda row: row['input_key']))
write_gzip(OUT / 'group_input_map.jsonl.gz', group_map)
write_gzip(OUT / 'history_selected_predictions.jsonl.gz', sorted(selected_history, key=lambda row: row['input_key']))
write_gzip(OUT / 'history_prediction_conflicts.jsonl.gz', conflicts)
write_gzip(OUT / 'formatted_documents.jsonl.gz', sorted(formatted_documents.values(), key=lambda row: (row['doc_id'], row['full_body_sha256'])))
(OUT / 'reader_config.json').write_text(json.dumps(reader_config, ensure_ascii=False, indent=2))
matrix_rows = [{'dataset': key[0], 'policy': key[1], 'K': key[2], **dict(counts)} for key, counts in sorted(matrix.items())]
(OUT / 'reuse_missing_matrix.json').write_text(json.dumps(matrix_rows, ensure_ascii=False, indent=2))
receipt = {
    'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status': 'complete_before_new_answer_generation',
    'groups': len(group_map), 'unique_reader_inputs': len(unique_inputs),
    'unique_reusable_inputs': len(selected_history),
    'unique_missing_inputs': len(unique_inputs) - len(selected_history),
    'reusable_group_rows': sum(row['historical_match'] for row in group_map),
    'missing_group_rows': sum(not row['historical_match'] for row in group_map),
    'history_rows_inspected': sum(item['counts']['total_rows'] for item in history_info),
    'history_rows_full_prompt_reconstructed': history_reconstruction_checks,
    'history_prediction_conflict_inputs': len(conflicts),
    'history_sources': history_info,
    'reader_config_sha256': reader_config_hash,
    'reuse_rule': 'Exact same dataset/question ID/question text, ordered raw document IDs/title/full body SHA256, serialized messages, reader model/tokenizer/template/params, and actual /tokenize count. Priority matched1800, pilot3600, AIR1200, then fixed source line. No outcome-based choice.',
    'answer_labels_read': False,
    'source_protocol_sha256': export_receipt['source_protocol_sha256'],
    'all_K_groups_use_existing_saved_rankings': True,
    'max_actual_prompt_tokens': max(data['tokenizer_input_tokens'] for data in prompt_cache.values()),
    'global_budget_removed_groups': 0,
    'files': {path.name: {'bytes': path.stat().st_size, 'sha256': file_hash(path)} for path in sorted(OUT.glob('*')) if path.is_file()},
    'builder_sha256': file_hash(__file__),
}
(OUT / 'input_preparation_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
print(json.dumps({key: value for key, value in receipt.items() if key not in ['history_sources', 'files']}, ensure_ascii=False, indent=2), flush=True)
