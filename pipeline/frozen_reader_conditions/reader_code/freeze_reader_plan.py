"""Freeze canonical inputs, generation settings, and answer analysis before new calls."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    destination = root / 'analysis_plan.json'
    assert not destination.exists()
    assert not (root / 'outputs/predictions.jsonl').exists()
    receipt = json.loads((root / 'inputs/input_preparation_receipt.json').read_text())
    assert receipt['status'] == 'complete_before_new_answer_generation'
    assert receipt['groups'] == 45600 and not receipt['answer_labels_read']
    for name, record in receipt['files'].items():
        assert file_hash(root / 'inputs' / name) == record['sha256'], name
    reader = json.loads((root / 'inputs/reader_config.json').read_text())
    source_protocol = json.loads((root / 'retrieval_protocol.json').read_text())
    files = {}
    for directory in ['inputs', 'labels', 'official_answer_scoring', 'code']:
        for path in sorted((root / directory).glob('*')):
            if path.is_file() and path.suffix in ['.json', '.gz', '.py']:
                files[str(path.relative_to(root))] = file_hash(path)
    files['retrieval_protocol.json'] = file_hash(root / 'retrieval_protocol.json')
    files['receipts/service_before.json'] = file_hash(root / 'receipts/service_before.json')
    plan = {
        'status': 'frozen_before_new_reader_generation',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Answer evaluation on the historical 20261105 fixed matched cohorts. No independent new-question confirmation and no complete-knowledge-base claim.',
        'datasets': source_protocol['datasets'],
        'policies': source_protocol['policies'],
        'K_values': source_protocol['K_values'],
        'n_per_dataset': 300,
        'total_answer_groups': 45600,
        'comparisons': source_protocol['comparisons'],
        'seed': 2026110602,
        'bootstrap_resamples': 5000,
        'statistics': {
            'unit': 'Uniform macro average over the same 300 questions per dataset',
            'EM_tests': 'Exact paired McNemar, Holm across 56 predeclared dataset/contrast/K comparisons',
            'EM_F1_intervals': 'Pointwise paired question bootstrap, 5000 resamples',
            'F1_tests': 'No F1 p values. Pointwise paired intervals are secondary descriptive statistics.',
        },
        'answer_metric': {
            'type': 'Canonical-reference Answer EM and token F1 from pinned original official dataset scripts',
            'gold_field': 'answer from each fixed scoring-label record',
            'alias_expansion': False,
            'alias_note': 'The newer 2Wiki v1.1 answer-ID alias protocol is distinct. No unverified alias reference is added.',
            'prediction_processing': 'Model response content.strip(), followed by official normalization in the scorer',
            'annotation_gate': 'Scorer opens answer labels after the generation completion marker and prediction hashes validate.',
        },
        'reader_config': reader,
        'request_params': reader['params'],
        'generation': {
            'endpoint': 'http://127.0.0.1:18271',
            'concurrency': 8,
            'max_attempts': 3,
            'question_order_seed': 2026110801,
            'order': 'SHA256(seed:dataset:id), then strict input_key. Fixed before new answer generation.',
            'cache': 'One prediction per identical question and full ordered raw context, actual prompt, reader configuration and token count',
            'history_priority': ['matched1800', 'pilot3600', 'AIR1200'],
            'history_tie_break': 'Fixed source line, with all prediction disagreements retained. Answer quality is not used for reuse or selection.',
            'failure_policy': 'Retry transport/server failures with unchanged parameters. Preserve successes. Integrity violations stop the run.',
            'latency_scope': 'Concurrent quality calls and historical cached outputs do not establish matched condition latency.',
        },
        'input_counts': {name: receipt[name] for name in ['groups', 'unique_reader_inputs', 'unique_reusable_inputs', 'unique_missing_inputs', 'reusable_group_rows', 'missing_group_rows']},
        'retrieval_unchanged': True,
        'index_rebuilt': False,
        'files': files,
    }
    destination.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': plan['status'], 'analysis_plan_sha256': file_hash(destination), 'counts': plan['input_counts']}), flush=True)


if __name__ == '__main__':
    main()
