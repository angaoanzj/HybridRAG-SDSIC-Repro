"""Generate missing frozen reader inputs, preserving successful predictions on retry."""

import argparse
import concurrent.futures
import datetime
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row['input_key']
        if key in indexed:
            raise ValueError('Duplicate input key ' + key)
        indexed[key] = row
    return indexed


def load_checkpoint(path):
    if not path.exists():
        return {}
    content = path.read_bytes()
    if content and not content.endswith(b'\n'):
        boundary = content.rfind(b'\n') + 1
        tail = content[boundary:]
        # A complete final JSON row can be committed by adding its newline.
        try:
            json.loads(tail)
            content += b'\n'
        except (json.JSONDecodeError, UnicodeDecodeError):
            evidence = path.with_name(path.name + f'.partial_tail_{time.time_ns()}')
            evidence.write_bytes(tail)
            content = content[:boundary]
        temporary = path.with_suffix(path.suffix + '.recovered')
        temporary.write_bytes(content)
        temporary.replace(path)
    return index_rows(load_rows(path))


def check_prediction(row, job, model_name):
    if row.get('failed') or not isinstance(row.get('prediction'), str):
        raise ValueError('Unsuccessful prediction ' + job['input_key'])
    if row['prompt_sha256'] != job['prompt_sha256']:
        raise ValueError('Prediction prompt mismatch')
    for field in ['dataset', 'id', 'raw_context_sha256', 'reader_config_sha256']:
        if row[field] != job[field]:
            raise ValueError('Prediction lineage mismatch ' + field)
    if row['usage']['prompt_tokens'] != job['meta']['tokenizer_input_tokens']:
        raise ValueError('Actual prompt token count differs from frozen input')
    if row.get('model') != model_name:
        raise ValueError('Reader model name differs from frozen request')
    if row.get('finish_reason') not in ['stop', 'length']:
        raise ValueError('Unexpected generation finish reason')


def request_answer(job, params, endpoint):
    payload = {**params, 'messages': job['messages']}
    request = urllib.request.Request(
        endpoint + '/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    choice = result['choices'][0]
    message = choice['message']
    row = {
        'input_key': job['input_key'],
        'dataset': job['dataset'],
        'id': job['id'],
        'prompt_sha256': job['prompt_sha256'],
        'raw_context_sha256': job['raw_context_sha256'],
        'reader_config_sha256': job['reader_config_sha256'],
        'prediction': (message.get('content') or '').strip(),
        'reasoning_content': message.get('reasoning_content', message.get('reasoning')),
        'usage': result.get('usage'),
        'finish_reason': choice.get('finish_reason'),
        'model': result.get('model'),
        'system_fingerprint': result.get('system_fingerprint'),
        'service_wall_seconds': time.perf_counter() - started,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prediction_origin': 'new_generation',
        'failed': False,
    }
    check_prediction(row, job, params['model'])
    return row


def generate_job(job, params, endpoint, attempts):
    errors = []
    for attempt in range(attempts):
        try:
            result = request_answer(job, params, endpoint)
            result['attempt_errors'] = errors
            return result
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise ValueError(f'Reader rejected frozen input with HTTP {error.code}') from error
            errors.append({'attempt': attempt + 1, 'error': repr(error)})
            if attempt + 1 < attempts:
                time.sleep(min(30, 2 ** attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            errors.append({'attempt': attempt + 1, 'error': repr(error)})
            if attempt + 1 < attempts:
                time.sleep(min(30, 2 ** attempt))
    return {
        'input_key': job['input_key'],
        'dataset': job['dataset'],
        'id': job['id'],
        'failed': True,
        'attempt_errors': errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = root / 'outputs'
    output.mkdir(exist_ok=True)
    lock = (root / 'generation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = root / 'analysis_plan.json'
    plan_hash = sha256(plan_path)
    plan = json.loads(plan_path.read_text())
    assert plan['status'] == 'frozen_before_new_reader_generation'
    for name, expected in plan['files'].items():
        if sha256(root / name) != expected:
            raise ValueError('Frozen file changed ' + name)

    jobs = index_rows(load_rows(root / 'inputs/unique_inputs.jsonl.gz'))
    history = index_rows(load_rows(root / 'inputs/history_selected_predictions.jsonl.gz'))
    destination = output / 'predictions.jsonl'
    completed = load_checkpoint(destination)
    assert set(history).issubset(jobs) and set(completed).issubset(jobs)
    assert not set(history).intersection(completed)
    params = plan['request_params']
    reader = json.loads((root / 'inputs/reader_config.json').read_text())
    assert params == reader['params'] == plan['reader_config']['params']
    assert len(jobs) == plan['input_counts']['unique_reader_inputs']
    assert len(history) == plan['input_counts']['unique_reusable_inputs']
    for key, row in {**history, **completed}.items():
        check_prediction(row, jobs[key], params['model'])

    pending = [row for key, row in jobs.items() if key not in history and key not in completed]
    seed = plan['generation']['question_order_seed']
    pending.sort(key=lambda row: (
        hashlib.sha256(f"{seed}:{row['dataset']}:{row['id']}".encode()).hexdigest(),
        row['input_key'],
    ))
    started = time.time()
    failures = []

    def write_progress(phase):
        atomic_json(root / 'generation_progress.json', {
            'phase': phase,
            'analysis_plan_sha256': plan_hash,
            'group_rows': 45600,
            'unique_inputs': len(jobs),
            'history_unique': len(history),
            'generated_unique': len(completed),
            'remaining_unique': len(jobs) - len(history) - len(completed),
            'failed_unique_this_execution': len(failures),
            'execution_elapsed_seconds': time.time() - started,
            'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

    write_progress('generating')
    with destination.open('a', buffering=1) as stream, (output / 'failed_attempts.jsonl').open('a', buffering=1) as errors:
        with concurrent.futures.ThreadPoolExecutor(max_workers=plan['generation']['concurrency']) as pool:
            futures = {
                pool.submit(generate_job, job, params, plan['generation']['endpoint'], plan['generation']['max_attempts']): job['input_key']
                for job in pending
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    if row.get('failed'):
                        failures.append(row['input_key'])
                        errors.write(json.dumps(row, ensure_ascii=False) + '\n')
                    else:
                        key = row['input_key']
                        assert key not in completed and key not in history
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        completed[key] = row
                    if (len(completed) + len(failures)) % 25 == 0:
                        write_progress('generating')
                    if len(completed) % 100 == 0 and not row.get('failed'):
                        print(json.dumps({'generated_unique': len(completed), 'remaining_unique': len(jobs) - len(history) - len(completed), 'elapsed_seconds': time.time() - started}), flush=True)
            except BaseException as error:
                key = futures[future]
                failures.append(key)
                errors.write(json.dumps({
                    'input_key': key,
                    'failed': True,
                    'integrity_error': repr(error),
                    'analysis_plan_sha256': plan_hash,
                }, ensure_ascii=False) + '\n')
                pool.shutdown(wait=True, cancel_futures=True)
                write_progress('stopped_on_integrity_error')
                raise

    if failures:
        write_progress('retry_required')
        raise RuntimeError(f'{len(failures)} inputs remain after transient retries. Successful predictions are preserved.')
    assert set(jobs) == set(history) | set(completed)
    assert sha256(plan_path) == plan_hash
    for name, expected in plan['files'].items():
        if sha256(root / name) != expected:
            raise ValueError('Frozen file changed during generation ' + name)
    result = {
        'status': 'complete',
        'analysis_plan_sha256': plan_hash,
        'predictions_path': 'outputs/predictions.jsonl',
        'predictions_sha256': sha256(destination),
        'unique_inputs': len(jobs),
        'generated_unique': len(completed),
        'history_unique': len(history),
        'group_rows': 45600,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json(root / 'generation_complete.json', result)
    write_progress('complete')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
