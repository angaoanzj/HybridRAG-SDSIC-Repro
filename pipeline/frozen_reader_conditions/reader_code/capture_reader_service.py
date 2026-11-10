"""Capture the local reader server identity without changing its process."""

import argparse
import datetime
import json
from pathlib import Path
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--name', choices=['before', 'after'], required=True)
args = parser.parse_args()
receipt = args.root / 'receipts' / ('service_' + args.name + '.json')
assert not receipt.exists()
receipt.parent.mkdir(exist_ok=True)
processes = []
for directory in Path('/proc').glob('[0-9]*'):
    try:
        command = (directory / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    words = [part.decode() for part in command if part]
    if 'vllm.entrypoints.openai.api_server' in words and '--port' in words and words[words.index('--port') + 1] == '18271':
        processes.append({'pid': int(directory.name), 'command': words})
assert len(processes) == 1
with urllib.request.urlopen('http://127.0.0.1:18271/v1/models', timeout=30) as response:
    models = json.load(response)
assert any(model['id'] == 'reader-gemma4-e4b' for model in models['data'])
with urllib.request.urlopen('http://127.0.0.1:18271/metrics', timeout=30) as response:
    metrics = response.read().decode()
queue_lines = [line for line in metrics.splitlines() if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
config = json.loads((args.root / 'inputs/reader_config.json').read_text())
assert config['model'] in processes[0]['command']
model_files = {}
for name in ['model.safetensors', 'config.json', 'tokenizer.json', 'chat_template.jinja']:
    stat = (Path(config['model']) / name).stat()
    model_files[name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
receipt.write_text(json.dumps({
    'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'server_process': processes[0],
    'models_response': models,
    'queue_metrics': queue_lines,
    'reader_config': config,
    'model_file_stats': model_files,
}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'captured': str(receipt), 'pid': processes[0]['pid'], 'queue': queue_lines}), flush=True)
