from pathlib import Path
import json,gzip,hashlib,datetime
R=Path('/root/autodl-tmp/reader_matched_20261105_side');B=Path('/root/autodl-tmp/hybridrag-paper-repair-20261105'); policies=('hybrid_rrf','no_second_branch','bridge_expand')
path=R/'reader_inputs.jsonl.gz'; assert not path.exists()
with gzip.open(path,'wt') as out:
 for ds in ('2wiki','hotpotqa'):
  corpus=B/'data'/ds/'corpus.jsonl'; docs={x['doc_id']:x for x in map(json.loads,corpus.open())};q=json.loads((B/'data'/ds/'queries_public.json').read_text());ids=[x['id'] for x in q]; needed=set(ids); scored={}
  with gzip.open(B/'runs/confirmation'/ds/'results.jsonl.gz','rt') as f:
   for l in f:
    z=json.loads(l);p=z.get('policy');i=z['id']
    if p not in policies or i not in needed or z.get('alpha')!={'2wiki':.3,'hotpotqa':.7}[ds] or z.get('source_top_n')!=5:continue
    key=(i,p);assert key not in scored;sel=z['selections']['50'];assert len(sel)==50 and len(set(sel))==50
    scored[key]=[{'doc_id':d,'title':docs[d]['title'],'text':docs[d]['text']} for d in sel]
  assert len(scored)==900
  for row in q:
   out.write(json.dumps({'dataset':ds,'id':row['id'],'question':row['question'],'policies':{p:{'docs':scored[(row['id'],p)]} for p in policies}},ensure_ascii=False)+'\n')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();model=Path('/root/autodl-tmp/gemma-paper/models/gemma-4-E4B-it-qat-w4a16-ct');r={'status':'frozen_before_generation','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sample':'20261105 historical 300 questions each; not independent holdout','K':50,'policies':list(policies),'n_per_dataset':300,'generations':1800,'input_sha256':h(path),'model_sha256':h(model/'model.safetensors'),'tokenizer_sha256':h(model/'tokenizer.json'),'chat_template_sha256':h(model/'chat_template.jinja'),'prompt':'prior reader_worker SYSTEM; 192 body tokens, 32 title tokens, 24512 total tokens','decoding':'temperature 0, max 64, seed 2026110792','answer_labels':'kept offline, not passed to reader','execution':'same reader, independent tasks; no cost/latency comparison','prepare_sha256':h(Path(__file__)),'reader_worker_sha256':h(Path('/root/reader-k-pilot-20261107/code/reader_worker.py'))};(R/'protocol.json').write_text(json.dumps(r,indent=2)+'\n');print('prepared',h(path),'protocol',h(R/'protocol.json'))
