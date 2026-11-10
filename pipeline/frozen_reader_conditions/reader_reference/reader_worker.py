import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import json,gzip,time,hashlib,random,traceback,urllib.request,sys,concurrent.futures,importlib.metadata
from pathlib import Path
from transformers import AutoTokenizer
R=Path('/root/reader-k-pilot-20261107');MODEL='/root/autodl-tmp/gemma-paper/models/gemma-4-E4B-it-qat-w4a16-ct'
SYSTEM='Answer the question using the supplied documents. Return only the short answer, without explanation. Use yes or no when appropriate. If the documents do not provide an answer, return unknown.'
PARAMS={'model':'reader-gemma4-e4b','temperature':0,'top_p':1,'top_k':-1,'max_tokens':64,'seed':2026110792,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1,'add_special_tokens':False,'chat_template_kwargs':{'enable_thinking':False}}
POLICIES=['no_second_branch','score_two_reverse'];KS=[20,50,100]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(name,v):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def load():
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 with gzip.open(R/'reader_inputs.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 return tok,rows

def chat_ids(tok,msg):
 enc=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=True)
 ids=enc['input_ids']
 if len(ids) and isinstance(ids[0],list):ids=ids[0]
 return ids

def service_token_count(msg):
 payload={'model':PARAMS['model'],'messages':msg,'add_generation_prompt':True,'add_special_tokens':False,'chat_template_kwargs':PARAMS['chat_template_kwargs']}
 req=urllib.request.Request('http://127.0.0.1:18271/tokenize',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as f:return json.load(f)['count']

def format_input(tok,row,policy,k):
 t0=time.perf_counter();docs=row['policies'][policy]['docs'][:k];parts=[];meta=[]
 for i,d in enumerate(docs,1):
  ids=tok.encode(d['text'],add_special_tokens=False);body=tok.decode(ids[:192],skip_special_tokens=True)
  title_ids=tok.encode(d['title'],add_special_tokens=False);title=tok.decode(title_ids[:32],skip_special_tokens=True)
  part=f'Document {i}\nTitle: {title}\n{body}'
  parts.append(part);meta.append({'doc_id':d['doc_id'],'title':d['title'],'body_original_tokens':len(ids),'body_used_tokens':min(192,len(ids)),'body_truncated':len(ids)>192,'body_sha256':hashlib.sha256(body.encode()).hexdigest()})
 def messages(p):return [{'role':'system','content':SYSTEM},{'role':'user','content':'\n\n'.join(p)+'\n\nQuestion: '+row['question']+'\nShort answer:'}]
 msg=messages(parts);ids=chat_ids(tok,msg)
 removed=0
 while len(ids)>24512 and parts:
  parts.pop();meta.pop();removed+=1;msg=messages(parts);ids=chat_ids(tok,msg)
 service_count=service_token_count(msg)
 while service_count>24512 and parts:
  parts.pop();meta.pop();removed+=1;msg=messages(parts);ids=chat_ids(tok,msg);service_count=service_token_count(msg)
 assert 50<service_count<=24512
 return msg,{'requested_K':k,'retained_doc_count':len(docs),'reader_doc_count':len(meta),'budget_removed_doc_count':removed,'body_truncated_doc_count':sum(x['body_truncated'] for x in meta),'tokenizer_input_tokens':service_count,'local_rendered_input_tokens':len(ids),'token_count_source':'same-reader service /tokenize chat endpoint; actual generation usage retained separately','documents':meta,'prompt_sha256':hashlib.sha256(json.dumps(msg,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'prompt_construction_seconds':time.perf_counter()-t0}

def request(msg,extra=None):
 body={**PARAMS,'messages':msg,**(extra or {})};start=time.perf_counter()
 req=urllib.request.Request('http://127.0.0.1:18271/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=900) as f:res=json.load(f)
 duration=time.perf_counter()-start;choice=res['choices'][0];message=choice['message'];content=message.get('content') or ''
 return {'prediction':content.strip(),'reasoning_content':message.get('reasoning_content',message.get('reasoning')),'usage':res.get('usage'),'finish_reason':choice.get('finish_reason'),'service_wall_seconds':duration,'model':res.get('model'),'system_fingerprint':res.get('system_fingerprint')}

def smoke():
 tok,rows=load();out=[]
 for ds in ['2wiki','hotpotqa']:
  row=next(x for x in rows if x['dataset']==ds);msg,meta=format_input(tok,row,POLICIES[0],20);res=request(msg);out.append({'dataset':ds,'id':row['id'],'question':row['question'],**meta,**res})
 atomic('smoke.json',{'status':'complete','examples':out});print(json.dumps([{k:v for k,v in x.items() if k not in ['documents','reasoning_content']} for x in out]))

def run():
 tok,rows=load();prep=json.loads((R/'input_preparation.json').read_text());rng=random.Random(2026110793);rng.shuffle(rows)
 atomic('run_progress.json',{'phase':'freezing_model','completed':0,'total':3600,'epoch':time.time()})
 model_hash=sha(Path(MODEL)/'model.safetensors')
 freeze={'status':'frozen_before_pilot_reader_outputs','scope':'development-set in-sample pilot, not independent holdout','model':MODEL,'model_weight_sha256':model_hash,'model_config_sha256':sha(Path(MODEL)/'config.json'),'tokenizer_sha256':sha(Path(MODEL)/'tokenizer.json'),'chat_template_sha256':sha(Path(MODEL)/'chat_template.jinja'),'params':PARAMS,'system_prompt':SYSTEM,'KS':KS,'policies':POLICIES,'question_count':600,'primary_generation_count':3600,'body_token_cap':192,'title_token_cap':32,'prompt_token_cap':24512,'quality_concurrency':4,'main_order_seed':2026110793,'sample_sha256':sha(R/'pilot_sample.json'),'reader_inputs_sha256':sha(R/'reader_inputs.jsonl.gz'),'repeat_questions_per_dataset':20,'repeat_total_per_condition':3,'timing_questions_per_dataset':20,'timing_repeats':3,'timing_cpu_threads':5,'latency_scope':'matched timing phase uses fresh retrieval and serial reader; quality wall time is under concurrency4 and is not end-to-end','software':{n:importlib.metadata.version(n) for n in ['torch','vllm','transformers','compressed-tensors']},'worker_sha256':sha(__file__),'code_sha256':{p.name:sha(p) for p in (R/'code').glob('*.py')},'frozen_epoch':time.time()}
 if (R/'run_freeze.json').exists():
  prev=json.loads((R/'run_freeze.json').read_text());assert prev['worker_sha256']==freeze['worker_sha256'] and prev['model_weight_sha256']==model_hash
 else:atomic('run_freeze.json',freeze)
 jobs=[]
 for row in rows:
  conditions=[(p,k) for p in POLICIES for k in KS];rng.shuffle(conditions)
  for p,k in conditions:
   msg,meta=format_input(tok,row,p,k);jobs.append({'dataset':row['dataset'],'id':row['id'],'policy':p,'K':k,'messages':msg,'meta':meta})
   if len(jobs)%60==0:atomic('run_progress.json',{'phase':'preparing_prompts','completed':len(jobs),'total':3600,'epoch':time.time()})
 with gzip.open(R/'prompt_manifest.jsonl.gz','wt') as f:
  for j in jobs:f.write(json.dumps(j,ensure_ascii=False)+'\n')
 atomic('prompt_manifest_receipt.json',{'sha256':sha(R/'prompt_manifest.jsonl.gz'),'jobs':len(jobs),'created_before_generations':True})
 done={}
 dest=R/'results/quality.jsonl'
 if dest.exists():
  for line in dest.open():
   x=json.loads(line);done[x['dataset'],x['id'],x['policy'],x['K']]=x
 pending=[j for j in jobs if (j['dataset'],j['id'],j['policy'],j['K']) not in done]
 started=time.time();errors=0
 def execute(j):
  result=None;attempts=[]
  for attempt in range(2):
   try:result=request(j['messages']);break
   except Exception as e:attempts.append({'attempt':attempt+1,'error':repr(e)});time.sleep(1)
  x={k:j[k] for k in ['dataset','id','policy','K']};x.update(j['meta']);x.update(result or {'prediction':'','failed':True});x['attempt_errors']=attempts;x['phase']='quality';return x
 with dest.open('a',buffering=1) as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(execute,j):j for j in pending}
  for future in concurrent.futures.as_completed(futures):
   x=future.result();f.write(json.dumps(x,ensure_ascii=False)+'\n');done[x['dataset'],x['id'],x['policy'],x['K']]=x;errors+=bool(x.get('failed'))
   atomic('run_progress.json',{'phase':'quality','completed':len(done),'total':3600,'errors_this_execution':errors,'elapsed_seconds':time.time()-started,'epoch':time.time()})
   if len(done)%60==0:print(json.dumps({'completed':len(done),'elapsed':time.time()-started}),flush=True)
 # Preselected repeatability cohort, never selected by observed answer quality.
 cohort={ds:sorted([r for r in rows if r['dataset']==ds],key=lambda r:hashlib.sha256(('repeat:'+r['id']).encode()).hexdigest())[:20] for ds in ['2wiki','hotpotqa']}
 repeats=R/'results/repeatability.jsonl';rdone=set()
 if repeats.exists():
  for l in repeats.open():
   x=json.loads(l);rdone.add((x['dataset'],x['id'],x['policy'],x['K'],x['repeat']))
 with repeats.open('a',buffering=1) as f:
  for ds in cohort:
   for row in cohort[ds]:
    for p in POLICIES:
     for k in KS:
      msg,meta=format_input(tok,row,p,k)
      for rep in [1,2]:
       key=(ds,row['id'],p,k,rep)
       if key in rdone:continue
       x={'dataset':ds,'id':row['id'],'policy':p,'K':k,'repeat':rep,'phase':'repeatability',**meta,**request(msg)};f.write(json.dumps(x,ensure_ascii=False)+'\n');rdone.add(key)
       atomic('run_progress.json',{'phase':'repeatability','completed':len(rdone),'total':480,'quality_completed':len(done),'epoch':time.time()})
 atomic('reader_quality_complete.json',{'status':'complete','quality_rows':len(done),'repeat_rows':len(rdone),'quality_sha256':sha(dest),'repeat_sha256':sha(repeats),'epoch':time.time()})
 # Isolated timing/scoring scripts are separate so failures cannot invalidate quality outputs.
 import subprocess
 for script in ['score_reader.py','timing_phase.py','score_reader.py']:
  p=subprocess.run([sys.executable,str(R/'code'/script)],cwd=R)
  if p.returncode:raise RuntimeError(f'{script} failed with code {p.returncode}; preserve completed quality outputs')
 atomic('run_complete.json',{'status':'complete','quality_rows':len(done),'repeat_rows':len(rdone),'epoch':time.time()})

if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1]=='smoke':smoke()
  else:run()
 except Exception:
  atomic('run_failed.json',{'traceback':traceback.format_exc(),'epoch':time.time()});raise
