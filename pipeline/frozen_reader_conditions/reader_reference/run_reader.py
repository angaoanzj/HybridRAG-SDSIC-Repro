import sys,json,gzip,hashlib,random,time,concurrent.futures,traceback
from pathlib import Path
R=Path('/root/autodl-tmp/reader_matched_20261105_side');sys.path.insert(0,'/root/reader-k-pilot-20261107/code');import reader_worker as w
w.R=R
r=json.loads((R/'protocol.json').read_text());h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert h(R/'reader_inputs.jsonl.gz')==r['input_sha256'] and h('/root/reader-k-pilot-20261107/code/reader_worker.py')==r['reader_worker_sha256']
tok,rows=w.load(); assert len(rows)==600
jobs=[(x,p) for x in rows for p in r['policies']];random.Random(2026110793).shuffle(jobs)
if len(sys.argv)>1 and sys.argv[1]=='smoke':jobs=[next((x,p) for x,p in jobs if x['dataset']==ds and p=='no_second_branch') for ds in ('2wiki','hotpotqa')]
name='smoke.jsonl' if len(sys.argv)>1 and sys.argv[1]=='smoke' else 'quality.jsonl';dest=R/name;assert not dest.exists()
(R/(name+'.plan.json')).write_text(json.dumps({'jobs':len(jobs),'worker_sha256':h(__file__),'protocol_sha256':h(R/'protocol.json'),'order_sha256':hashlib.sha256(json.dumps([(x['dataset'],x['id'],p) for x,p in jobs]).encode()).hexdigest(),'created_before_generation':True},indent=2)+'\n')
def work(j):
 x,p=j;meta=None
 try:
  msg,meta=w.format_input(tok,x,p,50);v=w.request(msg);return {'dataset':x['dataset'],'id':x['id'],'policy':p,'K':50,**meta,**v,'failed':False}
 except Exception:return {'dataset':x['dataset'],'id':x['id'],'policy':p,'K':50,'failed':True,'error':traceback.format_exc()}
with dest.open('w',buffering=1) as f,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 fs=[pool.submit(work,j) for j in jobs]; n=0;failed=0
 for future in concurrent.futures.as_completed(fs):
  x=future.result();f.write(json.dumps(x,ensure_ascii=False)+'\n');n+=1;failed+=int(x['failed'])
  if n%25==0 or n==len(jobs):print(json.dumps({'completed':n,'total':len(jobs),'failed':failed}),flush=True)
(R/(name+'.receipt.json')).write_text(json.dumps({'rows':n,'errors':failed,'sha256':h(dest),'status':'complete' if n==len(jobs) and failed==0 else 'failed'},indent=2)+'\n')
if failed:raise SystemExit(2)
