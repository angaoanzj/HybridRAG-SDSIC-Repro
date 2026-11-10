"""Own only this version's3 workers; immutable sources and launch guard."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
R=Path(__file__).resolve().parents[1];PY='/root/autodl-tmp/hybridrag-multiround-v2-20261107/venv/bin/python'
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def main():
    assert (R/'2wiki_preflight.json').exists();assert not (R/'run_complete.json').exists();jobs={};cfg=json.loads((R/'protocol.json').read_text());frozen=json.loads((R/'execution_sources.json').read_text())
    for n,h in frozen.items():assert hashlib.sha256((R/'code'/n).read_bytes()).hexdigest()==h
    env=dict(os.environ,OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    for ds in ['2wiki','hotpotqa','musique']:
        assert not (R/f'{ds}_launched.json').exists(),'Existing worker: attach instead of duplicate'
        with (R/f'{ds}.log').open('wb',buffering=0) as log:p=subprocess.Popen([PY,'-u','-B',str(R/'code/full_train.py'),ds],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[ds]=p;write(R/f'{ds}_launched.json',{'pid':p.pid,'unix':time.time(),'version':cfg['version']})
    while jobs:
        for ds,p in list(jobs.items()):
            rc=p.poll()
            if rc is None:continue
            if rc or not (R/f'{ds}_complete.json').exists():
                write(R/'run_failed.json',{'dataset':ds,'exit_code':rc,'other_workers':'left running, not killed; inspect individual status'});raise RuntimeError(ds+' failed')
            del jobs[ds]
        if jobs:time.sleep(15)
    completed={ds:json.loads((R/f'{ds}_complete.json').read_text()) for ds in cfg['datasets']}
    assert all(x['train_questions']==cfg['datasets'][ds]['train_n'] and x['validation_questions']==300 for ds,x in completed.items())
    write(R/'run_complete.json',{'version':cfg['version'],'status':'complete','unix':time.time(),'train_questions':sum(x['train_questions'] for x in completed.values()),'validation_questions':900,'test_questions':0,'records':sum(x['validation_records'] for x in completed.values()),'model_files':6,'scope':'all available canonical labels except300 validation/corpus; validation-selected results, no heldout test'})
if __name__=='__main__':main()
