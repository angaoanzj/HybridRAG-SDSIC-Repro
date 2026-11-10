import json,hashlib,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 src=ROOT/'musique_ans_v1.0_dev.jsonl';assert sha(src)=='15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b'
 rows=[json.loads(x) for x in src.open()];docs={};labels=[];public=[]
 for q in rows:
  gold=[];titles=[]
  for p in q['paragraphs']:
   title,text=p['title'],p['paragraph_text'];key=hashlib.sha256((title+'\0'+text).encode()).hexdigest();i='musique_'+key
   sentences=[s for s in re.split(r'(?<=[.!?])\s+',text) if s]
   docs[i]={'doc_id':i,'title':title,'text':text,'sentences':sentences}
   if p['is_supporting']:gold.append(i);titles.append(title)
  assert gold and q['answerable']
  labels.append({'id':q['id'],'gold_doc_ids':sorted(set(gold)),'gold_titles':sorted(set(titles)),'hops':len(q['question_decomposition'])})
  public.append({'id':q['id'],'question':q['question'],'split':'official_dev'})
 out=ROOT/'data/musique';out.mkdir(parents=True,exist_ok=True)
 with (out/'corpus.jsonl').open('w') as f:
  for i in sorted(docs):f.write(json.dumps(docs[i],ensure_ascii=False)+'\n')
 (out/'public.json').write_text(json.dumps(public,ensure_ascii=False));(out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False))
 stats={'source':'MuSiQue-Ans v1.0 official dev; HF mirror with cross-mirror LFS SHA256 match; Google Drive download unavailable','source_sha256':sha(src),'queries':len(public),'hops':dict(collections.Counter(q['hops'] for q in labels)),'documents':len(docs),'unique_titles':len(set(d['title'] for d in docs.values())),'raw_contexts':sum(len(q['paragraphs']) for q in rows),'dedup':'exact title + NUL + exact paragraph text; all dev contexts, no label filtering','corpus_sha256':sha(out/'corpus.jsonl'),'public_sha256':sha(out/'public.json'),'labels_sha256':sha(out/'labels.json'),'sentences':'deterministic split after . ! ? plus whitespace, exact substrings; no gold sentence access'}
 (ROOT/'data_manifest.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats))
if __name__=='__main__':main()
