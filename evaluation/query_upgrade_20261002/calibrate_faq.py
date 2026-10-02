"""Compare frozen FAQ questions and new source-only paraphrases; select on dev."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from cloudcare.faq import FAQBM25Index
OUT=Path(__file__).resolve().parent

def read(path):
    raw=path.read_bytes() if path.is_file() else gzip.decompress(Path(str(path)+'.gz').read_bytes())
    if path.suffix=='.gz':raw=gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode('utf-8').splitlines()]

def stats(rows,t):
    accepted=[r for r in rows if r['score']>=t]
    correct=sum(r['correct'] for r in accepted)
    return {'threshold':t,'samples':len(rows),'accepted':len(accepted),'correct':correct,
            'precision':correct/len(accepted) if accepted else None,
            'positive_coverage':correct/sum(r['kind']=='positive' for r in rows),
            'negative_accepts':sum(r['kind']=='negative' for r in accepted),
            'errors':[r['id'] for r in accepted if not r['correct']]}

def main():
    cases=read(ROOT/'evaluation/faq_softmax_20261002/results.jsonl')
    source_cases={}
    for filename in ('cases.json','extended_cases.json'):
        data=json.loads((ROOT/'evaluation'/filename).read_text(encoding='utf-8'))
        for i,(question,ids) in enumerate(data['positive']): source_cases[f'{filename}:positive:{i}']=(question,ids)
        for i,question in enumerate(data['negative']):source_cases[f'{filename}:negative:{i}']=(question,[])
    knowledge=read(ROOT/'data/knowledge.jsonl');categories={r['id']:r['category'] for r in knowledge}
    datasets={'old_templates':read(OUT/'faq_original.jsonl.gz'),'natural_variants':read(OUT/'faq_candidate.jsonl.gz')}
    results={}
    for name,faqs in datasets.items():
        for row in faqs:row['category']=categories[row['source_id']]
        for grouped in (False,True):
            key=name+('_grouped' if grouped else '_all_faq')
            index=FAQBM25Index(faqs,group_answers=grouped,normalization='softmax');rows=[]
            for case in cases:
                if case['id'] not in source_cases:continue
                question,ids=source_cases[case['id']];hit=index.search(question)
                rows.append({'id':case['id'],'partition':case.get('partition'),'kind':case['kind'],
                    'question':question,'expected_ids':ids,'source_id':hit['source_id'] if hit else None,
                    'score':hit['score'] if hit else 0,'raw_score':hit['raw_score'] if hit else 0,
                    'correct':bool(hit and hit['source_id'] in ids)})
            dev=[r for r in rows if r['partition']=='development'];held=[r for r in rows if r['partition']=='retained']
            sweep=[stats(dev,i/100) for i in range(1,100)]
            feasible=[r for r in sweep if r['accepted'] and r['precision']>=.95 and r['negative_accepts']==0]
            choice=max(feasible,key=lambda r:(r['correct'],r['precision'],r['threshold'])) if feasible else None
            results[key]={'selected_on_development':choice,'retained_regression':stats(held,choice['threshold']) if choice else None,
                          'all_at_selected':stats(rows,choice['threshold']) if choice else None,
                          'diagnostics':[stats(rows,t) for t in (.55,.75,.85,.9,.95,.99)],'rows':rows,'sweep':sweep}
    (OUT/'faq_calibration.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,value in results.items():print(name,json.dumps({k:v for k,v in value.items() if k not in ('rows','sweep')},ensure_ascii=False))
if __name__=='__main__':main()
