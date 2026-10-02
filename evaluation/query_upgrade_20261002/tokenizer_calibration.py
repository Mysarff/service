"""Exploratory tokenizer choice on known development/challenge data."""
import json,math,re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate_faq import read,stats,ROOT,OUT,FAQBM25Index
import jieba
cases=json.loads((OUT/'faq_final.json').read_text(encoding='utf-8'))['rows']
old=read(ROOT/'evaluation/faq_softmax_20261002/results.jsonl')
expected={r['id']:r.get('expected_source_ids',[]) for r in old}
results={}
for method in ('lcut','lcut_for_search'):
    class Index(FAQBM25Index):
        @staticmethod
        def tokenize(text):
            return [x for x in getattr(jieba,method)(str(text).lower()) if re.fullmatch(r'[a-z0-9_\-]+|[\u4e00-\u9fff]{2,}',x)]
    index=Index(read(ROOT/'data/faq.jsonl.gz'));rows=[]
    for case in cases:
        hit=index.search(case['query']);rows.append({**case,'score':hit['score'] if hit else 0,
            'source_id':hit['source_id'] if hit else None,'correct':bool(hit and hit['source_id'] in expected[case['id']])})
    dev=[r for r in rows if r['partition'] in ('development','challenge')];held=[r for r in rows if r['partition']=='retained']
    sweep=[stats(dev,i/100) for i in range(1,100)]
    feasible=[r for r in sweep if r['accepted'] and r['precision']>=.95 and r['negative_accepts']==0]
    choice=max(feasible,key=lambda r:(r['correct'],r['precision'],r['threshold'])) if feasible else None
    results[method]={'selected':choice,'retained':stats(held,choice['threshold']) if choice else None,'rows':rows,'sweep':sweep}
    print(method,choice,stats(held,choice['threshold']) if choice else None)
(OUT/'tokenizer_calibration.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
