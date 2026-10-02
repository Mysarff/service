import json, math
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate_faq import read,stats,ROOT,OUT,FAQBM25Index
data=json.loads((OUT/'faq_calibration.json').read_text(encoding='utf-8'))
index=FAQBM25Index(read(OUT/'faq_candidate.jsonl.gz'))
results={}
for variant in ('self_ratio','self_ratio_times_margin','upper_bound'):
    rows=[]
    for r in data['natural_variants_grouped']['rows']:
        row=dict(r);bag=Counter(index.tokenize(row['question']));n=len(index.rows)
        idfs={t:math.log(1+(n-len(index.postings.get(t,()))+.5)/(len(index.postings.get(t,()))+.5)) for t in bag}
        norm=1.2*(.25+.75*sum(bag.values())/index.avg_length)
        perfect=sum(idfs[t]*bag[t]*2.2/(bag[t]+norm) for t in bag)
        ratio=min(1,row['raw_score']/max(perfect,1e-10))
        row['score']=ratio if variant=='self_ratio' else ratio*row['score'] if variant=='self_ratio_times_margin' else row['raw_score']/max(sum(idfs.values())*2.2,1e-10)
        rows.append(row)
    dev=[r for r in rows if r['partition']=='development'];held=[r for r in rows if r['partition']=='retained']
    sweep=[stats(dev,i/100) for i in range(1,100)]
    feasible=[r for r in sweep if r['accepted'] and r['precision']>=.95 and r['negative_accepts']==0]
    choice=max(feasible,key=lambda r:(r['correct'],r['precision'],r['threshold'])) if feasible else None
    results[variant]={'dev':choice,'retained':stats(held,choice['threshold']) if choice else None,'rows':rows,'sweep':sweep}
    print(variant,choice,stats(held,choice['threshold']) if choice else None)
(OUT/'normalization_experiment.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
