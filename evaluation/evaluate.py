"""Same-corpus retrieval controls. This does not measure generated-answer correctness."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from engine import Engine

def main():
    cases=json.loads((ROOT/'evaluation/cases.json').read_text(encoding='utf-8'))
    engine=Engine(); report={'corpus_units':len(engine.docs),'scope':'synthetic development regression, not blind benchmark or LLM accuracy','methods':{}}
    for method in ['tfidf','plain_bm25','bm25']:
        top1=hit5=0; reciprocal=0; timings=[]; failures=[]
        for q,expected in cases['positive']:
            start=time.perf_counter(); found=engine.search(q,method=method); timings.append((time.perf_counter()-start)*1000)
            ids=[x['id'] for x in found]; ranks=[ids.index(x)+1 for x in expected if x in ids]
            top1+=bool(ids and ids[0] in expected); hit5+=bool(ranks); reciprocal+=1/min(ranks) if ranks else 0
            if not ids or ids[0] not in expected: failures.append({'query':q,'expected':expected,'top5':ids})
        report['methods'][method]={'n':len(cases['positive']),'top1':top1,'hit5':hit5,'mrr5':round(reciprocal/len(cases['positive']),4),
                                  'mean_retrieval_ms':round(sum(timings)/len(timings),3),'top1_failures':failures}
    negatives=[{'query':q,'rejected':not engine.supported(engine.search(q),q),'top1':[d['id'] for d in engine.search(q,limit=1)]} for q in cases['negative']]
    report['evidence_gate']={'n':len(negatives),'rejected':sum(d['rejected'] for d in negatives),'details':negatives}
    path=ROOT/'evaluation/results.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:({m:{a:b for a,b in v.items() if a!='top1_failures'} for m,v in x.items()} if k=='methods' else x) for k,x in report.items()},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
