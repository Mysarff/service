"""Reproducible local retrieval/HTTP benchmark. Never calls an external model.

Run: python evaluation/benchmark.py --requests 2000 --rounds 10
Reports raw observations, corpus/evaluation hashes, host and exact measurement scope.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
from urllib.request import Request, urlopen
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import engine
from app import Handler, make_server

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def percentile(xs, p):
    return round(sorted(xs)[max(0,math.ceil(len(xs)*p)-1)],3) if xs else None
def stats(xs):
    return dict(n=len(xs),mean_ms=round(statistics.mean(xs),3),p50_ms=percentile(xs,.5),p95_ms=percentile(xs,.95),max_ms=round(max(xs),3)) if xs else {}

def wilson(k,n):
    z=1.96; d=1+z*z/n; c=(k/n+z*z/(2*n))/d
    h=z*math.sqrt((k/n*(1-k/n)+z*z/(4*n))/n)/d
    return [round(c-h,4),round(c+h,4)]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--requests',type=int,default=2000); parser.add_argument('--rounds',type=int,default=10)
    args=parser.parse_args()
    if args.requests<8 or args.rounds<1: parser.error('requests >= 8 and rounds >= 1 required')
    groups={name:json.loads((ROOT/'evaluation'/name).read_text(encoding='utf-8')) for name in ('cases.json','extended_cases.json')}
    queries=[q for cases in groups.values() for q,_ in cases['positive']]
    rng=random.Random(20260927)
    # Never use user uploads in a published benchmark and never read a model config.
    with tempfile.TemporaryDirectory() as temp, patch.object(engine,'UPLOADS',Path(temp)/'uploads'):
        tracemalloc.start(); start=time.perf_counter(); e=engine.Engine(); e.key=''
        build_ms=(time.perf_counter()-start)*1000; current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'synthetic development audit; local loopback extractive HTTP; no external LLM; no production or 800k claim',
                'host':{'os':platform.platform(),'python':platform.python_version(),'processor':platform.processor(),'logical_cpus':os.cpu_count()},
                'seed':20260927,'corpus_units':len(e.docs),'corpus_sha256':sha(ROOT/'data/knowledge.jsonl'),
                'cases_sha256':{name:sha(ROOT/'evaluation'/name) for name in groups},
                'code_sha256':{name:sha(ROOT/name) for name in ('engine.py','app.py','evaluation/benchmark.py')},
                'index':{'build_ms_with_tracemalloc':round(build_ms,3),'traced_live_bytes':current,'traced_peak_bytes':peak,'vocabulary_terms':len(e.df),'scope':'Python allocations, not total process RSS; one build with tracing overhead'},
                'retrieval':{},'negative_evidence_gate':{},'search_performance':{},'http':[]}
        for name,cases in groups.items():
            report['retrieval'][name]={}
            for method in ('tfidf','plain_bm25','bm25'):
                observations=[]
                for q,expected in cases['positive']:
                    hits=e.search(q,method=method); ids=[x['id'] for x in hits]
                    rank=min((ids.index(x)+1 for x in expected if x in ids),default=0)
                    observations.append({'query':q,'expected':expected,'ids':ids,'rank':rank,'top1_correct':bool(ids and ids[0] in expected)})
                n=len(observations); k=sum(x['top1_correct'] for x in observations)
                report['retrieval'][name][method]={'n':n,'top1':k,'top1_rate':round(k/n,4),'top1_wilson95':wilson(k,n),
                    'hit5':sum(x['rank']>0 for x in observations),'mrr5':round(sum(1/x['rank'] if x['rank'] else 0 for x in observations)/n,4),'observations':observations}
            negatives=[]
            for q in cases['negative']:
                r=e.answer(q,force_extract=True)
                negatives.append({'query':q,'rejected':r['mode']=='insufficient_evidence','mode':r['mode'],'answer':r['answer'],'sources':[x['id'] for x in r['sources']]})
            report['negative_evidence_gate'][name]={'n':len(negatives),'rejected':sum(x['rejected'] for x in negatives),'observations':negatives}
        # Warm both implementations; alternate randomized order to reduce ordering bias.
        for q in queries:
            for strategy in ('scan','inverted'): e.search(q,strategy=strategy)
        timings={'scan':[],'inverted':[]}; parity=[]
        for round_id in range(args.rounds):
            shuffled=queries[:]; rng.shuffle(shuffled)
            for q in shuffled:
                strategies=['scan','inverted']; rng.shuffle(strategies); outputs={}
                for strategy in strategies:
                    start=time.perf_counter(); hits=e.search(q,strategy=strategy); elapsed=(time.perf_counter()-start)*1000
                    timings[strategy].append({'round':round_id,'query':q,'ms':round(elapsed,6)})
                    outputs[strategy]=[(x['id'],x['score']) for x in hits]
                parity.append(outputs['scan']==outputs['inverted'])
        report['search_performance']={s:{**stats([x['ms'] for x in rows]),'raw':rows} for s,rows in timings.items()}
        report['ranking_parity']={'passed':sum(parity),'n':len(parity)}
        with patch.object(Handler,'log_message',lambda *a:None):
            server=make_server(0); server.engine.key=''
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            url=f'http://127.0.0.1:{server.server_port}/api/chat'
            def request(i):
                q=queries[i%len(queries)]; start=time.perf_counter()
                try:
                    req=Request(url,data=json.dumps({'query':q}).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(req,timeout=15) as response: status=response.status; result=json.load(response)
                    return {'index':i,'query':q,'ms':round((time.perf_counter()-start)*1000,3),'status':status,'mode':result['mode'],'ok':status==200,'answered_from_evidence':result['mode']=='retrieval_only'}
                except Exception as exc:
                    return {'index':i,'query':q,'ms':round((time.perf_counter()-start)*1000,3),'ok':False,'error_type':type(exc).__name__}
            try:
                for i in range(20): request(i)
                for concurrency in (1,4,8):
                    start=time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool: rows=list(pool.map(request,range(args.requests)))
                    wall=time.perf_counter()-start
                    report['http'].append({'concurrency':concurrency,'requests':len(rows),'http_successful':sum(x['ok'] for x in rows),'evidence_answers':sum(x.get('answered_from_evidence',False) for x in rows),
                        'wall_seconds':round(wall,3),'successful_requests_per_second':round(sum(x['ok'] for x in rows)/wall,2),
                        'latency_all_requests':stats([x['ms'] for x in rows]),'modes':dict(Counter(x.get('mode','error') for x in rows)),'raw':rows})
            finally: server.shutdown(); server.server_close(); thread.join()
    path=ROOT/'evaluation/benchmark_results.json'; path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    compact={k:v for k,v in report.items() if k not in ('retrieval','negative_evidence_gate','search_performance','http')}
    compact['retrieval']={g:{m:{k:v for k,v in r.items() if k!='observations'} for m,r in values.items()} for g,values in report['retrieval'].items()}
    compact['negative_evidence_gate']={g:{'n':r['n'],'rejected':r['rejected']} for g,r in report['negative_evidence_gate'].items()}
    compact['search_performance']={m:{k:v for k,v in r.items() if k!='raw'} for m,r in report['search_performance'].items()}
    compact['http']=[{k:v for k,v in r.items() if k!='raw'} for r in report['http']]
    (ROOT/'evaluation/benchmark_summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(compact,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
