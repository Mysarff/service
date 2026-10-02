"""Matched real retrieval/generation diagnostic. No FAQ cache or SQL writes.

Bypasses routing deliberately to measure both RAG strategies on identical cases;
live FAQ/router behavior is checked separately. This is a development ablation.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cloudcare.pipeline import SupportPipeline
from cloudcare.settings import Settings
from cloudcare.query import denoise_query
from cloudcare.llm import evidence_contexts

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'evaluation/query_upgrade_20261002/query_results.json')
    parser.add_argument('--variant',choices=['both','single_rewrite','multi_strategy'],default='both')
    args=parser.parse_args();settings=Settings.load()
    path=ROOT/'evaluation/query_upgrade_20261002/query_cases.json';raw=path.read_bytes();cases=json.loads(raw)
    pipeline=SupportPipeline(settings);pipeline.start()
    previous=json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else {}
    records=previous.get('records',[]);done={r['id'] for r in records}
    metadata={'case_sha256':hashlib.sha256(raw).hexdigest(),'model':settings.llm_model,'corpus_revision':pipeline._revision,
        'retrieval_k':settings.retrieval_k,'rerank_k':settings.rerank_k,'scope':'matched RAG development diagnostic; FAQ/router intentionally bypassed',
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'cloudcare').glob('*.py')}}
    if previous and any(previous.get(k)!=metadata[k] for k in ('case_sha256','corpus_revision','code_sha256')):
        raise ValueError('Resume inputs/code changed; use a new output file')
    try:
        for case in cases:
            for variant in ('single_rewrite','multi_strategy'):
                if args.variant != 'both' and variant != args.variant:continue
                key=case['id']+':'+variant
                if key in done:continue
                start=time.perf_counter();calls=pipeline.llm.calls
                history=case.get('history',[]);query=pipeline._legacy.contextual_query(denoise_query(case['query']),history)
                plan=None;error=None
                try:
                    plan=(pipeline.llm.plan if variant=='multi_strategy' else pipeline.llm.rewrite)(query,history)
                except Exception as exc:error=type(exc).__name__
                hits,trace=pipeline.retrieve(query,additional_queries=[plan['query']] if plan else [],
                    query_plan=plan if variant=='multi_strategy' else None)
                pipeline._release(pipeline.reranker)
                context=evidence_contexts(hits);response=None;generation_error=None
                try:response=pipeline.llm.generate(query,hits,history)
                except Exception as exc:generation_error=type(exc).__name__
                expected=set(case['expected_ids']);retrieved=[r['id'] for r in hits]
                record={'id':key,'case_id':case['id'],'kind':case['kind'],'variant':variant,
                    'user_input':query,'original_query':case['query'],'history':history,'response':response,
                    'retrieved_contexts':[r['rendered'] for r in context],'reference':case['reference'],
                    'expected_ids':sorted(expected),'retrieved_ids':retrieved,'plan':plan,'plan_error':error,
                    'generation_error':generation_error,'trace':trace,'api_calls':pipeline.llm.calls-calls,
                    'hit5':bool(expected&set(retrieved)) if expected else None,
                    'source_recall5':len(expected&set(retrieved))/len(expected) if expected else None,
                    'all_sources5':expected.issubset(retrieved) if expected else None,
                    'elapsed_ms':round((time.perf_counter()-start)*1000,2)}
                records.append(record)
                summary={}
                for v in ('single_rewrite','multi_strategy'):
                    subset=[r for r in records if r['variant']==v];positive=[r for r in subset if r['expected_ids']]
                    summary[v]={'samples':len(subset),'positive_samples':len(positive),
                        'hit5':sum(r['hit5'] for r in positive)/len(positive) if positive else None,
                        'all_sources5':sum(r['all_sources5'] for r in positive)/len(positive) if positive else None,
                        'source_recall5':sum(r['source_recall5'] for r in positive)/len(positive) if positive else None,
                        'generation_successes':sum(r['response'] is not None for r in subset),
                        'plan_successes':sum(r['plan_error'] is None for r in subset),
                        'api_calls':sum(r['api_calls'] for r in subset),
                        'mean_ms':sum(r['elapsed_ms'] for r in subset)/len(subset) if subset else None}
                args.output.write_text(json.dumps({**metadata,'summary':summary,'records':records},ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps({k:record[k] for k in ('id','hit5','all_sources5','plan_error','generation_error','elapsed_ms')},ensure_ascii=False),flush=True)
    finally:pipeline.close()
if __name__=='__main__':main()
