"""Evaluate saved actual answers with real RAGAS/Qwen/BGE; resumable per record."""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('RAGAS_DO_NOT_TRACK','true')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

async def run(args):
    import httpx
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from cloudcare.settings import Settings
    from cloudcare.neural import BGEM3Encoder
    from cloudcare.evaluation import BGERagasEmbedding,score_record,summarize_scores
    settings=Settings.load()
    http=httpx.AsyncClient(proxy=settings.llm_proxy or None,timeout=90)
    client=AsyncOpenAI(api_key=settings.llm_api_key,base_url=settings.llm_base_url,http_client=http,max_retries=0)
    judge=llm_factory(settings.llm_model,client=client,temperature=0,max_tokens=4096,
                      extra_body={'enable_thinking':False})
    encoder=BGEM3Encoder(settings)
    metrics={'faithfulness':Faithfulness(llm=judge),
             'answer_relevancy':AnswerRelevancy(llm=judge,embeddings=BGERagasEmbedding(encoder),strictness=3),
             'context_precision':ContextPrecision(llm=judge),'context_recall':ContextRecall(llm=judge)}
    input_bytes=args.input.read_bytes()
    data=json.loads(input_bytes)
    records=data['records'] if isinstance(data,dict) else data
    records=[r for r in records if r.get('response') and r.get('retrieved_contexts')][:args.limit or None]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    previous=json.loads(args.output.read_text(encoding='utf-8')) if args.output.exists() else {}
    digest=hashlib.sha256(input_bytes).hexdigest()
    if previous and previous.get('input_sha256')!=digest: raise ValueError('Resume input has changed')
    done={r['id']:r for r in previous.get('records',[])}
    try:
        for record in records:
            if record['id'] in done: continue
            judged={**record,'metrics':await score_record(metrics,record)}
            done[record['id']]=judged
            output={'ragas_version':'0.4.3','judge_model':settings.llm_model,'embedding':'BGE-M3 local dense',
                    'same_generator_judge':True,'input_sha256':digest,
                    'scope':'synthetic local development set; not production accuracy',
                    'records':list(done.values()),'summary':summarize_scores(list(done.values()))}
            args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            print(json.dumps({'id':record['id'],'completed':len(done),'total':len(records),'metrics':judged['metrics']}),flush=True)
    finally:
        encoder.release_model(); await client.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=0)
    asyncio.run(run(parser.parse_args()))
