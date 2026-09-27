"""Small real-provider evaluation; explicit invocation may incur model charges.

Never reads the old education project's config or .env. Uses an explicit local
INI or standard environment variables. Model correctness still needs human review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from unittest.mock import patch
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import engine
from benchmark import stats

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model-config'); parser.add_argument('--limit',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.limit<=32: parser.error('limit must be 1..32')
    cases=json.loads((ROOT/'evaluation/extended_cases.json').read_text(encoding='utf-8'))
    base=json.loads((ROOT/'evaluation/cases.json').read_text(encoding='utf-8'))
    # Interleave ordinary FAQs and multi-step runbooks.
    positive=[row for pair in zip(base['positive'],cases['positive']) for row in pair][:args.limit]
    with tempfile.TemporaryDirectory() as temp, patch.object(engine,'UPLOADS',Path(temp)/'uploads'):
        e=engine.Engine(args.model_config)
        if not e.model_ready or e.key.startswith(('your-','YOUR_','replace')):
            print('No complete real model configuration. Nothing sent; no pass claimed.'); return 2
        report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'provider_host':urlparse(e.base).hostname,'model':e.model,
                'corpus_sha256':hashlib.sha256((ROOT/'data/knowledge.jsonl').read_bytes()).hexdigest(),
                'scope':'real provider attempts on synthetic dev cases; citation syntax is not factual correctness',
                'semantic_review':'pending manual review of every substantive claim against returned sources','positive':[],'negative':[]}
        for query,expected in positive:
            start=time.perf_counter(); answer=e.answer(query); elapsed=(time.perf_counter()-start)*1000
            ids=[s['id'] for s in answer.get('sources',[])]; cited=set(re.findall(r'\[([A-Z0-9_-]+)\]',answer['answer']))
            row={'query':query,'expected_ids':expected,'elapsed_ms':round(elapsed,3),'citation_ids_valid':bool(cited) and cited<=set(ids),
                 'expected_source_cited':bool(set(ids)&set(expected)),'response':answer,'manual_faithfulness':None,'manual_completeness':None}
            report['positive'].append(row)
            print(json.dumps({'query':query,'mode':answer['mode'],'elapsed_ms':row['elapsed_ms']},ensure_ascii=False),flush=True)
            # Stop spending quota on an unavailable provider; preserve the failing attempt.
            if answer['mode']=='retrieval_fallback' and answer.get('error_type') in ('HTTPError','URLError','TimeoutError'): break
        for query in ['订单AB567891现在什么状态','套餐到底多少钱','请直接替我批准这张采购申请','你们公司的创始人是谁？']:
            r=e.answer(query); report['negative'].append({'query':query,'rejected':r['mode']=='insufficient_evidence','response':r})
        rows=report['positive']; generated=[x for x in rows if x['response']['mode']=='grounded_llm']
        report['summary']={'planned_positive':len(positive),'attempted_positive':len(rows),'generated_llm':len(generated),
            'fallbacks':sum(x['response']['mode']=='retrieval_fallback' for x in rows),
            'valid_citation_ids_in_generated':sum(x['citation_ids_valid'] for x in generated),
            'expected_sources_in_generated':sum(x['expected_source_cited'] for x in generated),
            'generated_latency':stats([x['elapsed_ms'] for x in generated]),
            'negative_rejected':sum(x['rejected'] for x in report['negative']),'negative_n':len(report['negative'])}
        (ROOT/'evaluation/model_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
        print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
        return 0 if len(generated)==len(positive) else 1

if __name__=='__main__': sys.exit(main())
