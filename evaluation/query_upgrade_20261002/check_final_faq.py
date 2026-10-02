"""Verify actual default normalization/threshold against all frozen regressions."""
import json,gzip,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrate_faq import read,stats,ROOT,OUT,FAQBM25Index
from cloudcare.settings import Settings
from cloudcare.storage import RedisStore

settings=Settings.load();index=FAQBM25Index(read(ROOT/'data/faq.jsonl.gz'))
cases=read(ROOT/'evaluation/faq_softmax_20261002/results.jsonl');rows=[]
redis=RedisStore(settings);version=redis.faq_index_version()
try:
    for case in cases:
        hit=index.search(case['query']);live=redis.find_faq(case['query'],min_score=0.,index_version=version)
        if bool(hit)!=bool(live) or hit and (hit['id']!=live['id'] or abs(hit['score']-live['score'])>1e-12):
            raise ValueError('Live FAQ differs from versioned file')
        rows.append({'id':case['id'],'partition':case.get('partition','challenge'),'kind':case['kind'],
            'query':case['query'],'score':hit['score'] if hit else 0,'raw_score':hit['raw_score'] if hit else 0,
            'reference_score':hit['reference_score'] if hit else None,'source_id':hit['source_id'] if hit else None,
            'correct':bool(hit and hit['source_id'] in case.get('expected_source_ids',[]))})
    threshold=settings.faq_bm25_threshold
    report={'algorithm':'faq_bm25_query_reference_v4','threshold':threshold,'redis_file_agreement':True,
            'faq_sha256':hashlib.sha256((ROOT/'data/faq.jsonl.gz').read_bytes()).hexdigest(),
            'protocol_note':'exploratory normalization selection; retained cases previously inspected, not blind',
            'development':stats([r for r in rows if r['partition']=='development'],threshold),
            'retained_regression':stats([r for r in rows if r['partition']=='retained'],threshold),
            'challenge':stats([r for r in rows if r['partition']=='challenge'],threshold),'rows':rows}
    (OUT/'faq_final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
finally:redis.close()
