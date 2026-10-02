"""Real .55 FAQ/RAG routes with dedicated stores and neural models, no Qwen calls."""
from pathlib import Path
import hashlib
import gzip
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from sqlalchemy import delete

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cloudcare.pipeline import SupportPipeline
from cloudcare.settings import Settings

def main():
    output = Path(__file__).with_name('live_routes.json')
    settings = Settings.load()
    assert settings.faq_bm25_threshold == .55
    input_path = ROOT/'evaluation/faq_softmax_20261002/results.jsonl'
    raw = input_path.read_bytes() if input_path.exists() else gzip.decompress(Path(str(input_path)+'.gz').read_bytes())
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    selected = ['cases.json:positive:2', 'cases.json:positive:6', 'cases.json:positive:20',
                'cases.json:positive:9', 'cases.json:positive:0', 'missing-01', 'missing-05',
                'extended_cases.json:negative:9']
    by_id = {row['id']: row for row in rows}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'threshold': .55,
              'scope': 'Eight targeted real BERT/MySQL/Redis/FAQ and neural RAG routes; Qwen disabled; not a representative answer accuracy benchmark.',
              'selection': selected, 'observations': [], 'complete': False,
              'code_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                 for name in ('cloudcare/settings.py','cloudcare/faq.py','cloudcare/pipeline.py','cloudcare/storage.py')}}
    pipeline = SupportPipeline(settings)
    sessions, cache_keys = set(), set()
    original_cache = pipeline.redis.set_cached_answer
    def cache(key, result, ttl=None):
        actual_key = pipeline.redis._key('answer', key)
        assert not pipeline.redis.client.exists(actual_key)
        cache_keys.add(actual_key)
        return original_cache(key, result, ttl)
    pipeline.redis.set_cached_answer = cache
    pipeline._cache_config += ':threshold-probe:' + uuid.uuid4().hex
    started = time.perf_counter()
    try:
        pipeline.start()
        report['storage_before'] = pipeline.sql.counts()
        for identity in selected:
            case = by_id[identity]
            session_id = uuid.uuid4().hex
            assert not pipeline.sql.get_session(session_id)
            sessions.add(session_id)
            result = pipeline.answer(case['query'], session_id=session_id, force_extract=True)
            source_ids = [row['id'] for row in result['sources']]
            report['observations'].append({'case_id': identity, 'query': case['query'],
                'expected_source_ids': case['expected_source_ids'], 'source_id_hit': bool(set(source_ids)&set(case['expected_source_ids'])),
                'source_ids': source_ids, 'answer': result['answer'], 'mode': result['mode'],
                'trace': result['trace'], 'elapsed_ms': result['elapsed_ms']})
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps({'case_id':identity, 'mode':result['mode'], 'score':result['trace'].get('faq',{}).get('score')}, ensure_ascii=True), flush=True)
        report['provider_calls'] = pipeline.llm.calls
        assert report['provider_calls'] == 0
        assert all(not row['trace']['cache_hit'] for row in report['observations'])
        report['complete'] = True
    finally:
        # Only exact random session IDs and fresh probe cache keys owned by this run.
        with pipeline.sql.engine.begin() as connection:
            for session_id in sessions:
                connection.execute(delete(pipeline.sql.messages).where(pipeline.sql.messages.c.session_id == session_id))
                connection.execute(delete(pipeline.sql.sessions).where(pipeline.sql.sessions.c.id == session_id))
        owned_redis = list(cache_keys) + [pipeline.redis._key('session', sid) for sid in sessions]
        if owned_redis:
            pipeline.redis.client.delete(*owned_redis)
        report['storage_after_cleanup'] = pipeline.sql.counts()
        report['wall_seconds'] = time.perf_counter()-started
        for model in (pipeline.router, pipeline.vector.encoder, pipeline.reranker):
            pipeline._release(model)
        pipeline.close()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__':
    main()
