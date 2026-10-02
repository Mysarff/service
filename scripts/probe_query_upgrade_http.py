"""Live HTTP integration of current FAQ, multi-query RAG and server history."""
import json,gzip,time,uuid
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import httpx
from sqlalchemy import delete
from cloudcare.settings import Settings
from cloudcare.storage import MySQLStore,RedisStore
settings=Settings.load();sql=MySQLStore(settings);redis=RedisStore(settings)
output=ROOT/'evaluation/query_upgrade_20261002/live_http.json'
faqs=[json.loads(line) for line in gzip.decompress((ROOT/'data/faq.jsonl.gz').read_bytes()).decode().splitlines()]
question=next(r['question'] for r in faqs if r['id']=='FAQ-ACC-13-1')
sessions=[uuid.uuid4().hex for _ in range(4)]
cases=[('faq',question,sessions[0]),('multi','忘记密码要怎么重置，换了手机后验证器又该怎么迁移？',sessions[1]),
       ('unknown','密码重置邮件的链接具体有效多少分钟？',sessions[2]),
       ('history_first','我忘记了账号登录密码，怎么重新设置？',sessions[3]),
       ('history_followup','那链接过期了呢？',sessions[3])]
report={'before':sql.counts(),'observations':[],'complete':False}
try:
    assert all(not sql.get_session(sid) for sid in sessions)
    with httpx.Client(base_url='http://127.0.0.1:8088',timeout=180,trust_env=False) as client:
        report['health_before']=client.get('/api/health').json()
        for label,query,sid in cases:
            response=client.post('/api/chat',json={'query':query,'session_id':sid})
            response.raise_for_status();result=response.json()
            report['observations'].append({'label':label,'result':result})
            print(json.dumps({'label':label,'mode':result.get('mode'),'faq':result.get('trace',{}).get('faq'),
                'elapsed_ms':result.get('elapsed_ms')},ensure_ascii=False),flush=True)
        report['health_after']=client.get('/api/health').json()
        obs={r['label']:r['result'] for r in report['observations']}
        report['checks']={'faq_direct':obs['faq']['mode']=='faq',
            'multi_query_executed':len(obs['multi'].get('trace',{}).get('retrieval',{}).get('query_routes',[]))>2,
            'unknown_fact_not_faq':obs['unknown']['mode']!='faq',
            'unknown_fact_refused':'未提供' in obs['unknown']['answer'] or '不足' in obs['unknown']['answer'],
            'server_history_used':len(obs['history_followup'].get('history',[]))>=4}
        report['complete']=all(report['checks'].values())
finally:
    with sql.engine.begin() as conn:
        for sid in sessions:
            conn.execute(delete(sql.messages).where(sql.messages.c.session_id==sid))
            conn.execute(delete(sql.sessions).where(sql.sessions.c.id==sid))
    redis.client.delete(*[redis._key('session',sid) for sid in sessions])
    report['after_cleanup']=sql.counts()
    report['counts_preserved']=report['before']==report['after_cleanup']
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    sql.close();redis.close()
