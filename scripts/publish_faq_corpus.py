"""Update only existing CloudCare FAQ IDs, transactionally, then Redis version."""
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cloudcare.settings import Settings
from cloudcare.storage import MySQLStore,RedisStore,_canonical_query,_digest,_now

def main():
    settings=Settings.load();sql=MySQLStore(settings);redis=RedisStore(settings)
    out=ROOT/'evaluation/query_upgrade_20261002'
    raw=gzip.decompress((out/'faq_candidate.jsonl.gz').read_bytes())
    rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
    existing={r['id']:r for r in sql.list_faq()}
    if not set(r['id'] for r in rows).issubset(existing): raise ValueError('Expected existing FAQ IDs only')
    hashes=[_digest(_canonical_query(r['question'])) for r in rows]
    if len(set(hashes))!=len(hashes):raise ValueError('Duplicate canonical questions')
    before=sql.counts()
    for row in rows:
        old=existing[row['id']]
        if old['answer']!=row['answer'] or old['source_id']!=row['source_id']:raise ValueError('Answer/source mutation forbidden here')
        row['category']=old['category']
    # A failed validation/SQL statement cannot leave a partially replaced SQL FAQ.
    with sql.engine.begin() as conn:
        for row,question_hash in zip(rows,hashes):
            conn.execute(sql.faq.update().where(sql.faq.c.id==row['id']).values(question=row['question'],
                question_hash=question_hash,metadata=row,updated_at=_now()))
    published=redis.replace_faq_index(sql.list_faq())
    after=sql.counts()
    if before!=after:raise ValueError('FAQ-only update unexpectedly changed counts')
    (ROOT/'data/faq.jsonl').write_bytes(raw)
    (ROOT/'data/faq.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    (out/'faq_publication.json').write_text(json.dumps({'before':before,'after':after,'redis':published,
        'normalization':'bm25_query_reference','threshold':settings.faq_bm25_threshold},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps({'faq_rows':len(rows),'counts_unchanged':before==after,'threshold':settings.faq_bm25_threshold}))
    sql.close();redis.close()
if __name__=='__main__':main()
