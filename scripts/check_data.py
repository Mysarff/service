"""Validate corpus lineage, deterministic checksums and exact counts without calling a model."""
import hashlib
import gzip
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def rows(name):
    path=ROOT/'data'/name
    stream=path.open(encoding='utf-8') if path.exists() else gzip.open(str(path)+'.gz','rt',encoding='utf-8')
    with stream as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def main():
    manifest=json.loads((ROOT/'data/manifest.json').read_text(encoding='utf-8'))
    for rel,meta in manifest['files'].items():
        path=ROOT/rel
        content=path.read_bytes() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes())
        assert len(content)==meta['bytes'],f'Size mismatch: {rel}'
        assert hashlib.sha256(content).hexdigest()==meta['sha256'],f'Checksum mismatch: {rel}'
    knowledge=list(rows('knowledge.jsonl')); ids={x['id'] for x in knowledge}
    assert len(ids)==len(knowledge), 'Duplicate knowledge IDs'
    assert len({x['content'] for x in knowledge})==len(knowledge), 'Exact duplicate knowledge text'
    assert len(knowledge)==manifest['counts']['knowledge_units']
    for row in knowledge:
        assert row['synthetic'] is True
        assert ('## '+row['id']+' ') in (ROOT/row['source']).read_text(encoding='utf-8'),row['id']
        assert row['content'] in (ROOT/row['source']).read_text(encoding='utf-8'),row['id']
    totals={}
    for filename,expected in [('faq.jsonl','faq_variants'),('simulated_tickets.jsonl','simulated_tickets')]:
        n=0; seen=set()
        for row in rows(filename):
            assert row['source_id'] in ids
            assert row['id'] not in seen
            assert row['synthetic'] is True
            seen.add(row['id']);n+=1
        assert n==manifest['counts'][expected]
        totals[expected]=n
    assert len(list((ROOT/'data/manuals').glob('*.md')))==manifest['counts']['source_manuals']
    assert len(list((ROOT/'data/runbooks').glob('*.md')))==manifest['counts']['source_runbooks']
    sources={x['source'] for x in knowledge}
    assert len(sources)==manifest['counts']['source_documents']
    grams=[set(x['content'][i:i+3] for i in range(len(x['content'])-2)) for x in knowledge]
    near=[]
    for i,a in enumerate(grams):
        for j in range(i):
            b=grams[j]
            if min(len(a),len(b))/max(len(a),len(b))<.8: continue
            intersection=len(a & b); similarity=intersection/(len(a)+len(b)-intersection)
            if similarity>=.8: near.append({'a':knowledge[j]['id'],'b':knowledge[i]['id'],'jaccard':round(similarity,3)})
    report={'status':'passed','knowledge_units':len(ids),**totals,'source_documents':len(sources),
            'checked_files':len(manifest['files']),'content_characters':sum(len(x['content']) for x in knowledge),
            'exact_duplicate_texts':len(knowledge)-len({x['content'] for x in knowledge}),
            'near_duplicate_metric':'character-trigram Jaccard >= 0.8; lexical only, not semantic independence',
            'near_duplicate_pair_count':len(near),'near_duplicate_pairs':near,'units_by_module':dict(Counter(x['category'] for x in knowledge)),
            'units_by_topic':dict(Counter(x['topic'] for x in knowledge)),
            'note':'Four sections of each runbook share one scenario. FAQ and tickets add no independent facts. Checks do not certify business correctness.'}
    (ROOT/'evaluation/data_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='near_duplicate_pairs'},ensure_ascii=False))

if __name__=='__main__':main()
