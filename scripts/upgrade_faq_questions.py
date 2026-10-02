"""Generate source-only FAQ paraphrases, resumably. Never reads evaluation files.

Keeps IDs, answers and knowledge count unchanged; replaces two template variants
per source. Output is a candidate artifact until --apply is explicitly supplied.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloudcare.settings import Settings
from cloudcare.llm import QwenGateway
from langchain_core.prompts import ChatPromptTemplate

OUT = ROOT / 'evaluation/query_upgrade_20261002'

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'source_only_faq'; cache.mkdir(exist_ok=True)
    settings = Settings.load()
    knowledge = [json.loads(x) for x in (ROOT/'data/knowledge.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = {}
    for row in knowledge: groups.setdefault(row['source'], []).append(row)
    prompt = ChatPromptTemplate.from_messages([
        ('system', '为合成企业客服资料编写自然的FAQ问句。每个片段生成2条问法，一条自然完整问句、一条口语问句。'
         '问句必须由该片段本身回答，紧扣标题的主要业务问题，不问通用的记录资源ID或身份核验。'
         '不得添加资料未给出的价格、期限、编号，不写前缀“请说明模块的阶段”。'
         '不要复制整段内容，不输出答案。只输出JSON对象，键为提供的片段ID，值为2条不同问句的数组。'),
        ('human', '{sources}')])
    def generate(item):
        source, rows = item
        key = hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        dest = cache / (Path(source).stem+'-'+key[:10]+'.json')
        if dest.is_file(): return json.loads(dest.read_text(encoding='utf-8'))
        gateway = QwenGateway(settings)
        try:
            raw = gateway._invoke(prompt, {'sources':json.dumps([{k:r[k] for k in ('id','title','content')} for r in rows],ensure_ascii=False)})
            result = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()))
            if set(result) != {r['id'] for r in rows}: raise ValueError('Missing source IDs')
            for row in rows:
                values = result[row['id']]
                if not isinstance(values,list) or len(values)!=2 or any(not isinstance(x,str) or not 5<=len(x)<=100 for x in values):
                    raise ValueError('Invalid FAQ question')
                allowed = set(re.findall(r'\d+',row['title']+row['content']))
                if any(not set(re.findall(r'\d+',x)).issubset(allowed) for x in values): raise ValueError('Invented number')
            record = {'source':source,'source_sha256':key,'model':settings.llm_model,'questions':result}
            dest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            return record
        finally: gateway.close()
    records=[]; errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(generate,item):item[0] for item in groups.items()}
        for future in as_completed(futures):
            try: records.append(future.result())
            except Exception as exc: errors.append({'source':futures[future],'error_type':type(exc).__name__})
            print(json.dumps({'finished':len(records)+len(errors),'total':len(groups),'errors':len(errors)}),flush=True)
    questions={key:value for record in records for key,value in record['questions'].items()}
    old=gzip.decompress((ROOT/'data/faq.jsonl.gz').read_bytes()).decode('utf-8')
    rows=[json.loads(x) for x in old.splitlines()]
    changed=0; seen=set()
    for row in rows:
        variant=int(row['id'].rsplit('-',1)[1]); source=row['source_id']
        if variant in (1,2) and source in questions:
            question=questions[source][variant-1].strip()
            canonical=re.sub(r'[^\w\u4e00-\u9fff]+','',question).lower()
            # Cross-source duplicates are ambiguous; retain the original template.
            if canonical not in seen:
                row['question']=question; row['question_origin']='qwen_source_only_20261002'; changed+=1
        seen.add(re.sub(r'[^\w\u4e00-\u9fff]+','',row['question']).lower())
    text=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    (OUT/'faq_candidate.jsonl.gz').write_bytes(gzip.compress(text.encode('utf-8'),mtime=0))
    report={'documents':len(groups),'successful_documents':len(records),'changed_questions':changed,'faq_rows':len(rows),
            'knowledge_units':len(knowledge),'errors':errors,'generation_input':'source title and content only; no evaluation questions',
            'corpus_sha256':hashlib.sha256((ROOT/'data/knowledge.jsonl').read_bytes()).hexdigest()}
    (OUT/'faq_generation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.apply:
        if errors: raise RuntimeError('Retry failed source batches before applying')
        (ROOT/'data/faq.jsonl').write_text(text,encoding='utf-8')
        (ROOT/'data/faq.jsonl.gz').write_bytes(gzip.compress(text.encode('utf-8'),mtime=0))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
