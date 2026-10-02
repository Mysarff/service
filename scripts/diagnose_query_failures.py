"""Retain source-only model outputs for failed local synthetic eval records."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cloudcare.llm import QwenGateway
from cloudcare.settings import Settings
from cloudcare.storage import MySQLStore
settings=Settings.load();gateway=QwenGateway(settings);sql=MySQLStore(settings)
records=json.loads((ROOT/'evaluation/query_upgrade_20261002/query_results.json').read_text(encoding='utf-8'))['records']
results=[];invoke=gateway._invoke
def capture(prompt,values):
    result=invoke(prompt,values); current['raw_response']=result;return result
gateway._invoke=capture
try:
    for record in records:
        if record['variant']!='multi_strategy':continue
        for stage in ('plan','generation'):
            if not record['plan_error' if stage=='plan' else 'generation_error']:continue
            current={'id':record['id'],'stage':stage}
            try:
                if stage=='plan':gateway.plan(record['user_input'],record['history'])
                else:
                    sources=[sql.get_chunk(key) for key in record['retrieved_ids']]
                    for source in sources:source['parent_content']=sql.get_parent(source['parent_id'])['content']
                    gateway.generate(record['user_input'],sources,record['history'])
                current['status']='passed_on_retry'
            except Exception as exc:
                current['status']='failed';current['error_type']=type(exc).__name__
                if isinstance(exc,ValueError):current['reason']=str(exc)
            results.append(current)
            (ROOT/'evaluation/query_upgrade_20261002/failure_diagnostics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in current.items() if k!='raw_response'},ensure_ascii=False),flush=True)
finally:gateway.close();sql.close()
