"""Small opt-in end-to-end smoke run. May call the model configured on the running server."""
import json
import time
from pathlib import Path
from urllib.request import Request,urlopen

def main():
    cases=['我忘记登录密码了，怎么重置？','取消自动续费后会自动退款吗？','帮我查询订单ZX987654现在到哪了']
    results=[]
    for query in cases:
        request=Request('http://127.0.0.1:8088/api/chat',data=json.dumps({'query':query}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=65) as response: answer=json.load(response)
        results.append({'query':query,**answer})
        print(json.dumps({'query':query,'mode':answer['mode'],'sources':[x['id'] for x in answer['sources']],'elapsed_ms':answer['elapsed_ms']},ensure_ascii=False),flush=True)
    (Path(__file__).parent/'live_smoke_results.json').write_text(json.dumps({'date':time.strftime('%Y-%m-%d'),'cases':results},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
