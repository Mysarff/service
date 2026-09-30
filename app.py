"""Local CloudCare service. Bind to loopback; do not expose directly to the Internet."""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from engine import Engine, ROOT, DATA

class Handler(BaseHTTPRequestHandler):
    server_version='CloudCare/1.0'
    def parse_request(self):
        if not super().parse_request():
            return False
        # Validate every route before dispatch. Origin alone cannot stop reads
        # through an attacker-controlled hostname that resolves to loopback.
        hosts=self.headers.get_all('Host',[])
        allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
        if self.server.server_port == 80:
            allowed.update(('127.0.0.1','localhost'))
        if len(hosts) != 1 or hosts[0].lower() not in allowed:
            self.close_connection=True
            self.send_json({'error':'主机不受支持'},403)
            return False
        return True

    def send_json(self, value, code=200):
        raw=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Content-Length',str(len(raw)))
        self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/health':
            manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
            self.send_json(dict(status='ok',knowledge_units=len(self.server.engine.docs),counts=manifest['counts'],
                                categories=sorted({d['category'] for d in self.server.engine.docs}),
                                model_enabled=self.server.engine.model_ready,model=self.server.engine.model if self.server.engine.model_ready else '',retriever='BM25 + title/category weighting')); return
        if path=='/api/knowledge':
            self.send_json({'items':[{k:d.get(k) for k in ('id','title','category','source','content','synthetic')} for d in self.server.engine.docs]}); return
        if path in ('/','/index.html'):
            raw=(ROOT/'static/index.html').read_bytes(); self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(raw)))
            self.end_headers(); self.wfile.write(raw); return
        self.send_json({'error':'not_found'},404)

    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            self.send_json({'error':'来源不受支持'},403); return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=2_000_000: self.send_json({'error':'请求体为空或超过2 MB'},413); return
            payload=json.loads(self.rfile.read(size))
            if not isinstance(payload,dict): raise ValueError('JSON必须为对象')
            if self.path=='/api/chat':
                query=payload.get('query',''); history=payload.get('history',[])
                if not isinstance(query,str) or not query.strip() or len(query)>2000: raise ValueError('问题须为1至2000个字符')
                if not isinstance(history,list) or len(history)>8 or any(not isinstance(m,dict) or m.get('role') not in ('user','assistant') or not isinstance(m.get('content'),str) for m in history): raise ValueError('历史对话格式无效')
                category=payload.get('category','')
                if not isinstance(category,str): raise ValueError('模块格式无效')
                start=time.perf_counter(); result=self.server.engine.answer(query.strip(),category,history)
                result['elapsed_ms']=round((time.perf_counter()-start)*1000)
                self.send_json(result); return
            if self.path=='/api/upload':
                self.send_json(self.server.engine.ingest(payload.get('filename',''),payload.get('content',''))); return
            if self.path=='/api/tickets':
                summary=payload.get('summary','')
                if not isinstance(summary,str) or not 3<=len(summary.strip())<=2000: raise ValueError('工单摘要须为3至2000个字符')
                ident='LOCAL-'+uuid.uuid4().hex[:10].upper()
                path=ROOT/'runtime/tickets.jsonl'; path.parent.mkdir(parents=True,exist_ok=True)
                row={'id':ident,'summary':summary.strip(),'status':'待处理','created_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'local_only':True}
                with self.server.write_lock:
                    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                self.send_json({'ticket':row,'message':'已创建本地演示工单，尚未接入真人客服。'}); return
            self.send_json({'error':'not_found'},404)
        except (ValueError,UnicodeDecodeError): self.send_json({'error':'输入无效，请检查JSON格式、字段类型及长度'},400)
        except Exception: self.send_json({'error':'服务处理失败，请查看本地服务状态'},500)

    def log_message(self,fmt,*args): print('[CloudCare] '+fmt%args,flush=True)

def make_server(port=8088,model_config=None):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.engine=Engine(model_config); server.write_lock=threading.Lock()
    return server

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8088); parser.add_argument('--model-config')
    parser.add_argument('--offline',action='store_true',help='Disable external model calls even when environment credentials exist')
    args=parser.parse_args(); server=make_server(args.port,args.model_config)
    if args.offline: server.engine.key=''
    print(f'CloudCare http://127.0.0.1:{server.server_port} | units={len(server.engine.docs)} | model_enabled={server.engine.model_ready}',flush=True)
    server.serve_forever()
