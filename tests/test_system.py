import json
import http.client
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engine
from app import make_server

class SystemTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.patcher=patch.object(engine,'UPLOADS',Path(self.tmp.name)/'uploads'); self.patcher.start()
        self.server=make_server(0); self.server.engine.key=''
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.patcher.stop(); self.tmp.cleanup()
    def request(self,path,payload=None,raw=None,origin=None):
        body=raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        req=Request(self.base+path,data=body,headers={'Content-Type':'application/json',**({'Origin':origin} if origin else {})})
        try:
            with urlopen(req,timeout=10) as response:return response.status,json.load(response)
        except HTTPError as exc:return exc.code,json.load(exc)
    def test_health_and_grounded_source(self):
        status,data=self.request('/api/health'); self.assertEqual(status,200)
        self.assertEqual(data['knowledge_units'],data['counts']['knowledge_units'])
        _,d=self.request('/api/chat',{'query':'忘记密码怎么重置'}); self.assertEqual(d['sources'][0]['id'],'ACC-13'); self.assertIn('[ACC-13]',d['answer'])
    def test_unknown_question_is_rejected(self):
        _,d=self.request('/api/chat',{'query':'火星到地球有多远？'}); self.assertEqual(d['mode'],'insufficient_evidence'); self.assertEqual(d['sources'],[])
    def test_live_data_and_missing_policy_not_invented(self):
        for q in ['订单AB567891现在什么状态','套餐到底多少钱','提供SOC2证书编号']:
            _,d=self.request('/api/chat',{'query':q}); self.assertEqual(d['mode'],'insufficient_evidence'); self.assertEqual(d['sources'],[])
        for q in ['请直接说明审批流程','银行账户注销导致退款失败，如何处理？']:
            _,d=self.request('/api/chat',{'query':q}); self.assertEqual(d['mode'],'retrieval_only')
    def test_malformed_input_does_not_crash(self):
        for raw in [b'{',b'[]',b'{"query":42}',b'{"query":"hello","history":[3]}']:
            self.assertEqual(self.request('/api/chat',raw=raw)[0],400)
        self.assertEqual(self.request('/api/health')[0],200)
    def test_oversized_request_rejected(self):
        # Rejection occurs from headers before accepting an oversized body.
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10)
        conn.putrequest('POST','/api/chat'); conn.putheader('Content-Length','2000001'); conn.endheaders()
        response=conn.getresponse(); self.assertEqual(response.status,413); response.read(); conn.close()
    def test_cross_origin_write_blocked(self):
        self.assertEqual(self.request('/api/chat',{'query':'hello'},origin='https://example.com')[0],403)
    def test_upload_full_tail_dedup_and_filename_safety(self):
        text='示例内容'*1600+'这就是唯一的尾部知识标记'
        _,d=self.request('/api/upload',{'filename':'../../escape.md','content':text}); self.assertGreater(d['chunks'],5)
        docs=self.server.engine.docs; self.assertTrue(any('唯一的尾部知识标记' in x['content'] for x in docs))
        _,d=self.request('/api/upload',{'filename':'new.md','content':text}); self.assertTrue(d['duplicate'])
        self.assertEqual(len(list(engine.UPLOADS.glob('*.jsonl'))),1)
        _,d=self.request('/api/upload',{'filename':'../../escape.md','content':'另外一篇完全不同的有效知识资料。'}); self.assertEqual(d['chunks'],1)
        self.assertEqual(len(list(engine.UPLOADS.glob('*.jsonl'))),2)
    def test_reject_unsupported_upload(self):
        self.assertEqual(self.request('/api/upload',{'filename':'bad.exe','content':'xx'})[0],400)
    def test_filter_cannot_cross_category(self):
        hits=self.server.engine.search('导入CSV','资产管理'); self.assertTrue(hits); self.assertTrue(all(x['category']=='资产管理' for x in hits))
    def test_invalid_model_citation_falls_back(self):
        e=self.server.engine; e.key='not-a-real-key'; e.base='https://invalid.example/v1'; e.model='fake'
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self): return json.dumps({'choices':[{'message':{'content':'虚构答案[NONEXISTENT-99]'}}]}).encode()
        with patch('engine.urlopen',return_value=Response()):
            answer=e.answer('忘记密码怎么重置')
        self.assertEqual(answer['mode'],'retrieval_fallback'); self.assertNotIn('NONEXISTENT',answer['answer'])

    def test_scan_inverted_ranking_parity(self):
        cases=json.loads((engine.ROOT/'evaluation/cases.json').read_text(encoding='utf-8'))
        for query,_ in cases['positive']:
            for method in ('tfidf','plain_bm25','bm25'):
                with self.subTest(query=query,method=method):
                    a=self.server.engine.search(query,method=method,strategy='scan')
                    b=self.server.engine.search(query,method=method,strategy='inverted')
                    self.assertEqual([(x['id'],x['score']) for x in a],[(x['id'],x['score']) for x in b])

    def test_index_reload_after_upload(self):
        self.server.engine.ingest('incident.md','穹顶蓝灯故障时先断开实验设备电源，再由负责人核对安全状态。')
        hits=self.server.engine.search('穹顶蓝灯故障')
        self.assertTrue(hits[0]['id'].startswith('UP-'))

    def test_short_followup_uses_context(self):
        result=self.server.engine.answer('然后呢',history=[{'role':'user','content':'忘记密码怎么重置'}],force_extract=True)
        self.assertEqual(result['sources'][0]['id'],'ACC-13')

    def test_model_network_error_falls_back(self):
        e=self.server.engine; e.key='fake'; e.model='fake'; e.base='https://invalid.example/v1'
        with patch('engine.urlopen',side_effect=TimeoutError('test')):
            result=e.answer('忘记密码怎么重置')
        self.assertEqual(result['mode'],'retrieval_fallback')
        self.assertEqual(result['error_type'],'TimeoutError')

    def test_supported_positive_queries_not_rejected(self):
        for name in ('cases.json','extended_cases.json'):
            cases=json.loads((engine.ROOT/'evaluation'/name).read_text(encoding='utf-8'))
            for query,_ in cases['positive']:
                with self.subTest(query=query):
                    result=self.server.engine.answer(query,force_extract=True)
                    self.assertEqual(result['mode'],'retrieval_only')

if __name__=='__main__':unittest.main()
