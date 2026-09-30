"""Regressions for contextual refusal, repeated follow-ups and HTTP boundaries."""
import http.client
import json
import unittest
from unittest.mock import patch

import test_system


class ReviewRegressionTests(unittest.TestCase):
    # Reuse the isolated HTTP fixture without running inherited tests twice.
    setUp=test_system.SystemTests.setUp
    tearDown=test_system.SystemTests.tearDown
    request=test_system.SystemTests.request
    def test_contextual_capability_refusals(self):
        for topic in ('订单AB567891现在什么状态', '公司收款账号是多少'):
            history=[{'role':'user','content':topic}]
            for query in ('状态呢', '然后呢', '还有呢'):
                with self.subTest(topic=topic,query=query):
                    status,result=self.request('/api/chat',{'query':query,'history':history})
                    self.assertEqual(status,200)
                    self.assertEqual(result['mode'],'insufficient_evidence')
                    self.assertEqual(result['sources'],[])
                    self.assertEqual(result['answer'],self.server.engine.capability_gap(topic))
                history.extend(({'role':'assistant','content':result['answer']},
                                {'role':'user','content':query}))

    def test_repeated_followups_preserve_topic(self):
        history=[{'role':'user','content':'忘记密码怎么重置'}]
        for query in ('然后呢','还有呢','接着呢'):
            status,result=self.request('/api/chat',{'query':query,'history':history})
            self.assertEqual(status,200)
            self.assertEqual(result['sources'][0]['id'],'ACC-13')
            history.extend(({'role':'assistant','content':result['answer']},
                            {'role':'user','content':query}))

    def test_explicit_topic_change_resets_context(self):
        history=[{'role':'user','content':'订单AB567891现在什么状态'},
                 {'role':'user','content':'然后呢'},
                 {'role':'user','content':'忘记密码怎么重置'}]
        result=self.server.engine.answer('还有呢',history=history,force_extract=True)
        self.assertEqual(result['sources'][0]['id'],'ACC-13')
        result=self.server.engine.answer('忘记密码怎么重置',history=history[:2],force_extract=True)
        self.assertEqual(result['sources'][0]['id'],'ACC-13')

    def test_model_receives_resolved_question(self):
        e=self.server.engine; e.key='fake'; e.model='fake'; e.base='https://invalid.example/v1'
        history=[{'role':'user','content':'忘记密码怎么重置'},
                 {'role':'assistant','content':'原文'},
                 {'role':'user','content':'然后呢'}]
        with patch('engine.urlopen',side_effect=TimeoutError('test')) as call:
            e.answer('还有呢',history=history)
        payload=json.loads(call.call_args.args[0].data)
        self.assertIn('问题：忘记密码怎么重置 然后呢 还有呢',payload['messages'][-1]['content'])

    def host_request(self,method,path,hosts):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10)
        try:
            conn.putrequest(method,path,skip_host=True)
            for host in hosts:
                conn.putheader('Host',host)
            conn.putheader('Content-Length','0'); conn.endheaders()
            response=conn.getresponse(); response.read()
            return response.status
        finally:
            conn.close()

    def test_invalid_hosts_blocked_before_all_routes(self):
        port=self.server.server_port
        hosts=[[],['attacker.example'],[f'attacker.example:{port}'],
               [f'localhost:{port}.attacker.example'],['localhost:1'],
               [f'localhost:{port}',f'localhost:{port}'],
               [f'localhost:{port}',f'attacker.example:{port}']]
        for method,paths in (('GET',('/','/api/health','/api/knowledge','/missing')),
                             ('POST',('/api/chat','/api/upload','/api/tickets','/missing'))):
            for path in paths:
                for values in hosts:
                    with self.subTest(method=method,path=path,hosts=values):
                        self.assertEqual(self.host_request(method,path,values),403)

    def test_local_hosts_accepted(self):
        for hostname in ('localhost','LOCALHOST','127.0.0.1'):
            self.assertEqual(self.host_request('GET','/api/health',[f'{hostname}:{self.server.server_port}']),200)

    def test_invalid_history_returns_400(self):
        for query in ('然后呢','忘记密码怎么重置','hello'):
            for message in ({'role':'user'},{'role':'assistant'}, {'content':'文本'},
                            {'role':'system','content':'文本'}, {'role':'user','content':None},
                            {'role':[],'content':'文本'}):
                with self.subTest(query=query,message=message):
                    self.assertEqual(self.request('/api/chat',{'query':query,'history':[message]})[0],400)
        self.assertEqual(self.request('/api/health')[0],200)

    def test_oversized_history_rejected_before_retrieval_or_model(self):
        e=self.server.engine; e.key='fake'; e.model='fake'; e.base='https://invalid.example/v1'
        for content in ('忘记密码怎么重置'*10000, '忘记密码怎么重置'+'。'*1990+'公司收款账号是多少'):
            with patch.object(e,'search') as search, patch('engine.urlopen') as provider:
                status,_=self.request('/api/chat',{'query':'然后呢',
                    'history':[{'role':'user','content':content}]})
            self.assertEqual(status,400)
            search.assert_not_called(); provider.assert_not_called()

    def test_resolved_context_cap_rejects_without_truncating(self):
        history=[{'role':'user','content':'公司收款账号是多少'},
                 *[{'role':'user','content':'。'*2000} for _ in range(3)]]
        with patch.object(self.server.engine,'search') as search, patch('engine.urlopen') as provider:
            status,_=self.request('/api/chat',{'query':'然后呢','history':history})
        self.assertEqual(status,400)
        search.assert_not_called(); provider.assert_not_called()

    def test_boundary_sized_topic_is_preserved_in_provider_question(self):
        e=self.server.engine; e.key='fake'; e.model='fake'; e.base='https://invalid.example/v1'
        topic='忘记密码怎么重置'.ljust(2000,'。')
        with patch('engine.urlopen',side_effect=TimeoutError('test')) as provider:
            status,_=self.request('/api/chat',{'query':'然后呢',
                'history':[{'role':'user','content':topic}]})
        self.assertEqual(status,200)
        payload=json.loads(provider.call_args.args[0].data)
        self.assertTrue(payload['messages'][-1]['content'].endswith('问题：'+topic+' 然后呢'))
