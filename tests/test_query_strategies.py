import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from cloudcare.query import denoise_query, search_routes, validate_rewrite, explicit_subqueries
from cloudcare.llm import QwenGateway, evidence_contexts
from cloudcare.settings import Settings
from cloudcare.retrieval import MilvusStore
from cloudcare.faq import FAQBM25Index
import test_pipeline_contracts as fixtures

class QueryStrategiesTests(unittest.TestCase):
    def test_explicit_compound_question_is_split_without_rewriting_constraints(self):
        parts=explicit_subqueries('接口返回429应该怎么处理，密钥怀疑泄露又该怎么轮换？')
        self.assertEqual(len(parts),2)
        self.assertIn('429',parts[0])
        self.assertEqual(explicit_subqueries('API 429，不能换密钥吧？'),[])
        self.assertEqual(explicit_subqueries('我忘记密码了，怎么处理？'),[])

    def test_empty_model_subqueries_have_real_fallback_routes(self):
        query='忘记密码要怎么重置，换了手机后验证器又该怎么迁移？'
        gateway=QwenGateway(Settings())
        gateway._invoke=Mock(return_value=json.dumps({'query':query,'subqueries':[], 'keywords':[], 'hyde_document':''}))
        plan=gateway.plan(query,[])
        self.assertEqual(plan['subquery_method'],'explicit_clause_fallback')
        self.assertEqual(sum(r['kind']=='subquery' for r in search_routes(query,plan)),2)
    def test_query_relative_bm25_exact_and_missing_terms(self):
        index=FAQBM25Index([{'id':'A','source_id':'S','answer':'a','question':'alpha beta'},
                           {'id':'B','source_id':'T','answer':'b','question':'gamma delta'}])
        exact=index.search('alpha beta')
        self.assertEqual(exact['normalization'],'bm25_query_reference')
        self.assertAlmostEqual(exact['score'],1)
        partial=index.search('alpha unseen terms')
        self.assertLess(partial['score'],1)
        self.assertGreater(partial['reference_score'],partial['raw_score'])

    def test_valid_strategies_survive_invalid_rewrite(self):
        gateway=QwenGateway(Settings())
        gateway._invoke=Mock(return_value=json.dumps({'query':'可以更换API密钥','subqueries':[],
            'keywords':['限流'],'hyde_document':'需要核查调用频率和身份限制'}))
        result=gateway.plan('API 429不能换密钥',[])
        self.assertEqual(result['query'],'API 429不能换密钥')
        self.assertEqual(result['keywords'],['限流'])
        self.assertIn('rewrite_rejected_original_preserved',result['validation_warnings'])

    def test_parent_section_alias_requires_actual_section_quote(self):
        gateway=QwenGateway(Settings())
        source={'id':'ACC-13','title':'账号','content':'重置密码正文足够长',
                'parent_content':'## ACC-13 重置密码\n重置密码正文足够长。\n## ACC-14 换手机\n更换手机前先添加新验证器并验证成功。'}
        gateway._invoke=Mock(return_value=json.dumps({'steps':[{'source_id':'[ACC-14]',
            'quote':'更换手机前先添加新验证器并验证成功。'}]}))
        self.assertIn('[ACC-13]',gateway.generate('怎么换手机',[source],[]))
        gateway._invoke.return_value=json.dumps({'steps':[{'source_id':'ACC-14','quote':'重置密码正文足够长。'}]})
        with self.assertRaises(ValueError):gateway.generate('怎么换手机',[source],[])

    def test_denoise_preserves_business_constraints(self):
        self.assertEqual(denoise_query('您好， ＡＰＩ\u200b ４２９不能换密钥！！！'), 'API 429不能换密钥!')

    def test_constraints_reject_missing_number_and_negation(self):
        for value in ['API不能换密钥','API 429可以换密钥']:
            with self.assertRaises(ValueError): validate_rewrite('API 429不能换密钥',value)

    def test_query_budget_and_weights(self):
        plan={'query':'rewrite','subqueries':['one','two','three','four'], 'keywords':['k'], 'hyde_document':'hypothesis'}
        routes=search_routes('original',plan)
        self.assertEqual(len(routes),7)
        self.assertAlmostEqual(sum(x['weight'] for x in routes if x['kind']=='subquery'),1)
        self.assertEqual(routes[-1]['kind'],'hyde')
        self.assertFalse(any(x['kind']=='hyde' for x in search_routes('original',plan,hyde_enabled=False)))
        self.assertEqual(len(search_routes('original',plan,max_queries=2)),2)

    def test_plan_reads_history_and_rejects_invented_numeric_policy(self):
        gateway=QwenGateway(Settings())
        response={'query':'账号密码如何重置','subqueries':[], 'keywords':['重置密码'], 'hyde_document':'密码由企业管理员核验'}
        gateway._invoke=Mock(return_value=json.dumps(response))
        result=gateway.plan('账号密码如何重置',[{'role':'user','content':'我用企业SSO'}])
        self.assertEqual(result['history_messages'],1)
        self.assertIn('企业SSO',gateway._invoke.call_args.args[1]['history'])
        response['hyde_document']='链接有效期24小时'
        gateway._invoke.return_value=json.dumps(response)
        result=gateway.plan('账号密码如何重置',[])
        self.assertEqual(result['hyde_document'],'')
        self.assertIn('hyde_new_number_rejected',result['validation_warnings'])

    def test_evidence_budget_is_actual_prompt_budget(self):
        contexts=evidence_contexts([{'id':'A','title':'t','content':'x'*40}, {'id':'B','title':'t','content':'y'*40}],budget=70)
        self.assertLessEqual(len('\n\n'.join(r['rendered'] for r in contexts)),70)
        self.assertLess(len(contexts[-1]['text']),40)

    def test_hyde_dense_filter_is_required_and_enforced(self):
        encoder=SimpleNamespace(encode_documents=Mock(return_value=[SimpleNamespace(dense=[.1]*1024)]))
        client=SimpleNamespace(search=Mock(return_value=[[{'entity':{'id':'A','text':'source','tenant_id':'demo','visibility':'public','category':'account'},'distance':.8}]]))
        vector=MilvusStore(Settings(),encoder,client=client);vector._ready=True
        self.assertEqual(vector.dense_search('hypothesis',category='account')[0]['content'],'source')
        self.assertEqual(client.search.call_args.kwargs['anns_field'],'dense_vector')
        self.assertIn('tenant_id',client.search.call_args.kwargs['filter'])
        with self.assertRaises(RuntimeError): vector.dense_search('hypothesis',category='other')

class QueryPipelineTests(unittest.TestCase):
    def setUp(self): fixtures.PipelineContractTests.setUp(self)

    def test_plan_routes_run_and_hypothesis_is_not_evidence(self):
        self.pipeline.start()
        self.vector.dense_search=Mock(return_value=self.vector.hits)
        hits,trace=self.pipeline.retrieve('账号密码',query_plan={'query':'账号重置密码','subqueries':['密码找回'],
            'keywords':['身份核验'], 'hyde_document':'hypothetical secret'})
        self.vector.dense_search.assert_called_once()
        self.assertIn('keywords',[x['kind'] for x in trace['query_routes']])
        self.assertNotIn('hypothetical secret',str(hits))
        self.assertLessEqual(trace['merged_candidates'],20)

    def test_actual_planner_is_invoked_and_failure_falls_back(self):
        self.pipeline.settings=replace(self.settings,llm_api_key='fixture-key',query_expansion_enabled=True)
        self.gateway.plan=Mock(side_effect=ValueError('invalid plan'))
        result=self.pipeline.answer('账号密码如何处理')
        self.gateway.plan.assert_called_once()
        self.assertEqual(result['trace']['rewrite']['error_type'],'ValueError')
        self.assertEqual(result['trace']['retrieval']['queries'],['账号密码如何处理'])
