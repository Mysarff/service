"""Offline contracts for real LangChain prompts and grounded Qwen output."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from cloudcare.llm import QwenGateway

class QwenContracts(unittest.TestCase):
    def setUp(self):
        self.gateway = QwenGateway(SimpleNamespace(llm_configured=False))
        self.sources = [{'id':'ACC-01','title':'账户','content':'首次访问请确认当前组织与角色，不要重复注册账号。'}]

    def test_generation_prompt_and_exact_quote(self):
        def invoke(prompt, values):
            prompt.format_messages(**values)  # JSON braces must be escaped in templates.
            return '{"steps":[{"source_id":"ACC-01","quote":"首次访问请确认当前组织与角色，不要重复注册账号。"}]}'
        with patch.object(self.gateway,'_invoke',side_effect=invoke):
            self.assertIn('[ACC-01]', self.gateway.generate('如何开通？',self.sources,[]))

    def test_hallucinated_quote_fails(self):
        with patch.object(self.gateway,'_invoke',return_value='{"steps":[{"source_id":"ACC-01","quote":"系统自动识别当前组织并为所有用户开通权限。"}]}'):
            with self.assertRaises(ValueError):
                self.gateway.generate('如何开通？',self.sources,[])

    def test_unknown_source_fails(self):
        with patch.object(self.gateway,'_invoke',return_value='{"steps":[{"source_id":"UNKNOWN","quote":"首次访问请确认当前组织与角色，不要重复注册账号。"}]}'):
            with self.assertRaises(ValueError):
                self.gateway.generate('如何开通？',self.sources,[])

    def test_rewrite_preserves_identifier(self):
        with patch.object(self.gateway,'_invoke',return_value='{"query":"订单状态是什么？","keywords":[]}'):
            with self.assertRaises(ValueError):
                self.gateway.rewrite('订单AB123456状态是什么？',[])

    def test_whitespace_only_quote_difference_returns_canonical_source_span(self):
        canonical = '客服核验脱敏日志后，\n  提交值班人员审批。\n\t审批通过后转交人工跟进。'
        sources = [{'id': 'ACC-01', 'title': '工单升级',
                    'content': '其他上下文。\n' + canonical + '\n其他资料。'}]
        merged = ''.join(canonical.split())
        response = json.dumps({'steps': [{'source_id': 'ACC-01', 'quote': merged}]}, ensure_ascii=False)
        with patch.object(self.gateway, '_invoke', return_value=response):
            answer = self.gateway.generate('如何升级工单？', sources, [])
        self.assertIn(canonical, answer)
        self.assertIn('[ACC-01]', answer)
        self.assertNotIn('其他上下文', answer)
        self.assertNotIn('其他资料', answer)

    def test_nonwhitespace_number_or_negation_changes_are_rejected(self):
        canonical = '审批未通过时不能转交，核验金额100元后留存日志。'
        sources = [{'id': 'ACC-01', 'title': '工单升级', 'content': canonical}]
        changed = [canonical.replace('100', '200'), canonical.replace('不能', '能')]
        for quote in changed:
            with self.subTest(quote=quote):
                response = json.dumps({'steps': [{'source_id': 'ACC-01', 'quote': quote}]}, ensure_ascii=False)
                with patch.object(self.gateway, '_invoke', return_value=response):
                    with self.assertRaises(ValueError):
                        self.gateway.generate('如何升级工单？', sources, [])

    def test_whitespace_matching_cannot_return_source_span_over_600_chars(self):
        canonical = '客服核验脱敏日志后，' + ' \n' * 310 + '提交人工审批并保存交接结果。'
        self.assertGreater(len(canonical), 600)
        quote = ''.join(canonical.split())
        self.assertLessEqual(len(quote), 600)
        sources = [{'id': 'ACC-01', 'title': '工单升级', 'content': canonical}]
        response = json.dumps({'steps': [{'source_id': 'ACC-01', 'quote': quote}]}, ensure_ascii=False)
        with patch.object(self.gateway, '_invoke', return_value=response):
            with self.assertRaises(ValueError):
                self.gateway.generate('如何升级工单？', sources, [])

    def test_whitespace_only_quote_is_rejected(self):
        response = json.dumps({'steps': [{'source_id': 'ACC-01', 'quote': ' \n' * 10}]}, ensure_ascii=False)
        with patch.object(self.gateway, '_invoke', return_value=response):
            with self.assertRaises(ValueError):
                self.gateway.generate('如何开通？', self.sources, [])

if __name__=='__main__':
    unittest.main()
