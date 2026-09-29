"""Fixed-corpus, fixed-question checks for the seven reviewed failures."""
import gzip
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.retrieval_audit import audit
from engine import Engine


class RetrievalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit()
        with gzip.open(ROOT / 'evaluation/benchmark_results.json.gz', 'rt', encoding='utf-8') as file:
            cls.baseline = json.load(file)

    def test_only_one_source_label_changed_and_questions_unchanged(self):
        label_changes = []
        for group in ('cases.json', 'extended_cases.json'):
            cases = json.loads((ROOT / 'evaluation' / group).read_text(encoding='utf-8'))
            old = self.baseline['retrieval'][group]['bm25']['observations']
            self.assertEqual([q for q, _ in cases['positive']], [row['query'] for row in old])
            for (query, expected), row in zip(cases['positive'], old):
                if expected != row['expected']:
                    label_changes.append((query, row['expected'], expected))
        self.assertEqual(label_changes, [
            ('知识库搜索还在引用已经过期的政策，要怎么修正？', ['KB-RB01-2'], ['KB-RB01-2', 'KB-13'])
        ])
        docs = {row['id']: row for line in (ROOT / 'data/knowledge.jsonl').read_text(encoding='utf-8').splitlines()
                if (row := json.loads(line))}
        self.assertIn('撤回失效文档后重新构建检索索引', docs['KB-13']['content'])
        self.assertIn('撤回失效版本的可检索状态并重建相关索引', docs['KB-RB01-2']['content'])
        self.assertFalse(self.report['corpus_changed'])

    def test_original_seven_and_current_five_are_accounted_for(self):
        self.assertEqual({row['query']: row['ids'][0] for row in self.report['original_bm25_failures']}, {
            '订单重复创建了要重新扣钱吗？': 'ORD-02',
            '这次续费扣款失败该查哪里？': 'SUB-RB01-1',
            '运单轨迹好久没有更新了': 'SHIP-RB01-2',
            '报表统计结果和之前对不上': 'REPORT-RB01-2',
            '聊天转给另一坐席以后历史消息不见了，怎么查？': 'CHAT-RB05-1',
            '知识库搜索还在引用已经过期的政策，要怎么修正？': 'KB-13',
            '项目任务A等B、B又等A，负责人应怎么处理？': 'PRJ-RB04-2',
        })
        self.assertEqual(len(self.report['current_bm25_failures']), 5)
        self.assertEqual([x['id'] for x in Engine().search('运单轨迹好久没有更新了')][:2],
                         ['SHIP-09', 'SHIP-RB01-2'])

    def test_deterministic_metrics_and_scan_parity(self):
        self.assertEqual(self.report['methods']['bm25'], {
            'historical': {'top1': 57, 'hit5': 64, 'mrr5': 0.9414, 'n': 64},
            'labels_only': {'top1': 58, 'hit5': 64, 'mrr5': 0.9492, 'n': 64},
            'current': {'top1': 59, 'hit5': 64, 'mrr5': 0.957, 'n': 64},
        })
        search = Engine()
        for group in ('cases.json', 'extended_cases.json'):
            cases = json.loads((ROOT / 'evaluation' / group).read_text(encoding='utf-8'))
            for query, _ in cases['positive']:
                with self.subTest(query=query):
                    scan = search.search(query, strategy='scan')
                    inverted = search.search(query, strategy='inverted')
                    self.assertEqual([(row['id'], row['score']) for row in scan],
                                     [(row['id'], row['score']) for row in inverted])


if __name__ == '__main__':
    unittest.main()
