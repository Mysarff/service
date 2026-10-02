import asyncio
import math
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from cloudcare.evaluation import score_record,summarize_scores

class RagasContracts(unittest.TestCase):
    def test_actual_fields_and_failed_scores_stay_null(self):
        metrics={name:SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=.8)))
                 for name in ('faithfulness','answer_relevancy','context_precision','context_recall')}
        metrics['faithfulness'].ascore.side_effect=ValueError('parse failure')
        metrics['context_recall'].ascore.return_value=SimpleNamespace(value=float('nan'))
        record={'user_input':'q','response':'a','retrieved_contexts':['actual source'],'reference':'gold'}
        result=asyncio.run(score_record(metrics,record))
        self.assertIsNone(result['faithfulness']['value'])
        self.assertIsNone(result['context_recall']['value'])
        self.assertEqual(result['answer_relevancy']['value'],.8)
        self.assertEqual(metrics['context_precision'].ascore.call_args.kwargs['retrieved_contexts'],['actual source'])
        summary=summarize_scores([{'metrics':result}])
        self.assertEqual(summary['faithfulness']['errors'],1)
        self.assertIsNone(summary['faithfulness']['mean'])
