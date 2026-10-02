"""FAQ BM25 math and threshold routing regression contracts.

The lexical algorithm is real. Pipeline backends are injected fixtures, so
these checks establish decisions/cache isolation, not neural/model quality.
"""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
from unittest.mock import Mock

from cloudcare.faq import FAQBM25Index
from cloudcare.settings import Settings
import test_pipeline_contracts as fixtures


QUERY = "账号登录密码怎么重置"


class FAQBM25AlgorithmTests(unittest.TestCase):
    def test_raw_score_matches_bm25_formula(self):
        rows = [dict(id="A", question="alpha alpha beta", answer="reset", source_id="S"),
                dict(id="B", question="beta gamma", answer="other", source_id="T")]
        matched = FAQBM25Index(rows).search("alpha")
        idf = math.log(1 + (2 - 1 + .5) / (1 + .5))
        length_norm = 1.2 * (1 - .75 + .75 * 3 / 2.5)
        expected = idf * 2 * 2.2 / (2 + length_norm)
        self.assertEqual(matched["id"], "A")
        self.assertEqual(matched["method"], "bm25")
        self.assertAlmostEqual(matched["raw_score"], expected)
        self.assertAlmostEqual(matched["score"], math.exp(expected) / (math.exp(expected) + 1))

    def test_answer_body_is_not_indexed(self):
        index = FAQBM25Index([dict(id="A", question="alpha", answer="secretanswer", source_id="S")])
        self.assertIsNone(index.search("secretanswer"))

    def test_unicode_nfkc_query_normalization(self):
        index = FAQBM25Index([dict(id="A", question="ABC account", answer="reset", source_id="S")])
        self.assertEqual(index.search("ＡＢＣ")["id"], "A")

    def test_category_filter_excludes_other_faq(self):
        index = FAQBM25Index([dict(id="A", question="alpha", category="account", answer="reset", source_id="S"),
                              dict(id="B", question="alpha", category="billing", answer="invoice", source_id="T")])
        self.assertEqual(index.search("alpha", category="billing")["id"], "B")
        self.assertEqual(index.search("alpha", category="billing")["candidate_count"], 1)
        self.assertEqual(index.search("alpha", category="billing")["eligible_count"], 1)
        self.assertEqual(index.search("alpha", category="billing")["score"], 1)
        self.assertIsNone(index.search("alpha", category="other"))

    def test_ties_use_stable_faq_id(self):
        index = FAQBM25Index([dict(id="B", question="alpha", answer="second", source_id="T"),
                              dict(id="A", question="alpha", answer="first", source_id="S")])
        self.assertEqual(index.search("alpha")["id"], "A")
        self.assertAlmostEqual(index.search("alpha")["score"], .5)

    def test_exact_question_can_have_score_below_configured_threshold(self):
        index = FAQBM25Index([dict(id="A", question="alpha", answer="first", source_id="S"),
                              dict(id="B", question="beta", answer="second", source_id="T")])
        matched = index.search("alpha")
        self.assertGreater(matched["score"], 0)
        self.assertLess(matched["score"], .85)

    def test_softmax_denominator_includes_all_unmatched_zero_scores(self):
        rows = [dict(id="A", question="alpha", answer="first", source_id="S")]
        rows.extend(dict(id=f"B{index:03d}", question="beta", answer="other", source_id="T")
                    for index in range(99))
        matched = FAQBM25Index(rows).search("alpha")
        expected = math.exp(matched["raw_score"]) / (math.exp(matched["raw_score"]) + 99)
        self.assertAlmostEqual(matched["score"], expected)
        self.assertLess(matched["score"], 1)
        self.assertEqual(matched["candidate_count"], 1)

    def test_single_matching_candidate_is_not_normalized_to_one(self):
        matched = FAQBM25Index([dict(id="A", question="alpha", answer="reset", source_id="S"),
                               dict(id="B", question="beta", answer="bill", source_id="T")]).search("alpha")
        self.assertEqual(matched["candidate_count"], 1)
        self.assertAlmostEqual(matched["score"], 2 / 3)

    def test_equal_candidate_distribution_sums_to_one(self):
        rows = [dict(id=name, question="alpha", answer="reset", source_id="S") for name in "DCBA"]
        matched = FAQBM25Index(rows).search("alpha")
        self.assertEqual(matched["id"], "A")
        self.assertEqual(matched["candidate_count"], 4)
        self.assertAlmostEqual(matched["score"], .25)
        self.assertAlmostEqual(matched["score"] * len(rows), 1)

    def test_stable_softmax_accepts_large_raw_scores(self):
        query = " ".join(f"word{index}" for index in range(2200))
        rows = [dict(id="A", question=query, answer="reset", source_id="S"),
                dict(id="B", question="unmatched", answer="other", source_id="T")]
        matched = FAQBM25Index(rows).search(query)
        self.assertGreater(matched["raw_score"], 710)
        self.assertTrue(math.isfinite(matched["score"]))
        self.assertAlmostEqual(matched["score"], 1)


class FAQThresholdRoutingTests(unittest.TestCase):
    # Reuse only fixture setup, without inheriting the original test methods.
    def setUp(self):
        fixtures.PipelineContractTests.setUp(self)
        # The gate consumes Softmax-normalized BM25, as in the original
        # education FAQ layer; raw BM25 is preserved separately in tracing.
        self.settings = replace(self.settings, faq_bm25_threshold=.85)
        self.pipeline.settings = self.settings

    def faq(self, score=.85, **changes):
        self.redis.faq = {"id": "FAQ-A", "question": QUERY, "source_id": "A",
                          "answer": "通过工作邮箱重置密码", "method": "bm25",
                          "score": score, "raw_score": 12.0, "normalization": "softmax_all_faq",
                          "matches": 5, "candidate_count": 1, **changes}

    def test_conservative_trial_default_is_055(self):
        self.assertEqual(Settings().faq_bm25_threshold, .55)

    def test_055_trial_threshold_inclusive_boundary_and_fallback(self):
        self.pipeline.settings = replace(self.settings, faq_bm25_threshold=.55)
        for score, mode, accepted in [(.55, "faq", True), (.549999, "retrieval_only", False)]:
            with self.subTest(score=score):
                self.redis.answers.clear()
                self.vector.search_calls.clear()
                self.faq(score=score)
                result = self.pipeline.answer(QUERY, force_extract=True)
                self.assertEqual(result["mode"], mode)
                self.assertEqual(result["trace"]["faq"]["threshold"], .55)
                self.assertEqual(result["trace"]["faq"]["accepted"], accepted)
                self.assertEqual(result["trace"]["faq"]["reason"], "accepted" if accepted else "below_threshold")
                self.assertEqual(bool(self.vector.search_calls), not accepted)

    def test_score_equal_threshold_returns_faq(self):
        self.faq(score=.85)
        result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(result["mode"], "faq")
        self.assertTrue(result["trace"]["faq"]["accepted"])
        self.assertEqual(result["trace"]["faq"]["reason"], "accepted")
        self.assertEqual(result["trace"]["faq"]["threshold"], .85)
        self.assertEqual(result["trace"]["faq"]["score"], .85)
        self.assertEqual(result["trace"]["faq"]["raw_score"], 12.0)
        self.assertEqual(result["trace"]["faq"]["normalization"], "softmax_all_faq")
        self.assertEqual(len(self.router.calls), 1)
        self.assertEqual(self.vector.search_calls, [])

    def test_score_below_threshold_enters_rag_without_exact_bypass(self):
        self.faq(score=.849999)
        result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(result["mode"], "retrieval_only")
        self.assertFalse(result["trace"]["faq"]["accepted"])
        self.assertEqual(result["trace"]["faq"]["reason"], "below_threshold")
        self.assertEqual(result["trace"]["faq"]["question"], QUERY)
        self.assertGreater(len(self.vector.search_calls), 0)

    def test_no_candidate_enters_rag(self):
        self.redis.faq = None
        result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(result["mode"], "retrieval_only")
        self.assertEqual(result["trace"]["faq"]["reason"], "no_candidate")
        self.assertIsNone(result["trace"]["faq"]["score"])
        self.assertEqual(result["trace"]["faq"]["candidate_count"], 0)

    def test_missing_source_enters_rag_even_above_threshold(self):
        self.faq(score=.99, source_id="missing-source")
        result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(result["mode"], "retrieval_only")
        self.assertEqual(result["trace"]["faq"]["reason"], "invalid_source")
        self.assertGreater(len(self.vector.search_calls), 0)

    def test_private_faq_source_enters_rag(self):
        private = deepcopy(self.sql.chunks["A"])
        private.update(id="PRIVATE", visibility="internal")
        self.sql.chunks["PRIVATE"] = private
        self.faq(score=.99, source_id="PRIVATE")
        result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(result["mode"], "retrieval_only")
        self.assertEqual(result["trace"]["faq"]["reason"], "invalid_source")

    def test_null_answer_enters_rag(self):
        self.faq(score=.99, answer=None)
        result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(result["mode"], "retrieval_only")
        self.assertEqual(result["trace"]["faq"]["reason"], "empty_answer")

    def test_threshold_change_does_not_reuse_cached_faq(self):
        self.faq(score=.85)
        first = self.pipeline.answer(QUERY, force_extract=True)
        self.assertEqual(first["mode"], "faq")
        self.pipeline.settings = replace(self.settings, faq_bm25_threshold=.86)
        second = self.pipeline.answer(QUERY, force_extract=True)
        self.assertFalse(second["trace"]["cache_hit"])
        self.assertEqual(second["mode"], "retrieval_only")
        self.assertEqual(second["trace"]["faq"]["threshold"], .86)

    def test_faq_version_change_invalidates_cache(self):
        self.faq(score=.85)
        finder = Mock(wraps=self.redis.find_faq)
        self.redis.find_faq = finder
        self.redis.faq_index_version = Mock(return_value="fixture-v1")
        first = self.pipeline.answer(QUERY, force_extract=True)
        cached = self.pipeline.answer(QUERY, force_extract=True)
        self.assertFalse(first["trace"]["cache_hit"])
        self.assertTrue(cached["trace"]["cache_hit"])
        self.assertEqual(finder.call_count, 1)
        self.redis.faq_index_version.return_value = "fixture-v2"
        next_result = self.pipeline.answer(QUERY, force_extract=True)
        self.assertFalse(next_result["trace"]["cache_hit"])
        self.assertEqual(finder.call_count, 2)
        self.assertEqual(next_result["trace"]["faq"]["index_version"], "fixture-v2")
        self.assertEqual(finder.call_args.kwargs["index_version"], "fixture-v2")


if __name__ == "__main__":
    unittest.main()
