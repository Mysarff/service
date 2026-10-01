"""Contract and access-boundary tests; these do not claim model quality."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudcare.neural import BGEM3Encoder, BGEReranker, EncodedVectors, SupportBertRouter, _HFSequenceReranker, _file_sha256
from cloudcare.retrieval import MilvusStore, source_filter


def settings():
    return SimpleNamespace(embedding_model_path="unused", reranker_model_path="unused", device="cpu",
                           embedding_batch_size=4, embedding_max_length=128, reranker_max_length=128,
                           rerank_k=2, milvus_collection="cloudcare_support_test", dense_weight=.65,
                           sparse_weight=.35)


class FakeEncoderModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return {"dense_vecs": [[.1] * 1024 for _ in texts],
                "lexical_weights": [{"10": .4, "11": 0.} for _ in texts]}


class FakeReranker:
    def compute_score(self, pairs, **kwargs):
        return [.1, .8, .4][:len(pairs)]


class FakeSearchClient:
    def hybrid_search(self, **kwargs):
        self.request = kwargs
        return [[{"entity": {"id": "A", "text": "content", "tenant_id": "other", "visibility": "public"},
                  "distance": .7}]]


class NeuralContractTests(unittest.TestCase):
    def test_checkpoint_hash_streams_without_whole_file_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            content = b"checkpoint-contract" * 100000
            path.write_bytes(content)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("whole checkpoint allocation")):
                self.assertEqual(_file_sha256(path), hashlib.sha256(content).hexdigest())

    def test_support_router_rejects_education_report_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "training_report.json").write_text(json.dumps({
                "task": "education_routing", "labels": ["general", "professional"]}), encoding="utf-8")
            config = settings()
            config.bert_model_path = path
            with self.assertRaisesRegex(ValueError, "education labels"):
                SupportBertRouter(config).predict("账号密码如何重置")

    def test_support_router_rejects_tampered_weights_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "training_report.json").write_text(json.dumps({
                "task": "customer_support_routing",
                "labels": ["support_knowledge", "handoff", "out_of_scope"],
                "checkpoint_sha256": "incorrect"}), encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"tampered")
            config = settings()
            config.bert_model_path = path
            with self.assertRaisesRegex(ValueError, "weights do not match"):
                SupportBertRouter(config).predict("账号密码如何重置")

    def test_duplicate_queries_and_bounded_cache(self):
        model = FakeEncoderModel()
        encoder = BGEM3Encoder(settings(), model=model, query_cache_size=1)
        encoded = encoder.encode_queries(["同一问题", "同一问题"])
        self.assertEqual(len(encoded), 2)
        self.assertEqual(model.calls, [["同一问题"]])
        self.assertEqual(encoded[0].sparse, {10: .4})
        encoder.encode_queries(["同一问题"])
        self.assertEqual(encoder.query_cache_hits, 1)
        encoder.encode_queries(["第二个问题"])
        encoder.encode_queries(["同一问题"])
        self.assertEqual(len(model.calls), 3)

    def test_empty_query_never_enters_model(self):
        model = FakeEncoderModel()
        with self.assertRaises(ValueError):
            BGEM3Encoder(settings(), model=model).encode_queries([""])
        self.assertEqual(model.calls, [])

    def test_release_preserves_query_vectors_and_call_counters(self):
        model = FakeEncoderModel()
        encoder = BGEM3Encoder(settings(), model=model)
        expected = encoder.encode_queries(["已编码查询"])
        with patch("cloudcare.neural._collect_model_memory") as collect:
            encoder.release_model()
        collect.assert_called_once()
        self.assertFalse(encoder.status()["loaded"])
        self.assertEqual(encoder.status()["model_calls"], 1)
        # A cached vector is an earlier real model output, not an alternative
        # scoring algorithm. A cache hit must not reload cleared weights.
        with patch.object(encoder, "_load", side_effect=AssertionError("unnecessary reload")):
            self.assertEqual(encoder.encode_queries(["已编码查询"]), expected)
        self.assertEqual(encoder.query_cache_hits, 1)
        self.assertEqual(encoder.model_calls, 1)

    def test_release_clears_reranker_and_router_weight_references(self):
        reranker = BGEReranker(settings(), model=FakeReranker())
        reranker.model_calls = 7
        config = settings()
        config.bert_model_path = "unused"
        router = SupportBertRouter(config)
        router._model = object()
        router._tokenizer = object()
        router._torch = object()
        with patch("cloudcare.neural._collect_model_memory") as collect:
            reranker.release_model()
            router.release_model()
        self.assertEqual(collect.call_count, 2)
        self.assertFalse(reranker.status()["loaded"])
        self.assertEqual(reranker.model_calls, 7)
        self.assertIsNone(router._model)
        self.assertIsNone(router._tokenizer)
        self.assertIsNone(router._torch)

    def test_rerank_keeps_source_and_parent_metadata(self):
        candidates = [{"id": str(i), "content": "child", "source": f"file-{i}", "parent_id": "parent"}
                      for i in range(3)]
        ranked = BGEReranker(settings(), model=FakeReranker()).rerank("question", candidates)
        self.assertEqual([row["id"] for row in ranked], ["1", "2"])
        self.assertEqual(ranked[0]["source"], "file-1")
        self.assertEqual(ranked[0]["parent_id"], "parent")
        self.assertNotIn("rerank_score", candidates[1])

    def test_filter_uses_quoted_literals(self):
        category = '账单" or tenant_id == "other'
        expected = 'tenant_id == "demo" and visibility == "public" and category == ' + json.dumps(category, ensure_ascii=False)
        self.assertEqual(source_filter(tenant_id="demo", category=category), expected)
        with self.assertRaises(ValueError):
            source_filter(tenant_id="")

    def test_education_collection_rejected_before_connection(self):
        config = settings()
        config.milvus_collection = "education_qa"
        with self.assertRaises(ValueError):
            MilvusStore(config)

    def test_hash_and_tenant_identity(self):
        vector = EncodedVectors([.1] * 1024, {1: .3})
        row = {"id": "A", "content": "文本", "source": "manual.md"}
        first = MilvusStore._row(row, vector)
        second = MilvusStore._row({**row, "tenant_id": "other"}, vector)
        self.assertNotEqual(first["pk"], second["pk"])
        self.assertEqual(first["content_hash"], hashlib.sha256("文本".encode()).hexdigest())
        with self.assertRaises(ValueError):
            MilvusStore._row({**row, "content_hash": "invalid"}, vector)

    def test_wrong_tenant_hit_rejected(self):
        model = FakeEncoderModel()
        client = FakeSearchClient()
        store = MilvusStore(settings(), BGEM3Encoder(settings(), model=model), client=client)
        store._ready = True
        with self.assertRaisesRegex(RuntimeError, "access scope"):
            store.hybrid_search("问题")

    def test_hf_pair_encoding_and_sigmoid_preserve_score_order(self):
        import torch

        class Tokenizer:
            def __init__(self):
                self.batches = []

            def __call__(self, queries, passages, **kwargs):
                self.batches.append((queries, passages))
                return {"input_ids": torch.arange(len(queries)).reshape(-1, 1)}

            def prepare_for_model(self, *args, **kwargs):
                raise AssertionError("Removed tokenizer API must never be used")

        class Model:
            config = SimpleNamespace(num_labels=1)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, input_ids):
                return SimpleNamespace(logits=input_ids.float() * 4 - 2)

        tokenizer = Tokenizer()
        adapter = _HFSequenceReranker("unused", "cpu", 2, 128, tokenizer=tokenizer, model=Model())
        scores = adapter.compute_score([["q", "low"], ["q", "high"], ["q", "last"]])
        self.assertAlmostEqual(scores[0], .1192029, places=6)
        self.assertAlmostEqual(scores[1], .8807971, places=6)
        self.assertTrue(all(0 < value < 1 for value in scores))
        self.assertEqual(tokenizer.batches[0], (["q", "q"], ["low", "high"]))
        self.assertEqual(len(tokenizer.batches), 2)

    def test_compensation_deletes_only_exact_namespaced_primary_keys(self):
        class Client:
            def delete(self, **kwargs):
                self.request = kwargs
                return {"delete_count": len(kwargs["ids"])}

            def flush(self, **kwargs):
                self.flush_request = kwargs

        client = Client()
        store = MilvusStore(settings(), client=client)
        store._ready = True
        deleted = store.delete_documents(["A", "A"], tenant_id="demo")
        expected = hashlib.sha256("demo\0A".encode()).hexdigest()
        self.assertEqual(client.request, {"collection_name": "cloudcare_support_test", "ids": [expected]})
        self.assertEqual(deleted["requested"], 1)
        self.assertEqual(deleted["deleted"], 1)
        with self.assertRaises(ValueError):
            store.delete_documents(["A"], tenant_id="")


if __name__ == "__main__":
    unittest.main()
