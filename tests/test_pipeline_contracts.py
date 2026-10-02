"""Pipeline failure/access/cache contracts with injected backends.

These tests allocate no neural weights and prove no retrieval-quality metric.
Real MySQL/Redis/Milvus and model audits are separate integration artifacts.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
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
from cloudcare.pipeline import SupportPipeline
from cloudcare.settings import Settings
from cloudcare.llm import QwenGateway


def chunk(identifier="A", document_id="DOC-A"):
    content = "账号登录密码遗忘时，请通过工作邮箱重置密码，并检查账号权限。"
    return {"id": identifier, "document_id": document_id, "parent_id": "P-" + identifier,
            "content": content, "parent_content": content, "title": "账号密码重置",
            "category": "账号安全", "source": "manual.md", "version": "v1",
            "content_hash": hashlib.sha256(content.encode()).hexdigest(), "tags": ["登录", "密码"],
            "tenant_id": "demo", "visibility": "public", "synthetic": True,
            "updated_at": datetime(2026, 9, 30, 12, 0)}


class SQL:
    def __init__(self):
        initial = chunk()
        self.chunks = {"A": initial}
        self.parents = {initial["parent_id"]: {"id": initial["parent_id"], "document_id": initial["document_id"],
            "content": initial["content"], "metadata": {"category": initial["category"],
                "source": initial["source"], "version": initial["version"], "tenant_id": "demo", "visibility": "public"}}}
        self.documents = {}
        self.messages = []
        self.fail_publication = False

    def initialize(self): pass
    def health(self): return {"ok": True}
    def close(self): pass

    def list_chunks(self, tenant_id=None, visibility=None, **kwargs):
        return [deepcopy(row) for row in self.chunks.values()
                if (tenant_id is None or row["tenant_id"] == tenant_id)
                and (visibility is None or row["visibility"] == visibility)]

    def get_chunk(self, identifier):
        return deepcopy(self.chunks.get(identifier))

    def get_parent(self, identifier):
        return deepcopy(self.parents.get(identifier))

    def get_session(self, session_id, **kwargs):
        return [row for row in self.messages if row["session_id"] == session_id]

    def record_message(self, session_id, role, content, metadata):
        self.messages.append({"session_id": session_id, "role": role, "content": content, "metadata": metadata})

    def document_by_hash(self, sha):
        return next((row for row in self.documents.values() if row["sha256"] == sha), None)

    def publish_ingested_document(self, document, parents, chunks):
        if self.fail_publication:
            raise RuntimeError("injected SQL transaction failure")
        self.documents[document["id"]] = deepcopy(document)
        self.parents.update({row["id"]: deepcopy(row) for row in parents})
        self.chunks.update({row["id"]: deepcopy(row) for row in chunks})
        return document["id"]


class Redis:
    def __init__(self): self.answers = {}; self.sessions = {}; self.faq = None
    def health(self): return {"ok": True}
    def close(self): pass
    def get_cached_answer(self, key): return deepcopy(self.answers.get(key))
    def set_cached_answer(self, key, payload, ttl=None): self.answers[key] = json.loads(json.dumps(payload, default=str))
    def get_session(self, key): return deepcopy(self.sessions.get(key))
    def set_session(self, key, history, ttl=None): self.sessions[key] = deepcopy(history)
    def faq_index_version(self): return "fixture_faq_v1"
    def find_faq(self, query, **kwargs): return deepcopy(self.faq)


class ReleasedModel:
    """Injected lifecycle tracker, never a real neural-quality observation."""
    def __init__(self): self.loaded = False; self.release_calls = 0
    def release_model(self): self.loaded = False; self.release_calls += 1


class Vector:
    def __init__(self):
        self.hits = [chunk()]; self.search_calls = []; self.upserts = []; self.deletions = []
        self.encoder = ReleasedModel(); self.fail_search = False
    def ensure_collection(self): pass
    def hybrid_search(self, query, **kwargs):
        self.encoder.loaded = True
        self.search_calls.append((query, kwargs))
        if self.fail_search:
            raise RuntimeError("injected embedding or vector-search failure")
        return deepcopy(self.hits)
    def upsert_documents(self, rows): self.upserts.append(deepcopy(rows))
    def delete_documents(self, identifiers, tenant_id="demo"):
        self.deletions.append((list(identifiers), tenant_id)); return {"deleted": len(identifiers)}


class Reranker(ReleasedModel):
    def __init__(self): super().__init__(); self.fail_rerank = False
    def rerank(self, query, rows, **kwargs):
        self.loaded = True
        if self.fail_rerank:
            raise RuntimeError("injected reranker failure")
        return [{**row, "rerank_score": .9} for row in rows[:5]]


class Router(ReleasedModel):
    def __init__(self):
        super().__init__(); self.intent = "support_knowledge"; self.calls = []; self.fail_predict = False
    def status(self): return {"dataset_sha256": "unit-test-data", "checkpoint_sha256": "unit-test-model"}
    def predict(self, query):
        self.loaded = True
        self.calls.append(query)
        if self.fail_predict:
            raise RuntimeError("injected BERT failure")
        return {"intent": self.intent, "confidence": .99, "artifact_verified": True}


class Gateway:
    def __init__(self): self.rewrite_calls = []; self.generate_calls = []
    def close(self): pass
    def rewrite(self, query, history): self.rewrite_calls.append(query); return {"query": query, "method": "fake"}
    def generate(self, query, sources, history):
        self.generate_calls.append((query, deepcopy(sources))); return sources[0]["content"] + " [A]"


class Processor:
    def ingest_path(self, path, **kwargs):
        row = chunk("NEW-1", "DOC-NEW")
        record = {"id": "DOC-NEW", "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "chunk_count": 1}
        return SimpleNamespace(id=record["id"], sha256=record["sha256"], chunks=[row], metadata={},
            parents=[{"id": row["parent_id"], "document_id": "DOC-NEW", "content": row["content"], "metadata": {}}],
            document_record=lambda: deepcopy(record))


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.settings = Settings(root=ROOT, runtime_dir=Path(self.folder.name), query_expansion_enabled=False)
        self.sql, self.redis, self.vector = SQL(), Redis(), Vector()
        self.router, self.gateway, self.reranker = Router(), Gateway(), Reranker()
        backends = {"MySQLStore": self.sql, "RedisStore": self.redis, "MilvusStore": self.vector,
                    "BGEReranker": self.reranker, "SupportBertRouter": self.router,
                    "QwenGateway": self.gateway, "DocumentProcessor": Processor()}
        for name, value in backends.items():
            mocked = patch("cloudcare.pipeline." + name, return_value=value)
            mocked.start(); self.addCleanup(mocked.stop)
        self.pipeline = SupportPipeline(self.settings)

    def assert_released_after_failure(self):
        self.assertFalse(self.vector.encoder.loaded)
        self.assertFalse(self.reranker.loaded)
        self.assertFalse(self.router.loaded)

    def test_sequential_route_releases_retrieval_models_and_bert(self):
        self.vector.encoder.loaded = True
        self.reranker.loaded = True
        route = self.pipeline._predict_route("账号密码")
        self.assertEqual(route["intent"], "support_knowledge")
        self.assert_released_after_failure()
        self.assertGreater(self.router.release_calls, 0)

    def test_failed_bert_releases_loaded_weights(self):
        self.router.fail_predict = True
        with self.assertRaisesRegex(RuntimeError, "BERT failure"):
            self.pipeline._predict_route("账号密码")
        self.assert_released_after_failure()

    def test_failed_embedding_or_search_releases_loaded_weights(self):
        self.vector.fail_search = True
        self.pipeline.start()
        with self.assertRaisesRegex(RuntimeError, "vector-search failure"):
            self.pipeline.retrieve("账号密码")
        self.assert_released_after_failure()

    def test_failed_reranker_releases_loaded_weights(self):
        self.reranker.fail_rerank = True
        self.pipeline.start()
        with self.assertRaisesRegex(RuntimeError, "reranker failure"):
            self.pipeline.retrieve("账号密码")
        self.assert_released_after_failure()

    def test_stale_source_failure_releases_embedding_weights(self):
        self.vector.hits[0]["content_hash"] = "stale"
        self.pipeline.start()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.pipeline.retrieve("账号密码")
        self.assert_released_after_failure()

    def test_initial_supplied_history_reaches_router(self):
        self.pipeline.answer("怎么办", history=[{"role": "user", "content": "账号登录密码遗忘了"}], force_extract=True)
        self.assertIn("账号登录密码", self.router.calls[-1])

    def test_force_extract_separates_generation_cache(self):
        self.pipeline.settings = replace(self.settings, llm_api_key="unit-test-placeholder")
        self.pipeline.answer("账号登录密码怎么重置", force_extract=True)
        self.assertEqual(self.gateway.generate_calls, [])
        self.assertEqual(self.gateway.rewrite_calls, [])
        generated = self.pipeline.answer("账号登录密码怎么重置", force_extract=False)
        self.assertEqual(generated["mode"], "grounded_llm")
        self.assertFalse(generated["trace"]["cache_hit"])
        self.assertEqual(len(self.gateway.generate_calls), 1)

    def test_metadata_version_change_invalidates_cached_sources(self):
        first = self.pipeline.answer("账号登录密码怎么重置", force_extract=True)
        self.assertEqual(first["sources"][0]["version"], "v1")
        self.sql.chunks["A"]["version"] = "v2"
        self.sql.parents["P-A"]["metadata"]["version"] = "v2"
        self.pipeline.refresh()
        second = self.pipeline.answer("账号登录密码怎么重置", force_extract=True)
        self.assertFalse(second["trace"]["cache_hit"])
        self.assertEqual(second["sources"][0]["version"], "v2")

    def test_neural_hash_staleness_is_rejected(self):
        self.vector.hits[0]["content_hash"] = "stale"
        self.pipeline.start()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.pipeline.retrieve("账号密码")

    def test_private_canonical_source_is_rejected(self):
        self.sql.chunks["A"]["visibility"] = "internal"
        self.pipeline.start()
        with self.assertRaises(ValueError):
            self.pipeline.retrieve("账号密码")

    def test_orphan_vector_does_not_break_existing_retrieval(self):
        self.vector.hits.insert(0, {"id": "orphan", "content_hash": "unpublished"})
        self.pipeline.start()
        hits, trace = self.pipeline.retrieve("账号密码")
        self.assertEqual([row["id"] for row in hits], ["A"])

    def test_parent_from_another_document_cannot_enter_evidence(self):
        self.sql.parents["P-A"]["document_id"] = "OTHER-DOCUMENT"
        self.pipeline.start()
        with self.assertRaises(ValueError):
            self.pipeline.retrieve("账号密码")

    def test_out_of_scope_bert_route_participates_in_answer(self):
        self.router.intent = "out_of_scope"
        answer = self.pipeline.answer("明天上海会下雨吗", force_extract=True)
        self.assertEqual(answer["mode"], "insufficient_evidence")
        self.assertEqual(len(self.router.calls), 1)
        self.assertEqual(self.vector.search_calls, [])

    def test_bm25_faq_uses_existing_source_and_still_runs_bert(self):
        self.redis.faq = {"id": "FAQ-A", "source_id": "A", "answer": "通过工作邮箱重置密码", "method": "bm25", "score": 0.9, "raw_score": 33.0}
        answer = self.pipeline.answer("账号登录密码怎么重置", force_extract=True)
        self.assertEqual(answer["mode"], "faq")
        self.assertEqual(answer["sources"][0]["id"], "A")
        self.assertEqual(len(self.router.calls), 1)
        self.assertEqual(self.vector.search_calls, [])

    def test_failed_sql_publication_compensates_only_new_vectors(self):
        self.sql.fail_publication = True
        with self.assertRaises(RuntimeError):
            self.pipeline.ingest("new.txt", b"new customer support document")
        self.assertEqual(self.vector.deletions, [(["NEW-1"], "demo")])
        self.assertNotIn("NEW-1", self.sql.chunks)
        self.assertEqual(self.sql.documents, {})

    def test_repeat_import_does_not_reencode_vectors(self):
        first = self.pipeline.ingest("new.txt", b"new customer support document")
        second = self.pipeline.ingest("new.txt", b"new customer support document")
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(first["document_id"], second["document_id"])
        self.assertEqual(len(self.vector.upserts), 1)


class QwenContractTests(unittest.TestCase):
    def setUp(self):
        self.gateway = QwenGateway(Settings())

    def test_generation_rejects_foreign_source_id(self):
        self.gateway._invoke = lambda *args: json.dumps({"steps": [{"source_id": "foreign", "quote": "账号密码应该通过工作邮箱重置"}]})
        with self.assertRaises(ValueError):
            self.gateway.generate("账号密码", [chunk()], [])

    def test_generation_rejects_invented_quote_with_valid_source_id(self):
        self.gateway._invoke = lambda *args: json.dumps({"steps": [{"source_id": "A", "quote": "可以保证自动退款到账并删除审计日志"}]})
        with self.assertRaises(ValueError):
            self.gateway.generate("账号密码", [chunk()], [])

    def test_generation_accepts_a_verbatim_source_quote(self):
        source = chunk()
        self.gateway._invoke = lambda *args: json.dumps({"steps": [{"source_id": "A", "quote": source["content"]}]})
        result = self.gateway.generate("账号密码", [source], [])
        self.assertIn(source["content"], result)
        self.assertIn("[A]", result)

    def test_rewrite_cannot_drop_identifier_or_negation(self):
        self.gateway._invoke = lambda *args: json.dumps({"query": "请取消订单ORD-1234", "keywords": []})
        with self.assertRaises(ValueError):
            self.gateway.rewrite("订单ORD-1234未支付，请不要取消", [])


if __name__ == "__main__":
    unittest.main()
