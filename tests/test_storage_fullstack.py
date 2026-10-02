"""Opt-in integration tests for dedicated MySQL/Redis services.

Run with CLOUDCARE_INTEGRATION=1 after starting compose.  No fake Redis,
SQLite substitute, production database or FLUSHDB operation is used.
"""
from __future__ import annotations

import hashlib
import os
import unittest
import uuid

from cloudcare.settings import Settings
from cloudcare.storage import MySQLStore, RedisStore


class StorageIsolationTests(unittest.TestCase):
    def test_forbids_original_database_and_shared_redis_prefix(self):
        with self.assertRaises(ValueError):
            MySQLStore(Settings(mysql_database="education"))
        with self.assertRaises(ValueError):
            RedisStore(Settings(redis_prefix="answer:"))


@unittest.skipUnless(os.environ.get("CLOUDCARE_MYSQL_INTEGRATION") == "1", "requires dedicated MySQL service")
class MySQLIntegrationTests(unittest.TestCase):
    def test_mysql_document_publication_rolls_back_all_prior_batches(self):
        from sqlalchemy import event, select
        from sqlalchemy.exc import DataError
        store = MySQLStore(Settings.load())
        self.addCleanup(store.close)
        store.initialize()
        marker = "IT-" + uuid.uuid4().hex[:20]
        sha = hashlib.sha256(marker.encode()).hexdigest()
        doc = {"id": marker, "sha256": sha, "source": "integration-atomic.md", "title": "Atomic publication",
               "category": "integration", "version": "1", "chunk_count": 201}
        parents = [{"id": marker + "-P", "content": "完整处置流程"}]
        chunks = [{"id": marker + f"-C{index:03d}", "parent_id": marker + "-P", "content": f"处置步骤 {index}"}
                  for index in range(201)]
        chunks[-1]["id"] = marker + "X" * 97  # Actual MySQL VARCHAR(96) error in the second batch.
        completed_batches = []

        def after_statement(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.startswith("INSERT INTO knowledge_chunks"):
                completed_batches.append(statement)

        event.listen(store.engine, "after_cursor_execute", after_statement)
        self.addCleanup(event.remove, store.engine, "after_cursor_execute", after_statement)

        def cleanup():
            with store.engine.begin() as connection:
                connection.execute(store.chunks.delete().where(store.chunks.c.document_id == marker))
                connection.execute(store.parents.delete().where(store.parents.c.document_id == marker))
                connection.execute(store.documents.delete().where(store.documents.c.id == marker))

        self.addCleanup(cleanup)
        with self.assertRaises(DataError):
            store.publish_ingested_document(doc, parents, chunks)
        self.assertEqual(len(completed_batches), 1)  # 200 rows were submitted before batch 2 failed.
        self.assertIsNone(store.document_by_hash(sha))
        self.assertIsNone(store.get_parent(marker + "-P"))
        with store.engine.connect() as connection:
            self.assertEqual(connection.execute(select(store.chunks.c.id).where(
                store.chunks.c.document_id == marker)).all(), [])
        published = store.publish_ingested_document({**doc, "chunk_count": 2}, parents, chunks[:2])
        self.assertFalse(published["duplicate"])
        self.assertEqual(published["knowledge_units"], 2)
        self.assertEqual(store.get_chunk(chunks[0]["id"])["parent_id"], parents[0]["id"])
        duplicate = store.publish_ingested_document({**doc, "chunk_count": 2}, parents, chunks[:2])
        self.assertTrue(duplicate["duplicate"])

    def test_mysql_pooled_persistence_and_hash_update(self):
        store = MySQLStore(Settings.load())
        self.addCleanup(store.close)
        store.initialize()
        marker = "IT-" + uuid.uuid4().hex[:20]

        def cleanup():
            with store.engine.begin() as connection:
                connection.execute(store.messages.delete().where(store.messages.c.session_id == marker))
                connection.execute(store.sessions.delete().where(store.sessions.c.id == marker))
                connection.execute(store.tickets.delete().where(store.tickets.c.id == marker + "-T"))
                connection.execute(store.faq.delete().where(store.faq.c.id == marker + "-F"))
                connection.execute(store.chunks.delete().where(store.chunks.c.id == marker + "-C"))
                connection.execute(store.parents.delete().where(store.parents.c.id == marker + "-P"))
                connection.execute(store.documents.delete().where(store.documents.c.id == marker))

        self.addCleanup(cleanup)
        self.assertEqual(store.health()["database"], "cloudcare_support")
        sha = hashlib.sha256(marker.encode()).hexdigest()
        row = {"id": marker, "sha256": sha, "source": "integration.md", "title": "客服处置",
               "category": "integration", "version": "1", "chunk_count": 1, "content_chars": 12}
        self.assertEqual(store.upsert_document(row), marker)
        self.assertEqual(store.upsert_document(row), marker)
        changed_sha = hashlib.sha256((marker + "updated").encode()).hexdigest()
        self.assertEqual(store.upsert_document({**row, "sha256": changed_sha, "version": "2"}), marker)
        self.assertEqual(store.document_by_hash(changed_sha)["version"], "2")
        self.assertIsNone(store.document_by_hash(sha))
        store.upsert_parents([{"id": marker + "-P", "document_id": marker, "content": "完整工单升级流程"}])
        store.upsert_chunks([{"id": marker + "-C", "document_id": marker, "parent_id": marker + "-P",
                              "content": "工单升级", "parent_content": "完整工单升级流程", "category": "integration"}])
        self.assertEqual(store.get_chunk(marker + "-C")["content"], "工单升级")
        self.assertEqual(store.get_parent(marker + "-P")["content"], "完整工单升级流程")
        question = marker + " 怎么升级工单？"
        store.save_faq([{"id": marker + "-F", "question": question, "answer": "请转值班人员",
                         "source_id": marker + "-C"}])
        self.assertEqual(store.find_exact_faq(question)["source_id"], marker + "-C")
        store.record_message(marker, "user", "需要人工跟进")
        store.record_message(marker, "assistant", "请提供脱敏日志")
        self.assertEqual([row["role"] for row in store.get_session(marker)], ["user", "assistant"])
        store.create_ticket({"id": marker + "-T", "session_id": marker, "summary": "故障待处理"})
        self.assertEqual(store.get_ticket(marker + "-T")["question"], "故障待处理")


@unittest.skipUnless(os.environ.get("CLOUDCARE_INTEGRATION") == "1", "requires dedicated MySQL/Redis services")
class StorageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings.load()
        cls.mysql = MySQLStore(cls.settings)
        cls.mysql.initialize()
        cls.redis = RedisStore(cls.settings)

    @classmethod
    def tearDownClass(cls):
        cls.mysql.close()
        cls.redis.close()

    def test_real_document_chunk_faq_session_ticket_roundtrip(self):
        marker = "IT-" + uuid.uuid4().hex[:20]
        self.addCleanup(self._remove_owned_fixture, marker)
        sha = hashlib.sha256(marker.encode()).hexdigest()
        self.assertTrue(self.mysql.health()["ok"])
        self.assertTrue(self.redis.health()["ok"])
        record = {"id": marker, "sha256": sha, "source": "integration-generated.md", "title": "Integration",
                  "category": "integration", "version": "test", "chunk_count": 1, "content_chars": 18,
                  "metadata": {"integration": True}}
        self.assertEqual(self.mysql.upsert_document(record), marker)
        self.assertEqual(self.mysql.upsert_document(record), marker)
        self.assertEqual(self.mysql.document_by_hash(sha)["id"], marker)
        chunk = {"id": marker + "-C", "document_id": marker, "parent_id": marker + "-P", "title": "Test",
                 "category": "integration", "content": "客服故障工单升级步骤", "parent_content": "完整客服故障工单升级步骤",
                 "source": "integration-generated.md", "version": "test", "synthetic": True,
                 "tenant_id": "demo", "visibility": "public", "tags": ["故障"]}
        self.mysql.upsert_chunks([chunk])
        fetched = self.mysql.get_chunk(chunk["id"])
        self.assertEqual(fetched["content"], chunk["content"])
        self.assertEqual(fetched["parent_content"], chunk["parent_content"])
        faq = {"id": marker + "-F", "question": marker + " 如何升级工单？", "answer": "请联系值班人员。",
               "source_id": chunk["id"], "category": "integration"}
        self.mysql.save_faq([faq])
        self.assertEqual(self.mysql.get_faq(faq["id"])["source_id"], chunk["id"])
        self.assertEqual(self.mysql.find_exact_faq(faq["question"])["id"], faq["id"])
        self.mysql.record_message(marker, "user", "工单如何升级？")
        self.mysql.record_message(marker, "assistant", "请核验故障等级。", {"source_id": chunk["id"]})
        messages = self.mysql.get_session(marker)
        self.assertEqual([row["role"] for row in messages], ["user", "assistant"])
        ticket = self.mysql.create_ticket({"id": marker + "-T", "session_id": marker, "question": "故障待处理"})
        self.assertEqual(self.mysql.get_ticket(ticket["id"])["status"], "open")
        self.redis.set_session(marker, [{"role": "user", "content": "升级工单"}], ttl=60)
        self.assertEqual(self.redis.get_session(marker)[0]["content"], "升级工单")
        self.redis.set_cached_answer(marker, {"answer": "核验故障等级", "source_id": chunk["id"],
                                              "source": fetched}, ttl=60)
        self.assertEqual(self.redis.get_cached_answer(marker)["source_id"], chunk["id"])
        self.assertIsInstance(self.redis.get_cached_answer(marker)["source"]["updated_at"], str)
        self.assertGreater(self.redis.client.ttl(self.redis._key("answer", marker)), 0)
        self.redis.clear_session(marker)
        self.assertIsNone(self.redis.get_session(marker))

    def _remove_owned_fixture(self, marker):
        # Remove this test's exact IDs only, so audit counts describe business
        # data rather than retained integration fixtures.
        with self.mysql.engine.begin() as connection:
            connection.execute(self.mysql.messages.delete().where(self.mysql.messages.c.session_id == marker))
            connection.execute(self.mysql.sessions.delete().where(self.mysql.sessions.c.id == marker))
            connection.execute(self.mysql.tickets.delete().where(self.mysql.tickets.c.id == marker + "-T"))
            connection.execute(self.mysql.faq.delete().where(self.mysql.faq.c.id == marker + "-F"))
            connection.execute(self.mysql.chunks.delete().where(self.mysql.chunks.c.id == marker + "-C"))
            connection.execute(self.mysql.documents.delete().where(self.mysql.documents.c.id == marker))

    def test_real_isolated_faq_inverted_index(self):
        # Use an isolated test prefix so the live application index stays intact.
        from dataclasses import replace
        prefix = "cloudcare:integration:" + uuid.uuid4().hex + ":"
        redis = RedisStore(replace(self.settings, redis_prefix=prefix))
        self.addCleanup(redis.close)
        rows = [{"id": "FAQ-A", "question": "账号密码忘记了怎么办", "answer": "使用工作邮箱重置。",
                 "source_id": "ACC-01", "category": "账号安全"},
                {"id": "FAQ-B", "question": "订单退款如何处理", "answer": "核验退款状态。",
                 "source_id": "RET-01", "category": "退换售后"}]
        result = redis.replace_faq_index(rows)
        self.assertEqual(result["count"], 2)
        matched = redis.find_faq("账号密码忘记了怎么办？")
        self.assertEqual(matched["source_id"], "ACC-01")
        self.assertEqual(matched["method"], "bm25")
        approximate = redis.find_faq("账号密码忘记怎么办", min_score=0.5)
        self.assertEqual(approximate["id"], "FAQ-A")
        self.assertEqual(approximate["method"], "bm25")
        self.assertIsNone(redis.find_faq("某天气城市", min_score=0.9))
        redis.replace_faq_index([rows[1]])
        self.assertIsNone(redis.find_faq(rows[0]["question"]))
        self.assertEqual(redis.find_faq(rows[1]["question"])["id"], "FAQ-B")
        # Expire only this test namespace; never delete external/shared keys.
        for key in redis.client.scan_iter(match=prefix + "*"):
            redis.client.expire(key, 60)


if __name__ == "__main__":
    unittest.main()
