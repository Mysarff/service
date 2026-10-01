"""FastAPI transport tests; these do not claim database/model acceptance."""
from __future__ import annotations

import unittest
from dataclasses import replace

from fastapi.testclient import TestClient

from cloudcare.api import create_app
from cloudcare.settings import Settings


class TransportPipeline:
    """Transport fixture only; model/storage integration is tested separately."""
    def __init__(self):
        self.calls = []
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def health(self):
        return {"status": "ok", "knowledge_units": 2, "milvus_units": 2, "counts": {}, "categories": []}

    def knowledge(self):
        return {"items": []}

    def answer(self, **payload):
        self.calls.append(("answer", payload))
        return {"answer": "依据原文", "sources": [], "mode": "retrieval_only", "history": [],
                "session_id": "a" * 32}

    def ingest(self, filename, raw, category):
        self.calls.append(("ingest", {"filename": filename, "raw": raw, "category": category}))
        return {"knowledge_units": 1, "duplicate": False}

    def create_ticket(self, summary):
        self.calls.append(("ticket", {"summary": summary}))
        return {"id": "CC-TEST", "summary": summary}


class FullStackApiTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = TransportPipeline()
        self.settings = replace(Settings(), max_upload_bytes=100)
        self.context = TestClient(create_app(self.settings, pipeline=self.pipeline), base_url="http://localhost:8088")
        self.client = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def test_startup_and_chat_passthrough(self):
        self.assertTrue(self.pipeline.started)
        response = self.client.post("/api/chat", json={"query": "忘记密码怎么重置", "history": [],
                                    "force_extract": True, "session_id": None})
        self.assertEqual(response.status_code, 200)
        self.assertIn("elapsed_ms", response.json())
        self.assertTrue(self.pipeline.calls[0][1]["force_extract"])
        self.assertEqual(self.client.get("/api/health").json()["milvus_units"], 2)

    def test_host_and_origin_blocked_before_every_route(self):
        for path in ("/", "/api/health", "/api/knowledge", "/missing"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path, headers={"Host": "attacker.example"}).status_code, 403)
                self.assertEqual(self.client.get(path, headers={"Host": "localhost:1"}).status_code, 403)
        response = self.client.post("/api/chat", json={"query": "测试问题"},
                                    headers={"Origin": "http://attacker.example"})
        self.assertEqual(response.status_code, 403)
        duplicate = self.client.get("/api/health", headers=[("Host", "localhost:8088"), ("Host", "localhost:8088")])
        self.assertEqual(duplicate.status_code, 403)
        self.assertEqual(self.pipeline.calls, [])

    def test_history_types_and_lengths_checked_before_pipeline(self):
        invalid = [
            {"query": "  "}, {"query": 12}, {"query": "Q" * 2001},
            {"query": "问题", "history": [{"role": "system", "content": "禁止"}]},
            {"query": "问题", "history": [{"role": "user", "content": None}]},
            {"query": "问题", "history": [{"role": "user", "content": "x" * 2001}]},
            {"query": "问题", "history": [{"role": "user", "content": "问题"}] * 9},
            {"query": "问题", "force_extract": "false"},
        ]
        for payload in invalid:
            with self.subTest(payload=str(payload)[:100]):
                self.assertEqual(self.client.post("/api/chat", json=payload).status_code, 400)
        self.assertEqual(self.pipeline.calls, [])

    def test_binary_upload_uses_bytes_and_rejects_oversized_files(self):
        response = self.client.post("/api/upload-file", files={"file": ("scan.png", b"png-bytes", "image/png")},
                                    data={"category": "账号安全"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.pipeline.calls[-1][1]["raw"], b"png-bytes")
        response = self.client.post("/api/upload-file", files={"file": ("big.md", b"x" * 101, "text/markdown")})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(len(self.pipeline.calls), 1)

    def test_text_upload_checks_format_and_encoded_byte_limit(self):
        self.assertEqual(self.client.post("/api/upload", json={"filename": "bad.pdf", "content": "伪装文本"}).status_code, 400)
        self.assertEqual(self.client.post("/api/upload", json={"filename": "large.txt", "content": "中" * 34}).status_code, 413)
        response = self.client.post("/api/upload", json={"filename": "help.md", "content": "客服工单规则"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.pipeline.calls[-1][1]["raw"], "客服工单规则".encode())

    def test_declared_oversized_body_and_ticket_validation(self):
        response = self.client.post("/api/chat", content=b"{}", headers={"Content-Length": "1000101"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.post("/api/tickets", json={"summary": " " * 5}).status_code, 400)
        response = self.client.post("/api/tickets", json={"summary": "需要升级故障工单"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ticket"]["id"], "CC-TEST")

    def test_loopback_page_can_load(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("云栈", response.text)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")


if __name__ == "__main__":
    unittest.main()
