"""Preserved corpus identity and indexing safety checks."""
import hashlib
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.initialize import corpus_rows
from scripts.train_support_router import build_dataset


class CorpusInitializationTests(unittest.TestCase):
    def write_corpus(self, root, *, source="data/manual.md", duplicate=False):
        (root / "data").mkdir()
        (root / "data/manual.md").write_text("完整父文档\n两个步骤", encoding="utf-8")
        rows = [{"id": "A", "parent_id": "P", "source": source, "title": "操作指南",
                 "category": "订单", "content": "第一个步骤", "version": "v1", "synthetic": True},
                {"id": "A" if duplicate else "B", "parent_id": "P", "source": source,
                 "title": "操作指南", "category": "订单", "content": "第二个步骤", "synthetic": True}]
        (root / "data/knowledge.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        (root / "data/faq.jsonl").write_text(json.dumps({"id": "FAQ-A", "question": "如何操作", "answer": "步骤",
                                                        "source_id": "A"}), encoding="utf-8")

    def test_preserves_ids_and_linked_parent_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_corpus(root)
            documents, parents, chunks, faq = corpus_rows(root)
            self.assertEqual([row["id"] for row in chunks], ["A", "B"])
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(parents), 1)
            self.assertEqual(chunks[0]["parent_content"].splitlines(), ["完整父文档", "两个步骤"])
            self.assertEqual(chunks[0]["document_id"], parents[0]["document_id"])
            self.assertEqual(faq[0]["category"], "订单")
            self.assertEqual(chunks[0]["content_hash"], hashlib.sha256("第一个步骤".encode()).hexdigest())

    def test_duplicate_knowledge_ids_stop_ingestion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_corpus(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                corpus_rows(root)

    def test_compressed_faq_preserves_index_and_router_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_corpus(root)
            indexed = corpus_rows(root)
            router = build_dataset(root, 42, 32)
            plain = root / "data/faq.jsonl"
            (root / "data/faq.jsonl.gz").write_bytes(gzip.compress(plain.read_bytes()))
            plain.unlink()
            self.assertEqual(corpus_rows(root), indexed)
            self.assertEqual(build_dataset(root, 42, 32), router)

    def test_source_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_corpus(root, source="../private.md")
            with self.assertRaisesRegex(ValueError, "outside"):
                corpus_rows(root)


if __name__ == "__main__":
    unittest.main()
