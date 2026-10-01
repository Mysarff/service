"""Load the preserved support corpus into actual MySQL, Redis and Milvus.

Source IDs remain stable for the existing retrieval evaluation. FAQ variations
and simulated tickets are not counted as new knowledge or new vector rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudcare.settings import Settings


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def jsonl_bytes(path: Path) -> bytes:
    if path.is_file():
        return path.read_bytes()
    compressed = path.with_suffix(path.suffix + ".gz")
    if compressed.is_file():
        return gzip.decompress(compressed.read_bytes())
    raise FileNotFoundError(f"Missing corpus artifact: {path.name} or {compressed.name}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in jsonl_bytes(path).decode("utf-8").splitlines() if line.strip()]


def scoped_count(store, table, identifiers) -> int:
    from sqlalchemy import func, select
    identifiers = sorted(set(identifiers))
    total = 0
    with store.engine.connect() as connection:
        for offset in range(0, len(identifiers), 200):
            statement = select(func.count()).select_from(table).where(table.c.id.in_(identifiers[offset:offset + 200]))
            total += int(connection.execute(statement).scalar_one())
    return total


def corpus_rows(root: Path = ROOT) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    knowledge = read_jsonl(root / "data/knowledge.jsonl")
    if len({row["id"] for row in knowledge}) != len(knowledge):
        raise ValueError("Knowledge JSONL contains duplicate source IDs")
    sources: dict[str, list[dict]] = defaultdict(list)
    for row in knowledge:
        sources[row["source"]].append(row)
    documents = []
    by_source = {}
    source_contents = {}
    for source, children in sorted(sources.items()):
        path = (root / source).resolve()
        # A source reference cannot redirect the importer outside this project.
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Source path is outside the customer-support project")
        source_bytes = path.read_bytes() if path.is_file() else "\n\n".join(row["content"] for row in children).encode("utf-8")
        source_contents[source] = source_bytes.decode("utf-8-sig")
        identifier = "DOC-" + digest(source.encode("utf-8") + b"\0" + source_bytes)[:24]
        by_source[source] = identifier
        documents.append({"id": identifier, "sha256": digest(source_bytes), "source": source,
                          "title": children[0]["title"].split("｜")[0],
                          "category": children[0]["category"], "version": children[0].get("version", "1"),
                          "content_chars": len(source_contents[source]), "chunk_count": len(children),
                          "metadata": {"synthetic": all(row.get("synthetic", False) for row in children),
                                       "origin": "preserved_cloudcare_corpus", "source_file_exists": path.is_file()}})
    parent_groups: dict[str, list[dict]] = defaultdict(list)
    for row in knowledge:
        parent_groups[row.get("parent_id", row["id"])].append(row)
    parents = []
    parent_contents = {}
    for identifier, children in sorted(parent_groups.items()):
        source_set = {row["source"] for row in children}
        if len(source_set) != 1:
            raise ValueError(f"Parent {identifier} contains multiple source documents")
        source = children[0]["source"]
        content = source_contents[source]
        if len(content.encode("utf-8")) > 65535:
            raise ValueError(f"Parent {identifier} exceeds the Milvus parent-content limit; split the source first")
        parent_contents[identifier] = content
        parents.append({"id": identifier, "document_id": by_source[source], "content": content,
                        "metadata": {"source": source, "category": children[0]["category"],
                                     "version": children[0].get("version", "1"), "child_ids": [row["id"] for row in children],
                                     "synthetic": all(row.get("synthetic", False) for row in children)}})
    chunks = []
    for row in knowledge:
        identifier = row.get("parent_id", row["id"])
        chunks.append({**row, "parent_id": identifier, "parent_content": parent_contents[identifier],
                       "document_id": by_source[row["source"]], "tenant_id": row.get("tenant_id", "demo"),
                       "visibility": row.get("visibility", "public"),
                       "content_hash": digest(row["content"].encode("utf-8"))})
    faq_path = root / "data/faq.jsonl"
    by_id = {row["id"]: row for row in knowledge}
    faq_rows = []
    for faq in read_jsonl(faq_path):
        if faq["source_id"] not in by_id:
            raise ValueError("FAQ references a source ID absent from the knowledge corpus")
        source = by_id[faq["source_id"]]
        faq_rows.append({**faq, "category": source["category"], "tenant_id": source.get("tenant_id", "demo"),
                         "visibility": source.get("visibility", "public"), "version": source.get("version", "1")})
    if len({row["id"] for row in faq_rows}) != len(faq_rows):
        raise ValueError("FAQ contains duplicate IDs")
    return documents, parents, chunks, faq_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--skip-vectors", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("batch-size must be positive")
        settings = replace(settings, embedding_batch_size=args.batch_size)
    from cloudcare.storage import MySQLStore, RedisStore
    documents, parents, chunks, faq = corpus_rows()
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "corpus_sha256": digest((ROOT / "data/knowledge.jsonl").read_bytes()),
              "faq_sha256": digest(jsonl_bytes(ROOT / "data/faq.jsonl")),
              "input_counts": {"source_documents": len(documents), "parents": len(parents),
                               "knowledge_units": len(chunks), "faq_variants": len(faq)},
              "synthetic": True, "vectors_requested": not args.skip_vectors,
              "vector_index_verified": False}
    started = time.perf_counter()
    mysql = MySQLStore(settings)
    redis = RedisStore(settings)
    try:
        mysql.initialize()
        mapping = {}
        for document in documents:
            source_identifier = document['id']
            # A revised seed source replaces its own published document; uploads
            # continue to use content-addressed immutable IDs in the pipeline.
            from sqlalchemy import select
            with mysql.engine.connect() as connection:
                existing = connection.execute(select(mysql.documents).where(
                    mysql.documents.c.source == document['source'])).mappings().all()
            preserved = [row for row in existing if row['metadata'].get('origin') == 'preserved_cloudcare_corpus']
            if len(preserved) > 1:
                raise ValueError('Seed source maps to multiple durable documents')
            if preserved:
                document['id'] = preserved[0]['id']
            mapping[source_identifier] = mysql.upsert_document(document)
        # Existing identical content keeps its original durable document ID.
        for row in parents + chunks:
            row["document_id"] = mapping[row["document_id"]]
        mysql.upsert_parents(parents)
        mysql.upsert_chunks(chunks)
        mysql.save_faq(faq)
        print("MySQL documents, parents, chunks and FAQ persisted", flush=True)
        redis_index = redis.replace_faq_index(faq)
        report["mysql"] = {"health": mysql.health(), "counts": mysql.counts()}
        report["redis"] = {"health": redis.health(), "faq_index": redis_index, "counts": redis.counts()}
        # Verify counts from actual servers rather than submitted list lengths.
        count = report["mysql"]["counts"]
        expected_documents = len(set(mapping.values()))
        corpus_counts = {"documents": scoped_count(mysql, mysql.documents, mapping.values()),
                         "parents": scoped_count(mysql, mysql.parents, (row["id"] for row in parents)),
                         "chunks": scoped_count(mysql, mysql.chunks, (row["id"] for row in chunks)),
                         "faq": scoped_count(mysql, mysql.faq, (row["id"] for row in faq))}
        report["mysql"]["corpus_counts"] = corpus_counts
        report["database_counts_match"] = (corpus_counts["documents"] == expected_documents
                                             and corpus_counts["parents"] == len(parents)
                                             and corpus_counts["chunks"] == len(chunks)
                                             and corpus_counts["faq"] == len(faq)
                                             and report["redis"]["counts"]["faq"] == len(faq))
        if not args.skip_vectors:
            from cloudcare.neural import BGEM3Encoder
            from cloudcare.retrieval import MilvusStore
            print(f"Encoding {len(chunks)} knowledge units with BGE-M3 dense+sparse vectors", flush=True)
            encoder = BGEM3Encoder(settings)
            milvus = MilvusStore(settings, encoder)
            report["ingestion"] = milvus.upsert_documents(chunks)
            report["milvus"] = milvus.audit(expected_count=len(chunks))
            report["encoder"] = encoder.status()
            report["vector_index_verified"] = bool(report["milvus"]["count_matches"] and report["milvus"]["sample_valid"])
            if not report["vector_index_verified"]:
                raise RuntimeError("Actual Milvus row count/vector audit did not match the source corpus")
        else:
            report["milvus"] = {"status": "skipped", "row_count": None, "dense_dimension": None,
                                "reason": "--skip-vectors; no Milvus success is claimed"}
        report["complete"] = bool(report["database_counts_match"] and (args.skip_vectors or report["vector_index_verified"]))
        if not report["database_counts_match"]:
            raise RuntimeError("Actual database corpus counts do not match the submitted knowledge IDs")
    except Exception as error:
        # Avoid printing driver exception text that may include credentials or
        # SQL values. The failure type and failing stage remain auditable.
        report["complete"] = False
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        path = ROOT / "evaluation/fullstack_index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        mysql.close()
        redis.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
