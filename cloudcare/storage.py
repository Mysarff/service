"""Real MySQL persistence and isolated Redis caches for CloudCare.

No in-memory fallback is provided: unavailable services fail explicitly so a
deployment report cannot confuse a local dictionary with MySQL or Redis.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import (JSON, BigInteger, Column, DateTime, ForeignKey, Integer,
                        MetaData, String, Table, Text,
                        create_engine, func, select, text)
from sqlalchemy.dialects.mysql import LONGTEXT, insert as mysql_insert
from sqlalchemy.engine import URL

from .faq import FAQBM25Index, FAQ_BM25_ALGORITHM, FAQ_BM25_B, FAQ_BM25_K1


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _canonical_query(query: str) -> str:
    """Stable SQL question identity; this is not the FAQ answer-routing gate."""
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", str(query)).lower())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MySQLStore:
    """Connection-pooled document/FAQ/session/ticket repository.

    The dedicated database is provisioned by Docker/startup configuration.
    initialize() creates only missing tables; it never drops existing data.
    Each method owns a short connection/transaction, unlike the educational
    client's long-lived shared cursor.
    """

    def __init__(self, settings: Any):
        self.settings = settings
        database = str(getattr(settings, "mysql_database", "cloudcare_support"))
        if not re.fullmatch(r"cloudcare_[a-zA-Z0-9_]+", database):
            raise ValueError("CloudCare 必须使用独立的 cloudcare_ 前缀数据库")
        url = URL.create(
            "mysql+pymysql", username=getattr(settings, "mysql_user", "cloudcare"),
            password=getattr(settings, "mysql_password", ""),
            host=getattr(settings, "mysql_host", "127.0.0.1"),
            port=int(getattr(settings, "mysql_port", 3308)), database=database,
            query={"charset": "utf8mb4"},
        )
        self.engine = create_engine(
            url, pool_size=int(getattr(settings, "mysql_pool_size", 5)),
            max_overflow=int(getattr(settings, "mysql_pool_overflow", 5)),
            pool_pre_ping=True, pool_recycle=1800, pool_timeout=10,
            connect_args={"connect_timeout": 5, "read_timeout": 10, "write_timeout": 10},
            hide_parameters=True, echo=False,
        )
        metadata = self.metadata = MetaData()
        self.documents = Table("documents", metadata,
            Column("id", String(96), primary_key=True),
            Column("sha256", String(64), nullable=False, unique=True),
            Column("source", String(500), nullable=False),
            Column("title", String(255), nullable=False),
            Column("category", String(128), nullable=False),
            Column("version", String(64), nullable=False),
            Column("content_chars", BigInteger, nullable=False, default=0),
            Column("chunk_count", Integer, nullable=False, default=0),
            Column("metadata", JSON, nullable=False),
            Column("created_at", DateTime, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )
        self.chunks = Table("knowledge_chunks", metadata,
            Column("id", String(96), primary_key=True),
            Column("document_id", String(96), nullable=True, index=True),
            Column("parent_id", String(96), nullable=False, index=True),
            Column("title", String(255), nullable=False),
            Column("category", String(128), nullable=False, index=True),
            Column("content", LONGTEXT, nullable=False),
            Column("parent_content", LONGTEXT, nullable=False),
            Column("content_hash", String(64), nullable=False, index=True),
            Column("source", String(500), nullable=False),
            Column("version", String(64), nullable=False),
            Column("tenant_id", String(96), nullable=False, index=True),
            Column("visibility", String(32), nullable=False),
            Column("synthetic", Integer, nullable=False),
            Column("tags", JSON, nullable=False),
            Column("metadata", JSON, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )
        self.parents = Table("knowledge_parents", metadata,
            Column("id", String(96), primary_key=True),
            Column("document_id", String(96), nullable=True, index=True),
            Column("content", LONGTEXT, nullable=False),
            Column("metadata", JSON, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )
        self.faq = Table("faq", metadata,
            Column("id", String(96), primary_key=True),
            Column("question", Text, nullable=False),
            Column("question_hash", String(64), nullable=False, index=True),
            Column("answer", LONGTEXT, nullable=False),
            Column("source_id", String(96), nullable=False, index=True),
            Column("category", String(128), nullable=False, index=True),
            Column("metadata", JSON, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )
        self.sessions = Table("sessions", metadata,
            Column("id", String(96), primary_key=True),
            Column("created_at", DateTime, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )
        self.messages = Table("messages", metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("session_id", String(96), ForeignKey("sessions.id"), nullable=False, index=True),
            Column("role", String(24), nullable=False),
            Column("content", LONGTEXT, nullable=False),
            Column("metadata", JSON, nullable=False),
            Column("created_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )
        self.tickets = Table("tickets", metadata,
            Column("id", String(96), primary_key=True),
            Column("session_id", String(96), nullable=True, index=True),
            Column("question", LONGTEXT, nullable=False),
            Column("status", String(32), nullable=False),
            Column("metadata", JSON, nullable=False),
            Column("created_at", DateTime, nullable=False),
            mysql_charset="utf8mb4",
        )

    def initialize(self) -> None:
        self.metadata.create_all(self.engine, checkfirst=True)

    def health(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            database = connection.execute(text("SELECT DATABASE()" )).scalar_one()
        return {"ok": True, "backend": "mysql", "database": database, "pool": self.engine.pool.status()}

    def upsert_document(self, document: dict[str, Any], *, connection: Any = None) -> str:
        now = _now()
        value = {
            "id": str(document["id"]), "sha256": str(document["sha256"]),
            "source": str(document.get("source", "")), "title": str(document.get("title", "")),
            "category": str(document.get("category", "uploaded")), "version": str(document.get("version", "1")),
            "content_chars": int(document.get("content_chars", 0)),
            "chunk_count": int(document.get("chunk_count", 0)), "metadata": document.get("metadata", {}),
            "created_at": now, "updated_at": now,
        }
        statement = mysql_insert(self.documents).values(value)
        # On duplicate content, preserve its durable original ID and timestamp;
        # a deliberate update of the same ID may also change its content hash.
        statement = statement.on_duplicate_key_update(**{
            name: statement.inserted[name] for name in value if name not in {"id", "created_at"}
        })
        def execute(current):
            current.execute(statement)
            return current.execute(select(self.documents.c.id).where(
                self.documents.c.sha256 == value["sha256"])).scalar_one()
        if connection is not None:
            return execute(connection)
        with self.engine.begin() as current:
            return execute(current)

    def document_by_hash(self, sha256: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.documents).where(self.documents.c.sha256 == sha256)).mappings().first()
        return dict(row) if row else None

    def upsert_parents(self, rows: Iterable[dict[str, Any]], *, connection: Any = None) -> int:
        values = [{"id": str(row["id"]), "document_id": row.get("document_id"),
                   "content": str(row.get("content", "")), "metadata": row.get("metadata", {}),
                   "updated_at": _now()} for row in rows]
        self._bulk_upsert(self.parents, values, immutable={"id"}, connection=connection)
        return len(values)

    def upsert_chunks(self, rows: Iterable[dict[str, Any]], *, connection: Any = None) -> int:
        values = []
        for row in rows:
            content = str(row.get("content", ""))
            values.append({
                "id": str(row["id"]), "document_id": row.get("document_id"),
                "parent_id": str(row.get("parent_id", "")), "title": str(row.get("title", "")),
                "category": str(row.get("category", "uploaded")), "content": content,
                "parent_content": str(row.get("parent_content", "")),
                "content_hash": _digest(content), "source": str(row.get("source", "")),
                "version": str(row.get("version", "1")), "tenant_id": str(row.get("tenant_id", "demo")),
                "visibility": str(row.get("visibility", "public")), "synthetic": int(bool(row.get("synthetic", False))),
                "tags": list(row.get("tags", [])),
                "metadata": {key: value for key, value in row.items() if key not in {
                    "content", "parent_content", "tags", "metadata",
                }} | dict(row.get("metadata", {})), "updated_at": _now(),
            })
        self._bulk_upsert(self.chunks, values, immutable={"id"}, connection=connection)
        return len(values)

    def _bulk_upsert(self, table: Table, values: list[dict[str, Any]], immutable: set[str],
                     *, connection: Any = None) -> None:
        # Bounded batches avoid enormous packets. A caller-supplied connection
        # keeps all batches inside one atomic document-publication transaction.
        for offset in range(0, len(values), 200):
            batch = values[offset:offset + 200]
            statement = mysql_insert(table).values(batch)
            statement = statement.on_duplicate_key_update(**{
                name: statement.inserted[name] for name in batch[0] if name not in immutable
            })
            if connection is not None:
                connection.execute(statement)
            else:
                with self.engine.begin() as current:
                    current.execute(statement)

    def publish_ingested_document(self, document: dict[str, Any], parents: list[dict[str, Any]],
                                  chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Atomically publish one parsed document and all its parent/child rows.

        Expensive parsing/embedding happens before this short SQL transaction.
        Any SQL or validation failure rolls back metadata and every chunk batch.
        Content already published is returned as a duplicate without rewriting
        its source metadata. The HTTP pipeline serializes concurrent uploads.
        """
        parent_ids = {row["id"] for row in parents}
        child_ids = {row["id"] for row in chunks}
        if len(parent_ids) != len(parents) or len(child_ids) != len(chunks):
            raise ValueError("Duplicate IDs in document publication")
        if any(row.get("parent_id") not in parent_ids for row in chunks):
            raise ValueError("A child chunk references an absent parent")
        if int(document.get("chunk_count", len(chunks))) != len(chunks):
            raise ValueError("Document chunk count differs from parsed records")
        with self.engine.begin() as connection:
            existing = connection.execute(select(self.documents).where(
                self.documents.c.sha256 == str(document["sha256"]))).mappings().first()
            if existing:
                return {"document_id": existing["id"], "duplicate": True,
                        "knowledge_units": existing["chunk_count"]}
            reused_id = connection.execute(select(self.documents.c.id).where(
                self.documents.c.id == str(document["id"]))).first()
            if reused_id:
                raise ValueError("New document publication cannot overwrite an existing document ID")
            identifier = self.upsert_document(document, connection=connection)
            parent_rows = [{**row, "document_id": identifier} for row in parents]
            child_rows = [{**row, "document_id": identifier} for row in chunks]
            self.upsert_parents(parent_rows, connection=connection)
            self.upsert_chunks(child_rows, connection=connection)
        return {"document_id": identifier, "duplicate": False,
                "parents": len(parents), "knowledge_units": len(chunks)}

    @staticmethod
    def _chunk_record(row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row.get("metadata", {})) | row
        result["synthetic"] = bool(result["synthetic"])
        return result

    def get_chunk(self, identifier: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.chunks).where(self.chunks.c.id == identifier)).mappings().first()
        return self._chunk_record(dict(row)) if row else None

    def get_parent(self, identifier: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.parents).where(self.parents.c.id == identifier)).mappings().first()
        return dict(row) if row else None

    def list_chunks(self, category: str = "", limit: int = 100_000, tenant_id: str | None = None,
                    visibility: str | None = None) -> list[dict[str, Any]]:
        statement = select(self.chunks).order_by(self.chunks.c.id).limit(max(1, min(int(limit), 1_000_000)))
        if category:
            statement = statement.where(self.chunks.c.category == category)
        if tenant_id is not None:
            statement = statement.where(self.chunks.c.tenant_id == tenant_id)
        if visibility is not None:
            statement = statement.where(self.chunks.c.visibility == visibility)
        with self.engine.connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [self._chunk_record(dict(row)) for row in rows]

    def save_faq(self, rows: Iterable[dict[str, Any]]) -> int:
        values = []
        for row in rows:
            question = str(row.get("question", row.get("query", "")))
            if not question.strip():
                raise ValueError("FAQ 问题不能为空")
            values.append({
                "id": str(row.get("id") or "FAQ-" + _digest(question)[:24]),
                "question": question, "question_hash": _digest(_canonical_query(question)),
                "answer": str(row.get("answer", "")),
                "source_id": str(row.get("source_id", row.get("knowledge_id", ""))),
                "category": str(row.get("category", "")), "metadata": dict(row), "updated_at": _now(),
            })
        self._bulk_upsert(self.faq, values, immutable={"id"})
        return len(values)

    def get_faq(self, identifier: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.faq).where(self.faq.c.id == identifier)).mappings().first()
        return dict(row.get("metadata", {})) | dict(row) if row else None

    def list_faq(self, limit: int = 100_000) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(self.faq).order_by(self.faq.c.id).limit(limit)).mappings().all()
        return [dict(row.get("metadata", {})) | dict(row) for row in rows]

    def find_exact_faq(self, query: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.faq).where(
                self.faq.c.question_hash == _digest(_canonical_query(query))).order_by(self.faq.c.id)).mappings().first()
        return dict(row.get("metadata", {})) | dict(row) if row else None

    def record_message(self, session_id: str, role: str, content: str,
                       metadata: dict[str, Any] | None = None) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("未知会话角色")
        if not session_id or len(session_id) > 96:
            raise ValueError("会话 ID 长度必须为 1 到 96")
        now = _now()
        with self.engine.begin() as connection:
            statement = mysql_insert(self.sessions).values(id=session_id, created_at=now, updated_at=now)
            connection.execute(statement.on_duplicate_key_update(updated_at=now))
            result = connection.execute(self.messages.insert().values(
                session_id=session_id, role=role, content=content, metadata=metadata or {}, created_at=now,
            ))
            return int(result.inserted_primary_key[0])

    def get_session(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(select(self.messages).where(self.messages.c.session_id == session_id)
                .order_by(self.messages.c.id.desc()).limit(max(1, min(int(limit), 1000)))).mappings().all()
        return [dict(row) for row in reversed(rows)]

    def create_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        identifier = str(ticket.get("id") or "CC-" + uuid.uuid4().hex[:20])
        question = str(ticket.get("question", ticket.get("summary", ticket.get("description", ""))))
        if not question.strip():
            raise ValueError("工单问题不能为空")
        row = {"id": identifier, "session_id": ticket.get("session_id"), "question": question,
               "status": str(ticket.get("status", "open")), "metadata": dict(ticket), "created_at": _now()}
        with self.engine.begin() as connection:
            connection.execute(self.tickets.insert().values(row))
        return row

    def get_ticket(self, identifier: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(self.tickets).where(self.tickets.c.id == identifier)).mappings().first()
        return dict(row) if row else None

    def counts(self) -> dict[str, int]:
        result = {}
        with self.engine.connect() as connection:
            for name, table in {"documents": self.documents, "parents": self.parents, "chunks": self.chunks,
                                "faq": self.faq, "sessions": self.sessions, "messages": self.messages,
                                "tickets": self.tickets}.items():
                result[name] = int(connection.execute(select(func.count()).select_from(table)).scalar_one())
        return result

    def close(self) -> None:
        self.engine.dispose()


class RedisStore:
    """Redis answer/session cache and FAQ records with versioned BM25 snapshots."""

    def __init__(self, settings: Any):
        import redis
        self.settings = settings
        self.prefix = str(getattr(settings, "redis_prefix", "cloudcare:v2:"))
        if not self.prefix.startswith("cloudcare:") or not self.prefix.endswith(":"):
            raise ValueError("Redis 键前缀必须为独立的 cloudcare: 命名空间，并以冒号结束")
        self.client = redis.Redis.from_url(
            getattr(settings, "redis_url", "redis://127.0.0.1:6381/0"), decode_responses=True,
            socket_connect_timeout=5, socket_timeout=5, health_check_interval=30,
        )
        self.cache_ttl = int(getattr(settings, "cache_ttl_seconds", 300))
        self.session_ttl = int(getattr(settings, "session_ttl_seconds", 86400))
        self._faq_lock = threading.Lock()
        self._faq_version: str | None = None
        self._faq_index: FAQBM25Index | None = None

    def health(self) -> dict[str, Any]:
        return {"ok": bool(self.client.ping()), "backend": "redis", "prefix": self.prefix}

    def _key(self, family: str, identifier: str) -> str:
        return f"{self.prefix}{family}:{_digest(str(identifier))}"

    def get_cached_answer(self, cache_key: str) -> dict[str, Any] | None:
        value = self.client.get(self._key("answer", cache_key))
        return json.loads(value) if value is not None else None

    def set_cached_answer(self, cache_key: str, payload: dict[str, Any], ttl: int | None = None) -> None:
        self.client.set(self._key("answer", cache_key), json.dumps(payload, ensure_ascii=False, default=str),
                        ex=max(1, int(ttl if ttl is not None else self.cache_ttl)))

    def get_session(self, session_id: str) -> list[dict[str, Any]] | None:
        value = self.client.get(self._key("session", session_id))
        return json.loads(value) if value is not None else None

    def set_session(self, session_id: str, messages: list[dict[str, Any]], ttl: int | None = None) -> None:
        self.client.set(self._key("session", session_id), json.dumps(messages, ensure_ascii=False, default=str),
                        ex=max(1, int(ttl if ttl is not None else self.session_ttl)))

    def clear_session(self, session_id: str) -> None:
        self.client.delete(self._key("session", session_id))

    def replace_faq_index(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Build a version beside the current one, then atomically publish it.

        Redis retains actual questions, answers and source IDs. Each worker
        builds a BM25 question index once per version; it is not a fallback
        store and cannot serve FAQ data when Redis is unavailable. Legacy
        versions containing a records hash remain readable without reingestion.
        """
        rows = [dict(row) for row in records]
        version = uuid.uuid4().hex
        namespace = f"{self.prefix}faq:{version}:"
        record_map: dict[str, str] = {}
        indexed_rows = []
        for row in rows:
            question = str(row.get("question", row.get("query", "")))
            if not question.strip():
                raise ValueError("FAQ 问题不能为空")
            identifier = str(row.get("id") or "FAQ-" + _digest(question)[:24])
            row["id"] = identifier
            row["question"] = question
            row["source_id"] = str(row.get("source_id", row.get("knowledge_id", "")))
            if identifier in record_map:
                raise ValueError("FAQ ID 必须唯一")
            record_map[identifier] = json.dumps(row, ensure_ascii=False, default=str)
            indexed_rows.append(row)
        snapshot = FAQBM25Index(sorted(indexed_rows, key=lambda row: row["id"]))
        metadata = {"algorithm": FAQ_BM25_ALGORITHM, "count": len(record_map),
                    "terms": len(snapshot.postings), "avg_length": snapshot.avg_length,
                    "k1": FAQ_BM25_K1, "b": FAQ_BM25_B}
        # Publish the pointer only after all records and metadata are written.
        with self.client.pipeline(transaction=False) as pipeline:
            items = list(record_map.items())
            for offset in range(0, len(items), 500):
                pipeline.hset(namespace + "records", mapping=dict(items[offset:offset + 500]))
            pipeline.hset(namespace + "metadata", mapping=metadata)
            pipeline.execute()
        old_version = self.client.get(self.prefix + "faq:active")
        with self.client.pipeline(transaction=True) as pipeline:
            pipeline.set(self.prefix + "faq:active", version)
            pipeline.set(self.prefix + "faq:count", len(record_map))
            if old_version and old_version != version:
                for suffix in ("records", "metadata", "exact", "postings"):
                    pipeline.expire(f"{self.prefix}faq:{old_version}:{suffix}", 3600)
            pipeline.execute()
        with self._faq_lock:
            self._faq_version, self._faq_index = version, snapshot
        return {**metadata, "version": version}

    def faq_index_version(self) -> str:
        return self.client.get(self.prefix + "faq:active") or ""

    def find_faq(self, query: str, category: str = "", min_score: float = 0.0,
                 index_version: str | None = None) -> dict[str, Any] | None:
        """Return the highest scoring FAQ question, using raw BM25 only.

        The answer pipeline requests the unfiltered top candidate so its trace
        records the score even when it fails the configured direct-answer gate.
        """
        if not math.isfinite(min_score) or min_score < 0:
            raise ValueError("FAQ BM25 minimum score must be finite and non-negative")
        version = self.faq_index_version() if index_version is None else index_version
        if not version:
            return None
        namespace = f"{self.prefix}faq:{version}:"
        with self._faq_lock:
            if self._faq_index is None or self._faq_version != version:
                record_map = self.client.hgetall(namespace + "records")
                rows = [json.loads(raw) for _, raw in sorted(record_map.items())]
                self._faq_index = FAQBM25Index(rows)
                self._faq_version = version
            snapshot = self._faq_index
        best = snapshot.search(query, category)
        if not best or best["score"] < min_score:
            return None
        return {**best, "index_version": version, "algorithm": FAQ_BM25_ALGORITHM,
                "k1": FAQ_BM25_K1, "b": FAQ_BM25_B}

    def counts(self) -> dict[str, int]:
        return {"faq": int(self.client.get(self.prefix + "faq:count") or 0)}

    def close(self) -> None:
        self.client.close()
