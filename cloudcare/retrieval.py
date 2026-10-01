"""Independent customer-support Milvus collection with dense/sparse retrieval."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .neural import BGEM3Encoder, EncodedVectors

OUTPUT_FIELDS = ["id", "title", "category", "text", "tags", "source", "version",
                 "parent_id", "parent_content", "content_hash", "synthetic", "tenant_id", "visibility"]


def literal(value: str) -> str:
    """Quote a scalar instead of interpolating user text into filter syntax."""
    return json.dumps(value, ensure_ascii=False)


def source_filter(*, tenant_id: str, category: str = "", visibility: str = "public") -> str:
    if not tenant_id:
        raise ValueError("A tenant_id is required for every vector search")
    if visibility not in {"public", "internal"}:
        raise ValueError("visibility must be public or internal")
    clauses = [f"tenant_id == {literal(tenant_id)}", f"visibility == {literal(visibility)}"]
    if category:
        clauses.append(f"category == {literal(category)}")
    return " and ".join(clauses)


class MilvusStore:
    """No method deletes a collection; all writes use the CloudCare namespace."""
    def __init__(self, settings: Any, encoder: BGEM3Encoder | None = None, *, client: Any = None):
        self.settings = settings
        self.collection = settings.milvus_collection
        if not self.collection.startswith("cloudcare_"):
            raise ValueError("Customer-support vectors require a separate cloudcare_ collection")
        self.encoder = encoder or BGEM3Encoder(settings)
        self._client = client
        self._ready = False

    @property
    def client(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=self.settings.milvus_uri,
                                        token=self.settings.milvus_token,
                                        db_name=self.settings.milvus_database, timeout=30)
        return self._client

    def ensure_collection(self) -> None:
        from pymilvus import DataType
        client = self.client
        if client.has_collection(self.collection):
            description = client.describe_collection(self.collection)
            fields = {field["name"]: field for field in description.get("fields", [])}
            required = set(OUTPUT_FIELDS) | {"pk", "dense_vector", "sparse_vector"}
            if not required.issubset(fields):
                raise ValueError("Existing support collection schema is incompatible; use a new collection name")
            dim = fields["dense_vector"].get("params", {}).get("dim")
            if dim is not None and int(dim) != self.encoder.dimension:
                raise ValueError("Milvus dense dimension differs from BGE-M3")
        else:
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=64)
            for name, length in [("id", 256), ("parent_id", 256), ("title", 2048),
                                 ("category", 256), ("source", 4096), ("version", 128),
                                 ("tenant_id", 128), ("visibility", 32), ("content_hash", 64),
                                 ("text", 65535), ("parent_content", 65535)]:
                schema.add_field(name, DataType.VARCHAR, max_length=length)
            schema.add_field("tags", DataType.JSON)
            schema.add_field("synthetic", DataType.BOOL)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.encoder.dimension)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            indexes = client.prepare_index_params()
            indexes.add_index("dense_vector", index_name="dense_index", index_type="HNSW",
                              metric_type="IP", params={"M": 16, "efConstruction": 128})
            indexes.add_index("sparse_vector", index_name="sparse_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                              params={"inverted_index_algo": "DAAT_MAXSCORE"})
            client.create_collection(collection_name=self.collection, schema=schema, index_params=indexes)
        client.load_collection(self.collection)
        self._ready = True

    @staticmethod
    def _row(document: dict[str, Any], vector: EncodedVectors) -> dict[str, Any]:
        tenant_id = str(document.get("tenant_id", "demo"))
        identifier = str(document["id"])
        content = str(document["content"])
        if not tenant_id or not identifier or not content.strip():
            raise ValueError("Document tenant, id and text must be nonempty")
        if len(vector.dense) != 1024 or not vector.sparse:
            raise ValueError("Each Milvus row requires 1024 dense values and nonempty sparse weights")
        if not all(math.isfinite(value) for value in vector.dense + list(vector.sparse.values())):
            raise ValueError("Milvus vectors must contain finite numbers")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if document.get("content_hash") and document["content_hash"] != content_hash:
            raise ValueError(f"Content hash mismatch for {identifier}")
        # Stable tenant/id identity updates versions rather than leaving stale
        # versions searchable. Different tenants can retain the same source ID.
        primary = hashlib.sha256(f"{tenant_id}\0{identifier}".encode("utf-8")).hexdigest()
        return {"pk": primary, "id": identifier, "text": content,
                "title": str(document.get("title", "")), "category": str(document.get("category", "")),
                "tags": document.get("tags", []), "source": str(document.get("source", "")),
                "version": str(document.get("version", "")), "parent_id": str(document.get("parent_id", identifier)),
                "parent_content": str(document.get("parent_content", content)), "content_hash": content_hash,
                "synthetic": bool(document.get("synthetic", False)), "tenant_id": tenant_id,
                "visibility": str(document.get("visibility", "public")),
                "dense_vector": vector.dense, "sparse_vector": vector.sparse}

    def upsert_documents(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._ready:
            self.ensure_collection()
        identities = [(row.get("tenant_id", "demo"), row["id"]) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate tenant/id keys in ingestion batch")
        submitted = 0
        size = self.settings.embedding_batch_size
        for offset in range(0, len(rows), size):
            batch = rows[offset:offset + size]
            texts = [f"{row.get('title', '')}\n{row['content']}" for row in batch]
            vectors = self.encoder.encode_documents(texts)
            data = [self._row(row, vector) for row, vector in zip(batch, vectors, strict=True)]
            result = self.client.upsert(collection_name=self.collection, data=data)
            submitted += int(result.get("upsert_count", len(data)))
        self.client.flush(collection_name=self.collection)
        return {"collection": self.collection, "submitted_rows": submitted, "total_rows": self.count(),
                "dense_dimension": self.encoder.dimension, "sparse_type": "BGE-M3 lexical_weights"}

    def hybrid_search(self, query: str, category: str = "", tenant_id: str = "demo",
                      limit: int = 20, *, visibility: str = "public", fusion: str = "weighted") -> list[dict[str, Any]]:
        if not self._ready:
            self.ensure_collection()
        if not query.strip() or limit <= 0:
            return []
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
        vectors = self.encoder.encode_queries([query])[0]
        expression = source_filter(tenant_id=tenant_id, category=category, visibility=visibility)
        requests = [
            AnnSearchRequest(data=[vectors.dense], anns_field="dense_vector", expr=expression,
                             param={"metric_type": "IP", "params": {"ef": max(64, limit)}}, limit=limit),
            AnnSearchRequest(data=[vectors.sparse], anns_field="sparse_vector", expr=expression,
                             param={"metric_type": "IP", "params": {}}, limit=limit)]
        if fusion == "weighted":
            ranker = WeightedRanker(self.settings.dense_weight, self.settings.sparse_weight)
        elif fusion == "rrf":
            ranker = RRFRanker(k=60)
        else:
            raise ValueError("fusion must be weighted or rrf")
        results = self.client.hybrid_search(collection_name=self.collection, reqs=requests, ranker=ranker,
                                             limit=limit, output_fields=OUTPUT_FIELDS,
                                             consistency_level="Strong")[0]
        rows = []
        for hit in results:
            row = dict(hit["entity"])
            # Defense in depth: a server or SDK regression cannot silently
            # pass a mismatched tenant/visibility into the generation context.
            if row.get("tenant_id") != tenant_id or row.get("visibility") != visibility:
                raise RuntimeError("Milvus returned a row outside the requested access scope")
            if category and row.get("category") != category:
                raise RuntimeError("Milvus returned a row outside the requested category")
            row["content"] = row.pop("text")
            row["score"] = float(hit["distance"])
            row["retrieval_method"] = f"bge_m3_dense_sparse_{fusion}"
            rows.append(row)
        return rows

    def delete_documents(self, identifiers: list[str], tenant_id: str = "demo") -> dict[str, Any]:
        """Compensate a failed new import by exact tenant/id primary keys.

        No free-form filter, collection-name argument or collection deletion is
        accepted. The caller must serialize imports and establish that these
        IDs belong to its unpublished import before requesting compensation.
        """
        if not tenant_id or any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError("Nonempty tenant and source IDs are required for import compensation")
        unique = sorted(set(identifiers))
        if not unique:
            return {"collection": self.collection, "requested": 0, "deleted": 0}
        if not self._ready:
            self.ensure_collection()
        primary_keys = [hashlib.sha256(f"{tenant_id}\0{identifier}".encode("utf-8")).hexdigest()
                        for identifier in unique]
        result = self.client.delete(collection_name=self.collection, ids=primary_keys)
        self.client.flush(collection_name=self.collection)
        return {"collection": self.collection, "requested": len(unique),
                "deleted": int(result.get("delete_count", 0))}

    def count(self, *, tenant_id: str | None = None) -> int:
        if not self._ready:
            self.ensure_collection()
        expression = f"tenant_id == {literal(tenant_id)}" if tenant_id else ""
        result = self.client.query(collection_name=self.collection, filter=expression,
                                   output_fields=["count(*)"], consistency_level="Strong")
        return int(result[0]["count(*)"]) if result else 0

    def audit(self, *, expected_count: int | None = None, sample_size: int = 8) -> dict[str, Any]:
        if not self._ready:
            self.ensure_collection()
        total = self.count()
        rows = self.client.query(collection_name=self.collection, filter="",
                                 output_fields=OUTPUT_FIELDS + ["dense_vector", "sparse_vector"],
                                 limit=sample_size, consistency_level="Strong")
        samples = []
        for row in rows:
            content_matches = hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["content_hash"]
            dimension = len(row["dense_vector"])
            nonzero = len(row["sparse_vector"])
            samples.append({"id": row["id"], "source": row["source"], "parent_id": row["parent_id"],
                            "version": row["version"], "tenant_id": row["tenant_id"],
                            "dense_dimension": dimension, "sparse_nonzero": nonzero,
                            "content_hash_matches": content_matches})
        sample_valid = bool(samples) and all(row["dense_dimension"] == 1024 and row["sparse_nonzero"] > 0
                                             and row["content_hash_matches"] for row in samples)
        return {"collection": self.collection, "row_count": total, "expected_count": expected_count,
                "count_matches": total == expected_count if expected_count is not None else None,
                "dense_dimension": 1024, "vectors_per_knowledge_unit": 2,
                "sparse_type": "BGE-M3 learned lexical weights", "sample_valid": sample_valid,
                "samples": samples, "schema": self.client.describe_collection(self.collection)}


MilvusHybridStore = MilvusStore
