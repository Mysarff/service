"""Portable configuration. Credentials stay in ignored INI/.env files."""
from __future__ import annotations

import configparser
from dataclasses import dataclass, field
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _local_env(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    path = root / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name.strip()] = value.strip().strip("\"'")
    values.update(os.environ)
    return values


def _model_path(root: Path, local_relative: str, public_name: str) -> str:
    candidate = root.parent / local_relative
    return str(candidate) if candidate.is_dir() else public_name


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    runtime_dir: Path = ROOT / "runtime"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3308
    mysql_user: str = "cloudcare"
    mysql_password: str = field(default="", repr=False)
    mysql_database: str = "cloudcare_support"
    mysql_pool_size: int = 5
    redis_url: str = field(default="redis://127.0.0.1:6381/0", repr=False)
    redis_prefix: str = "cloudcare:v2:"
    cache_ttl_seconds: int = 300
    session_ttl_seconds: int = 86400
    milvus_uri: str = "http://127.0.0.1:19531"
    milvus_token: str = field(default="", repr=False)
    milvus_database: str = "default"
    milvus_collection: str = "cloudcare_support_v2"
    embedding_model_path: str = "BAAI/bge-m3"
    reranker_model_path: str = "BAAI/bge-reranker-large"
    bert_base_model_path: str = "google-bert/bert-base-chinese"
    bert_model_path: Path = ROOT / "runtime/models/support-router"
    device: str = "auto"
    model_memory_policy: str = "sequential"
    embedding_batch_size: int = 4
    embedding_max_length: int = 512
    reranker_max_length: int = 512
    retrieval_k: int = 20
    rerank_k: int = 5
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    parent_chunk_size: int = 1200
    child_chunk_size: int = 300
    chunk_overlap: int = 50
    max_upload_bytes: int = 10_000_000
    llm_model: str = "qwen-plus"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = field(default="", repr=False)
    llm_proxy: str = field(default="", repr=False)
    llm_timeout_seconds: int = 50
    llm_max_tokens: int = 900
    query_rewrite_enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8088

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        env = _local_env(ROOT)
        ini = configparser.ConfigParser(interpolation=None)
        ini.read(config_path or ROOT / "config.ini", encoding="utf-8")

        def value(name: str, default, section: str = "app", key: str | None = None):
            found = env.get(name, ini.get(section, key or name.removeprefix("CLOUDCARE_").lower(), fallback=str(default)))
            if isinstance(default, bool):
                return str(found).lower() in {"1", "true", "yes", "on"}
            if isinstance(default, int):
                return int(found)
            if isinstance(default, float):
                return float(found)
            return found

        model = env.get("CLOUDCARE_QWEN_MODEL") or ini.get("llm", "model", fallback="qwen-plus")
        base = env.get("CLOUDCARE_QWEN_BASE_URL") or ini.get("llm", "base_url", fallback=ini.get("llm", "dashscope_base_url", fallback=cls.llm_base_url))
        secret = env.get("CLOUDCARE_QWEN_API_KEY") or ini.get("llm", "api_key", fallback=ini.get("llm", "dashscope_api_key", fallback=""))
        if model.startswith("YOUR_"):
            model = "qwen-plus"
        if "YOUR_" in base:
            base = cls.llm_base_url
        if secret.startswith("YOUR_"):
            secret = ""
        settings = cls(
            mysql_host=value("CLOUDCARE_MYSQL_HOST", "127.0.0.1", "mysql", "host"),
            mysql_port=value("CLOUDCARE_MYSQL_PORT", 3308, "mysql", "port"),
            mysql_user=value("CLOUDCARE_MYSQL_USER", "cloudcare", "mysql", "user"),
            mysql_password=value("CLOUDCARE_MYSQL_PASSWORD", "", "mysql", "password"),
            mysql_database=value("CLOUDCARE_MYSQL_DATABASE", "cloudcare_support", "mysql", "database"),
            mysql_pool_size=value("CLOUDCARE_MYSQL_POOL_SIZE", 5, "mysql", "pool_size"),
            redis_url=value("CLOUDCARE_REDIS_URL", "redis://127.0.0.1:6381/0", "redis", "url"),
            redis_prefix=value("CLOUDCARE_REDIS_PREFIX", "cloudcare:v2:", "redis", "prefix"),
            cache_ttl_seconds=value("CLOUDCARE_CACHE_TTL_SECONDS", 300, "redis", "cache_ttl_seconds"),
            session_ttl_seconds=value("CLOUDCARE_SESSION_TTL_SECONDS", 86400, "redis", "session_ttl_seconds"),
            milvus_uri=value("CLOUDCARE_MILVUS_URI", "http://127.0.0.1:19531", "milvus", "uri"),
            milvus_token=value("CLOUDCARE_MILVUS_TOKEN", "", "milvus", "token"),
            milvus_database=value("CLOUDCARE_MILVUS_DATABASE", "default", "milvus", "database"),
            milvus_collection=value("CLOUDCARE_MILVUS_COLLECTION", "cloudcare_support_v2", "milvus", "collection"),
            embedding_model_path=value("CLOUDCARE_BGE_M3_PATH", _model_path(ROOT, "rag_qa/models/bge-m3", "BAAI/bge-m3"), "models", "bge_m3"),
            reranker_model_path=value("CLOUDCARE_RERANKER_PATH", _model_path(ROOT, "rag_qa/models/bge-reranker-large", "BAAI/bge-reranker-large"), "models", "reranker"),
            bert_base_model_path=value("CLOUDCARE_BERT_BASE_PATH", _model_path(ROOT, "rag_qa/core/bert_query_classifier", "google-bert/bert-base-chinese"), "models", "bert_base"),
            bert_model_path=Path(value("CLOUDCARE_BERT_PATH", str(ROOT / "runtime/models/support-router"), "models", "bert")),
            device=value("CLOUDCARE_DEVICE", "auto", "models", "device"),
            model_memory_policy=value("CLOUDCARE_MODEL_MEMORY_POLICY", "sequential", "models", "memory_policy"),
            embedding_batch_size=value("CLOUDCARE_EMBEDDING_BATCH_SIZE", 4, "retrieval", "embedding_batch_size"),
            embedding_max_length=value("CLOUDCARE_EMBEDDING_MAX_LENGTH", 512, "retrieval", "embedding_max_length"),
            reranker_max_length=value("CLOUDCARE_RERANKER_MAX_LENGTH", 512, "retrieval", "reranker_max_length"),
            retrieval_k=value("CLOUDCARE_RETRIEVAL_K", 20, "retrieval", "retrieval_k"),
            rerank_k=value("CLOUDCARE_RERANK_K", 5, "retrieval", "rerank_k"),
            dense_weight=value("CLOUDCARE_DENSE_WEIGHT", .65, "retrieval", "dense_weight"),
            sparse_weight=value("CLOUDCARE_SPARSE_WEIGHT", .35, "retrieval", "sparse_weight"),
            parent_chunk_size=value("CLOUDCARE_PARENT_CHUNK_SIZE", 1200, "retrieval", "parent_chunk_size"),
            child_chunk_size=value("CLOUDCARE_CHILD_CHUNK_SIZE", 300, "retrieval", "child_chunk_size"),
            chunk_overlap=value("CLOUDCARE_CHUNK_OVERLAP", 50, "retrieval", "chunk_overlap"),
            max_upload_bytes=value("CLOUDCARE_MAX_UPLOAD_BYTES", 10_000_000),
            llm_model=model, llm_base_url=base.rstrip("/"), llm_api_key=secret,
            llm_proxy=value("CLOUDCARE_QWEN_PROXY", "", "llm", "proxy"),
            query_rewrite_enabled=value("CLOUDCARE_QUERY_REWRITE_ENABLED", True, "llm", "query_rewrite_enabled"),
            port=value("CLOUDCARE_PORT", 8088),
        )
        if not settings.mysql_database.startswith("cloudcare") or not settings.milvus_collection.startswith("cloudcare") or not settings.redis_prefix.startswith("cloudcare:"):
            raise ValueError("Use independent cloudcare database, collection and Redis prefix")
        if settings.child_chunk_size <= settings.chunk_overlap or settings.parent_chunk_size < settings.child_chunk_size:
            raise ValueError("Invalid parent/child chunk sizes")
        if settings.model_memory_policy not in {'sequential', 'resident'}:
            raise ValueError('Model memory policy must be sequential or resident')
        return settings

    from_env = load

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)


def get_settings() -> Settings:
    return Settings.load()
