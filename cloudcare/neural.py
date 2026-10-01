"""Local BGE inference and a separately trained customer-support BERT router.

Missing neural models raise an explicit error. No lexical scorer is substituted
for an embedding, cross-encoder, or BERT classification score.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import gc
import json
from pathlib import Path
import threading
from typing import Any, Iterable

SUPPORT_LABELS = ("support_knowledge", "handoff", "out_of_scope")


def _file_sha256(path: Path) -> str:
    """Verify large local checkpoints without allocating their entire bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_model_memory() -> None:
    gc.collect()
    import torch
    # Unit tests and CPU deployments should not initialize a CUDA context just
    # to release a model. Empty only a context previously used by inference.
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return device


@dataclass(frozen=True)
class EncodedVectors:
    dense: list[float]
    sparse: dict[int, float]

    def as_dict(self) -> dict[str, Any]:
        return {"dense": self.dense, "sparse": self.sparse}


class BGEM3Encoder:
    """BGE-M3 dense and learned sparse vectors; bounded query cache."""
    dimension = 1024

    def __init__(self, settings: Any, *, model: Any = None, query_cache_size: int = 256):
        self.settings = settings
        self.model_identifier = str(settings.embedding_model_path)
        self.model_path = Path(settings.embedding_model_path)
        self._model = model
        self._lock = threading.RLock()
        self._query_cache: OrderedDict[str, EncodedVectors] = OrderedDict()
        self._cache_size = max(0, query_cache_size)
        self.model_calls = 0
        self.query_cache_hits = 0

    def _load(self) -> Any:
        if self._model is None:
            if self.model_path.is_dir() and not (self.model_path / "sparse_linear.pt").is_file():
                raise FileNotFoundError(f"BGE-M3 sparse head missing: {self.model_path}")
            from FlagEmbedding import BGEM3FlagModel
            device = resolve_device(self.settings.device)
            self._model = BGEM3FlagModel(
                self.model_identifier, devices=device, use_fp16=device == "cuda",
                batch_size=self.settings.embedding_batch_size,
                query_max_length=self.settings.embedding_max_length,
                passage_max_length=self.settings.embedding_max_length,
                return_dense=True, return_sparse=True, return_colbert_vecs=False,
                trust_remote_code=False)
        return self._model

    def _encode(self, texts: list[str], *, queries: bool) -> list[EncodedVectors]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("BGE-M3 input must be nonempty text")
        model = self._load()
        method = model.encode_queries if queries and hasattr(model, "encode_queries") else model.encode
        output = method(texts, batch_size=self.settings.embedding_batch_size,
                        max_length=self.settings.embedding_max_length,
                        return_dense=True, return_sparse=True, return_colbert_vecs=False)
        self.model_calls += 1
        rows = []
        for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"], strict=True):
            dense_row = [float(value) for value in dense]
            sparse_row = {int(index): float(weight) for index, weight in sparse.items() if float(weight) > 0}
            if len(dense_row) != self.dimension or not sparse_row:
                raise ValueError("BGE-M3 must return 1024 dense dimensions and a nonempty sparse vector")
            rows.append(EncodedVectors(dense_row, sparse_row))
        if len(rows) != len(texts):
            raise ValueError("BGE-M3 returned a different number of vectors than input texts")
        return rows

    def encode_documents(self, texts: Iterable[str]) -> list[EncodedVectors]:
        with self._lock:
            return self._encode(list(texts), queries=False)

    def encode_queries(self, texts: Iterable[str]) -> list[EncodedVectors]:
        queries = list(texts)
        with self._lock:
            result: list[EncodedVectors | None] = [None] * len(queries)
            missing: dict[str, list[int]] = {}
            for index, query in enumerate(queries):
                if query in self._query_cache:
                    result[index] = self._query_cache[query]
                    self._query_cache.move_to_end(query)
                    self.query_cache_hits += 1
                else:
                    missing.setdefault(query, []).append(index)
            vectors = self._encode(list(missing), queries=True)
            for (query, positions), vector in zip(missing.items(), vectors, strict=True):
                for index in positions:
                    result[index] = vector
                if self._cache_size:
                    self._query_cache[query] = vector
                    while len(self._query_cache) > self._cache_size:
                        self._query_cache.popitem(last=False)
            return [vector for vector in result if vector is not None]

    def status(self) -> dict[str, Any]:
        return {"component": "BGE-M3", "loaded": self._model is not None,
                "model_path": str(self.model_path), "dense_dimension": self.dimension,
                "sparse_type": "BGE-M3 lexical_weights", "query_cache_entries": len(self._query_cache),
                "query_cache_hits": self.query_cache_hits, "model_calls": self.model_calls}

    def release_model(self) -> None:
        """Drop weights while retaining bounded Python query-vector cache."""
        with self._lock:
            self._model = None
            _collect_model_memory()


class _HFSequenceReranker:
    """Direct HF pair encoding avoids removed prepare_for_model in v5.

    Uses the original BGE sequence-classification checkpoint, including its
    pretrained scoring head. No reranker training or lexical scoring is added.
    """
    def __init__(self, model_identifier: str, device: str, batch_size: int,
                 max_length: int, *, tokenizer: Any = None, model: Any = None):
        import torch
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        if tokenizer is None or model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            local = Path(model_identifier).is_dir()
            tokenizer = AutoTokenizer.from_pretrained(model_identifier, local_files_only=local)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_identifier, local_files_only=local,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map={"": device}, low_cpu_mem_usage=True)
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        if self.model.config.num_labels != 1:
            raise ValueError("BGE reranker checkpoint must have a single relevance logit")

    def compute_score(self, pairs: list[list[str]], *, normalize: bool = True,
                      max_length: int | None = None) -> list[float]:
        scores = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset:offset + self.batch_size]
            encoded = self.tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                                     padding=True, truncation=True,
                                     max_length=max_length or self.max_length, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.reshape(-1).float()
                values = self.torch.sigmoid(logits) if normalize else logits
                scores.extend(values.cpu().tolist())
        return scores


class BGEReranker:
    """BGE cross-encoder scores retaining child and source trace metadata."""
    def __init__(self, settings: Any, *, model: Any = None):
        self.settings = settings
        self.model_identifier = str(settings.reranker_model_path)
        self.model_path = Path(settings.reranker_model_path)
        self._model = model
        self._lock = threading.RLock()
        self.model_calls = 0

    def _load(self) -> Any:
        if self._model is None:
            device = resolve_device(self.settings.device)
            self._model = _HFSequenceReranker(
                self.model_identifier, device=device,
                max_length=self.settings.reranker_max_length,
                batch_size=self.settings.embedding_batch_size)
        return self._model

    def rerank(self, query: str, candidates: list[dict[str, Any]], k: int | None = None) -> list[dict[str, Any]]:
        if not candidates:
            return []
        limit = self.settings.rerank_k if k is None else k
        if limit <= 0:
            return []
        # Score the retrieved child; keep full parent text for later expansion.
        pairs = [[query, f"{row.get('title', '')}\n{row.get('content', row.get('text', ''))}"]
                 for row in candidates]
        with self._lock:
            scores = self._load().compute_score(pairs, normalize=True,
                                                max_length=self.settings.reranker_max_length)
            self.model_calls += 1
        if isinstance(scores, (int, float)):
            scores = [scores]
        if len(scores) != len(candidates):
            raise ValueError("BGE reranker returned a different number of scores")
        ranked = []
        for original_index, (candidate, score) in enumerate(zip(candidates, scores, strict=True)):
            row = dict(candidate)
            row["rerank_score"] = float(score)
            row["retrieval_rank"] = original_index + 1
            ranked.append(row)
        ranked.sort(key=lambda row: (-row["rerank_score"], row["retrieval_rank"]))
        return ranked[:limit]

    def status(self) -> dict[str, Any]:
        return {"component": "BGE-Reranker", "loaded": self._model is not None,
                "model_path": str(self.model_path), "model_calls": self.model_calls,
                "inference": "HF AutoTokenizer pair encoding + pretrained sequence-classification logits"}

    def release_model(self) -> None:
        with self._lock:
            self._model = None
            _collect_model_memory()


class SupportBertRouter:
    """Accepts only a support checkpoint with a task and dataset report."""
    def __init__(self, settings: Any):
        self.settings = settings
        self.model_path = Path(settings.bert_model_path)
        self._tokenizer = None
        self._model = None
        self._report: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        report_path = self.model_path / "training_report.json"
        if not report_path.is_file():
            raise FileNotFoundError("Support BERT is not trained; run scripts/train_support_router.py")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("labels") != list(SUPPORT_LABELS) or report.get("task") != "customer_support_routing":
            raise ValueError("BERT checkpoint contains education labels or an unknown routing task")
        weights = self.model_path / "model.safetensors"
        if not weights.is_file() or _file_sha256(weights) != report.get("checkpoint_sha256"):
            raise ValueError("BERT weights do not match the recorded customer-support training artifact")
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer
        self._tokenizer = BertTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self._model = BertForSequenceClassification.from_pretrained(str(self.model_path), local_files_only=True)
        if tuple(self._model.config.id2label[index] for index in range(3)) != SUPPORT_LABELS:
            self._model = None
            raise ValueError("BERT checkpoint label map differs from support routing report")
        self._device = resolve_device(self.settings.device)
        self._model.to(self._device).eval()
        self._report = report
        self._torch = torch

    def predict(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("A nonempty query is required for BERT routing")
        with self._lock:
            self._load()
            encoded = self._tokenizer(query, truncation=True, max_length=128, return_tensors="pt")
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                probabilities = self._torch.softmax(self._model(**encoded).logits, dim=-1)[0].cpu().tolist()
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {"intent": SUPPORT_LABELS[winner], "confidence": probabilities[winner],
                "probabilities": dict(zip(SUPPORT_LABELS, probabilities)), "artifact_verified": True,
                "dataset_sha256": self._report["dataset_sha256"],
                "training_scope": self._report["training_scope"]}

    def status(self) -> dict[str, Any]:
        path = self.model_path / "training_report.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return {"component": "BERT", "loaded": self._model is not None,
                "checkpoint_exists": (self.model_path / "config.json").is_file(),
                "task": report.get("task"), "labels": report.get("labels", []),
                "training_scope": report.get("training_scope"),
                "evaluation_scope": report.get("evaluation_scope"),
                "dataset_sha256": report.get("dataset_sha256"),
                "checkpoint_sha256": report.get("checkpoint_sha256")}

    def release_model(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._torch = None
            _collect_model_memory()
