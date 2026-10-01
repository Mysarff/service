"""Read-only probes: distinguish configured paths from real working services.

This command does not create schemas, collections, keys or test records. Optional
model/OCR probes perform inference; --probe-llm makes one billable API request.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKAGES = (
    "torch", "transformers", "FlagEmbedding", "pymilvus", "fastapi", "uvicorn",
    "langchain", "langchain-core", "langchain-openai", "langchain-community",
    "langchain-text-splitters", "PyMySQL", "SQLAlchemy", "redis",
    "rapidocr-onnxruntime", "onnxruntime", "python-multipart", "python-dotenv",
)
SECRETS: list[str] = []


def safe_error(exc: BaseException) -> str:
    message = str(exc)
    for secret in SECRETS:
        if secret:
            message = message.replace(secret, "[redacted]")
    for key, value in os.environ.items():
        if value and any(x in key.upper() for x in ("PASSWORD", "SECRET", "TOKEN", "API_KEY")):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?:https?|redis|mysql(?:\+pymysql)?)://[^\s\"'<>]+", "[endpoint]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message)
    return f"{type(exc).__name__}: {message[:400]}"


def check(callable_) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = callable_()
    except Exception as exc:
        result = {"status": "failed", "error": safe_error(exc)}
    result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    return result


def value(settings, name, default=None):
    return getattr(settings, name, default)


def load_settings():
    # The project owns configuration parsing; never read the education config.ini.
    from cloudcare.settings import Settings
    factory = getattr(Settings, "from_env", None)
    return factory() if factory else Settings()


def packages():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return {"status": "installed" if all(result.values()) else "missing_dependencies", "versions": result}


def hardware():
    import torch
    result = {"status": "checked", "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "cuda_build": torch.version.cuda, "cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    try:
        import psutil
        result["ram_total_bytes"] = psutil.virtual_memory().total
        result["ram_available_bytes"] = psutil.virtual_memory().available
    except ImportError:
        pass
    return result


def docker():
    executable = shutil.which("docker")
    if not executable and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker/Docker/resources/bin/docker.exe"
        executable = str(candidate) if candidate.is_file() else None
    if not executable:
        return {"status": "unavailable", "reason": "Docker CLI is not installed or discoverable."}
    proc = subprocess.run([executable, "info", "--format", "{{json .ServerVersion}}"],
                          capture_output=True, text=True, timeout=12)
    if proc.returncode:
        return {"status": "unavailable", "reason": "Docker engine could not be queried; start Docker Desktop."}
    status = subprocess.run([executable, "compose", "--project-name", "cloudcare-v2", "--file", str(ROOT / "compose.yaml"),
                             "ps", "--format", "json"], cwd=ROOT, capture_output=True, text=True, timeout=12)
    services = []
    if status.returncode == 0:
        payload = status.stdout.strip()
        entries = json.loads(payload) if payload.startswith("[") else [json.loads(line) for line in payload.splitlines() if line.strip()]
        for row in entries:
            # Only selected metadata; never return container environments.
            services.append({"service": row.get("Service"), "name": row.get("Name"), "image": row.get("Image"),
                             "state": row.get("State"), "health": row.get("Health")})
    return {"status": "connected", "server_version": json.loads(proc.stdout.strip()), "cloudcare_services": services,
            "all_five_services_healthy": len(services) == 5 and all(row["health"] == "healthy" for row in services)}


def corpus():
    path = ROOT / "data" / "knowledge.jsonl"
    if not path.is_file():
        return {"status": "missing", "knowledge_units": 0}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"status": "available", "knowledge_units": len(rows),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "unique_ids": len({x["id"] for x in rows}), "source_documents": len({x["source"] for x in rows}),
            "knowledge_topics": len({x["category"] for x in rows}),
            "synthetic": bool(rows) and all(x.get("synthetic") is True for x in rows)}


def mysql(settings):
    import pymysql
    database = value(settings, "mysql_database", "cloudcare_support")
    if not database.startswith("cloudcare_"):
        raise ValueError("Doctor only inspects a CloudCare database")
    connection = pymysql.connect(host=value(settings, "mysql_host", "127.0.0.1"),
                                 port=int(value(settings, "mysql_port", 3308)),
                                 user=value(settings, "mysql_user", "cloudcare"),
                                 password=value(settings, "mysql_password", ""), database=database,
                                 charset="utf8mb4", connect_timeout=4, read_timeout=5, write_timeout=5)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            server = cursor.fetchone()[0]
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=%s", (database,))
            tables = [x[0] for x in cursor.fetchall()]
            counts = {}
            for name in tables:
                if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                    continue
                cursor.execute(f"SELECT COUNT(*) FROM `{name}`")
                counts[name] = int(cursor.fetchone()[0])
            required = {"documents", "knowledge_chunks", "knowledge_parents", "faq", "sessions", "messages", "tickets"}
            missing = sorted(required.difference(tables))
            populated = not missing and counts.get("knowledge_chunks", 0) > 0
            return {"status": "connected_populated" if populated else "connected_incomplete_schema", "server_version": server,
                    "database": database, "table_rows": counts, "missing_tables": missing,
                    "knowledge_units": counts.get("knowledge_chunks", 0), "query_verified": True}
    finally:
        connection.close()


def redis(settings):
    import redis as redis_module
    client = redis_module.Redis.from_url(value(settings, "redis_url", "redis://127.0.0.1:6381/0"),
                                         socket_connect_timeout=4, socket_timeout=5)
    try:
        if not client.ping():
            raise RuntimeError("Redis PING did not return success")
        prefix = value(settings, "redis_prefix", "cloudcare:v2:")
        if not prefix.startswith("cloudcare:"):
            raise ValueError("Doctor only counts CloudCare key prefixes")
        count = sum(1 for _ in client.scan_iter(match=prefix + "*", count=100))
        return {"status": "connected", "ping_verified": True, "cloudcare_key_count": count,
                "server_version": client.info("server").get("redis_version"), "key_prefix": prefix}
    finally:
        client.close()


def milvus(settings):
    from pymilvus import MilvusClient, __version__
    collection = value(settings, "milvus_collection", "cloudcare_support_v2")
    if not collection.startswith("cloudcare_"):
        raise ValueError("Doctor only inspects CloudCare collections")
    client = MilvusClient(uri=value(settings, "milvus_uri", "http://127.0.0.1:19531"),
                          token=value(settings, "milvus_token", ""),
                          db_name=value(settings, "milvus_database", "default"), timeout=5)
    try:
        server = client.get_server_version(timeout=5)
        exists = client.has_collection(collection, timeout=5)
        result = {"status": "connected_missing_collection", "sdk_version": __version__, "server_version": server,
                  "recommended_sdk_series_match": __version__.split(".")[:2] == server.lstrip("v").split(".")[:2],
                  "collection": collection, "collection_exists": exists, "knowledge_units": 0}
        if exists:
            rows = client.query(collection, filter="", output_fields=["count(*)"],
                                consistency_level="Strong", timeout=10)
            count = int(rows[0]["count(*)"])
            result.update(status="connected_populated" if count else "connected_empty_collection",
                          knowledge_units=count, count_query_verified=True)
            samples = client.query(collection, filter="",
                                   output_fields=["id", "text", "content_hash", "dense_vector", "sparse_vector"],
                                   limit=8, consistency_level="Strong", timeout=10)
            result["vector_samples"] = [{"id": row["id"], "dense_dimension": len(row["dense_vector"]),
                                         "sparse_nonzero": len(row["sparse_vector"]),
                                         "content_hash_matches": hashlib.sha256(row["text"].encode("utf-8")).hexdigest() == row["content_hash"]}
                                        for row in samples]
            result["sample_valid"] = bool(samples) and all(row["dense_dimension"] == 1024 and row["sparse_nonzero"] > 0
                                                           and row["content_hash_matches"] for row in result["vector_samples"])
        return result
    finally:
        client.close()


def model_files(path):
    directory = Path(path)
    if not directory.is_dir():
        return {"status": "missing", "path": str(directory)}
    weights = list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
    required = [directory / "config.json", directory / "tokenizer_config.json"]
    complete = bool(weights) and all(x.is_file() for x in required)
    return {"status": "files_present_unverified" if complete else "incomplete", "path": str(directory),
            "weight_bytes": sum(x.stat().st_size for x in weights), "inference_verified": False}


def infer_model(kind, settings):
    import torch
    from cloudcare.neural import BGEM3Encoder, BGEReranker, SupportBertRouter
    device = value(settings, "device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if kind == "embedding":
        model = BGEM3Encoder(settings, query_cache_size=0)
        try:
            vectors = model.encode_queries(["CloudCare 客服账号登录失败如何处理"])
        finally:
            model.release_model()
        shape = [len(vectors), len(vectors[0].dense)]
        response = {"status": "inference_verified", "device": device, "dense_shape": shape,
                    "sparse_nonzero_terms": len(vectors[0].sparse), "inference_verified": True,
                    "adapter": "cloudcare.neural.BGEM3Encoder"}
        if shape != [1, 1024] or not vectors[0].sparse:
            raise RuntimeError("BGE-M3 did not return expected dense/sparse vectors")
    elif kind == "reranker":
        model = BGEReranker(settings)
        try:
            ranked = model.rerank("登录失败如何处理", [{"id": "fixture", "content": "请先确认账号状态并检查认证日志"}], k=1)
        finally:
            model.release_model()
        scores = [row["rerank_score"] for row in ranked]
        response = {"status": "inference_verified", "device": device, "scores": scores,
                    "inference_verified": True, "adapter": "cloudcare.neural.BGEReranker",
                    "note": "Pretrained BGE cross-encoder logits; no retrieval-quality claim."}
    else:
        model = SupportBertRouter(settings)
        try:
            prediction = model.predict("账号登录失败如何处理")
        finally:
            model.release_model()
        response = {"status": "inference_verified", "device": device, "labels": list(prediction["probabilities"]),
                    "predicted_intent": prediction["intent"], "artifact_verified": prediction["artifact_verified"],
                    "inference_verified": True, "adapter": "cloudcare.neural.SupportBertRouter",
                    "note": "Inference verifies execution, not customer-service classification accuracy."}
    del model
    return response


def ocr_probe():
    from rapidocr_onnxruntime import RapidOCR
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    image = Image.new("RGB", (800, 160), "white")
    fonts = [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    font_path = next((x for x in fonts if x.is_file()), None)
    font = ImageFont.truetype(str(font_path), 42) if font_path else ImageFont.load_default(size=42)
    ImageDraw.Draw(image).text((30, 50), "CloudCare Support 2026", fill="black", font=font)
    engine = RapidOCR()
    result, _ = engine(np.array(image))
    recognized = " ".join(x[1] for x in (result or []))
    if "cloudcare" not in recognized.lower().replace(" ", ""):
        raise RuntimeError("OCR fixture text was not recognized")
    return {"status": "inference_verified", "recognized_fixture": recognized, "lines": len(result),
            "inference_verified": True, "note": "Generated text fixture; no customer document is sent anywhere."}


def llm(settings):
    from langchain_openai import ChatOpenAI
    api_key = value(settings, "llm_api_key", "")
    model_name = value(settings, "llm_model", "qwen-plus")
    endpoint = value(settings, "llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        return {"status": "missing_api_key", "request_made": False}
    proxy = value(settings, "llm_proxy", "")
    model = ChatOpenAI(model=model_name, api_key=api_key, base_url=endpoint, timeout=20, max_retries=0,
                       max_tokens=16, openai_proxy=proxy or None,
                       extra_body={"enable_thinking": False})
    response = model.invoke("这是连接测试，请仅回答 OK。")
    if not response.content:
        raise RuntimeError("Model API returned empty content")
    return {"status": "request_verified", "model": model_name, "request_made": True, "nonempty_response": True,
            "note": "One connectivity request; no RAG answer-quality claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-models", action="store_true", help="Load local BGE-M3, reranker and support BERT one at a time")
    parser.add_argument("--probe-ocr", action="store_true", help="Run RapidOCR on a generated text fixture")
    parser.add_argument("--probe-llm", action="store_true", help="Make one billable request to the configured model endpoint")
    parser.add_argument("--services-only", action="store_true", help="Exit status covers real stores and matching corpus counts; does not assert model inference")
    parser.add_argument("--output", type=Path, help="Save sanitized JSON evidence")
    args = parser.parse_args()
    report = {"schema_version": 2, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version.split()[0], "executable": sys.executable,
              "checks": {"dependencies": check(packages), "hardware": check(hardware), "docker": check(docker),
                         "corpus": check(corpus)}}
    try:
        settings = load_settings()
    except Exception as exc:
        report["checks"]["configuration"] = {"status": "failed", "error": safe_error(exc)}
    else:
        SECRETS.extend(str(value(settings, x, "")) for x in ("mysql_password", "milvus_token", "llm_api_key", "redis_url"))
        for name, probe in (("mysql", mysql), ("redis", redis), ("milvus", milvus)):
            report["checks"][name] = check(lambda p=probe: p(settings))
        for name in ("embedding", "reranker", "bert"):
            report["checks"][name] = check(lambda n=name: infer_model(n, settings)) if args.probe_models else \
                model_files(value(settings, f"{name}_model_path", ROOT / "runtime" / "models" / name))
        report["checks"]["ocr"] = check(ocr_probe) if args.probe_ocr else {"status": "not_probed", "inference_verified": False}
        report["checks"]["llm"] = check(lambda: llm(settings)) if args.probe_llm else \
            {"status": "configured_unverified" if value(settings, "llm_api_key", "") else "missing_api_key", 
             "request_made": False, "note": "Use --probe-llm only when the API call is authorized."}
    # An installed package or a model file path is never an end-to-end success.
    report["full_stack_verified"] = report["checks"]["dependencies"]["status"] == "installed" and all(report["checks"].get(x, {}).get("status") in
        {"connected", "connected_populated", "inference_verified", "request_verified"}
        for x in ("mysql", "redis", "milvus", "embedding", "reranker", "bert", "ocr", "llm"))
    report["services_verified"] = all(report["checks"].get(name, {}).get("status") in {"connected", "connected_populated"}
                                       for name in ("mysql", "redis", "milvus")) and report["checks"].get("milvus", {}).get("sample_valid") is True
    report["knowledge_count_consistent"] = (report["checks"].get("corpus", {}).get("knowledge_units") ==
                                           report["checks"].get("mysql", {}).get("knowledge_units") ==
                                           report["checks"].get("milvus", {}).get("knowledge_units")) if report["services_verified"] else False
    report["probe_scope"] = {"local_models": args.probe_models, "ocr": args.probe_ocr, "external_llm": args.probe_llm}
    report["note"] = "Read-only component probes; does not replace retrieval, answer-quality or load evaluation."
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    accepted = report["full_stack_verified"]
    if args.services_only:
        accepted = report["services_verified"] and report["knowledge_count_consistent"]
        if args.probe_llm:
            accepted = accepted and report["checks"]["llm"].get("status") == "request_verified"
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
