"""Actual neural retrieval and routing audit on the frozen synthetic 88 cases.

No provider call is made unless --llm-probe is explicitly supplied. Retrieval
source-ID hits do not measure answer correctness. Negative cases use the real
SupportPipeline.answer(force_extract=True), including BERT and evidence rules.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows: list[dict]) -> dict:
    size = len(rows)
    top1 = sum(row["rank"] == 1 for row in rows)
    hit5 = sum(1 <= row["rank"] <= 5 for row in rows)
    return {"n": size, "top1": top1, "hit_at_1": top1 / size if size else None,
            "hit5": hit5, "hit_at_5": hit5 / size if size else None,
            "mrr_at_5": sum(1 / row["rank"] if 1 <= row["rank"] <= 5 else 0 for row in rows) / size if size else None}


def latency(rows: list[dict]) -> dict:
    timings = sorted(row["elapsed_ms"] for row in rows)
    if not timings:
        return {"n": 0}
    return {"n": len(timings), "mean_ms": statistics.mean(timings),
            "p50_ms": timings[max(0, math.ceil(len(timings) * .5) - 1)],
            "p95_ms": timings[max(0, math.ceil(len(timings) * .95) - 1)], "max_ms": timings[-1]}


def observation(query, expected, hits, elapsed_ms, trace=None) -> dict:
    ids = [row["id"] for row in hits[:5]]
    rank = min((ids.index(identifier) + 1 for identifier in expected if identifier in ids), default=0)
    scores = [{key: row[key] for key in ("id", "source", "parent_id", "version", "category",
               "content_hash", "score", "fusion_score", "rerank_score", "retrieval_rank", "branches") if key in row}
              for row in hits[:5]]
    return {"query": query, "expected": expected, "ids": ids, "rank": rank,
            "elapsed_ms": elapsed_ms, "hits": scores, "trace": trace or {}}


def save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/fullstack_metrics.json")
    parser.add_argument("--llm-probe", action="store_true", help="Call configured Qwen on 8 fixed positive cases; separate report")
    parser.add_argument("--route-probe", action="store_true", help="Audit actual FAQ/RAG answer routing on all selected positive cases without Qwen")
    parser.add_argument("--positive-limit", type=int, help="Smoke audit only; result explicitly records reduced sample count")
    parser.add_argument("--negative-limit", type=int, help="Smoke audit only; result explicitly records reduced sample count")
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    from cloudcare.pipeline import SupportPipeline
    from cloudcare.settings import Settings
    import engine
    settings = Settings.load()
    groups = {name: json.loads((ROOT / "evaluation" / name).read_text(encoding="utf-8"))
              for name in ("cases.json", "extended_cases.json")}
    positives = [(name, query, expected) for name, group in groups.items() for query, expected in group["positive"]]
    negatives = [(name, query) for name, group in groups.items() for query in group["negative"]]
    if args.positive_limit is not None:
        positives = positives[:args.positive_limit]
    if args.negative_limit is not None:
        negatives = negatives[:args.negative_limit]
    code_files = ["cloudcare/neural.py", "cloudcare/retrieval.py", "cloudcare/pipeline.py", "cloudcare/storage.py", "cloudcare/faq.py", "cloudcare/settings.py",
                  "cloudcare/llm.py", "engine.py", "scripts/evaluate_fullstack.py"]
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "scope": "synthetic development regression, actual Milvus/BGE cross-encoder, no external LLM",
              "labels_used_for_tuning": True, "blind_test": False,
              "metrics_meaning": "source-ID retrieval hits; neither answer factual accuracy nor production effectiveness",
              "host": {"os": platform.platform(), "python": platform.python_version()},
              "model_memory_policy": settings.model_memory_policy,
              "latency_scope": "end-to-end retrieval in one process, including any model loading/reloading, "
                               "vector cache, live database, reranking and parent expansion; not model-only inference",
              "corpus_sha256": sha256(ROOT / "data/knowledge.jsonl"),
              "evaluation_sha256": {name: sha256(ROOT / "evaluation" / name) for name in groups},
              "code_sha256": {name: sha256(ROOT / name) for name in code_files},
              "sample_counts": {"positive": len(positives), "negative": len(negatives),
                                "full_frozen_positive": 64, "full_frozen_negative": 24},
              "faq_bm25_threshold": settings.faq_bm25_threshold,
              "faq_score_kind": "BM25 raw ranking followed by Softmax over all eligible FAQ; not answer correctness probability",
              "retrieval": {}, "positive_routing": {"enabled": args.route_probe, "observations": []},
              "negative": {"observations": []}, "complete": False}
    pipeline = None
    stage = "construct"
    current_query = None
    rows_by_method = {"plain_bm25": [], "enhanced_bm25": [], "bge_milvus_bm25_rerank": []}

    def checkpoint() -> None:
        # Preserve completed observations if a later model or provider fails.
        # Partial results remain explicitly incomplete and are never compared
        # as if they covered all frozen cases.
        for method, observations in rows_by_method.items():
            report["retrieval"][method] = {**metrics(observations), "latency": latency(observations),
                "groups": {name: metrics([row for row in observations if row["group"] == name]) for name in groups},
                "observations": observations}
        report["last_stage"] = stage
        report["current_query"] = current_query
        report["wall_seconds"] = time.perf_counter() - started
        save(args.output, report)

    started = time.perf_counter()
    try:
        constructed = time.perf_counter()
        pipeline = SupportPipeline(settings)
        pipeline.start()
        # Existing production/demo caches must not hide current routing changes.
        pipeline._cache_config = hashlib.sha256((pipeline._cache_config + uuid.uuid4().hex).encode()).hexdigest()
        report["cold_start"] = {"construction_and_service_start_ms": (time.perf_counter() - constructed) * 1000}
        stage = "warmup_retrieval"
        warm_query = "账号忘记密码以后如何重置登录密码？"
        constructed = time.perf_counter()
        warm_hits, _ = pipeline.retrieve(warm_query)
        report["cold_start"].update({"first_neural_retrieval_ms": (time.perf_counter() - constructed) * 1000,
                                     "warmup_query": warm_query, "warmup_ids": [row["id"] for row in warm_hits]})
        constructed = time.perf_counter()
        stage = "warmup_bert"
        pipeline._predict_route(warm_query)
        report["cold_start"]["first_bert_inference_ms"] = (time.perf_counter() - constructed) * 1000
        report["cold_start"]["scope"] = "single process first-request diagnostics, loading included; " \
                                        "excluded from repeated-request latency statistics. Sequential policy " \
                                        "reloads weights between stages and that cost is included in every timed request."
        with tempfile.TemporaryDirectory() as folder, patch.object(engine, "UPLOADS", Path(folder) / "uploads"):
            baseline = engine.Engine()
            baseline.key = ""
            report["baseline_corpus_units"] = len(baseline.docs)
            pipeline_ids = {row["id"] for row in pipeline.bm25.rows}
            baseline_ids = {row["id"] for row in baseline.docs}
            if not baseline_ids.issubset(pipeline_ids):
                raise RuntimeError("The full-stack index does not contain the frozen source-ID corpus")
            report["fullstack_corpus_units"] = len(pipeline_ids)
            report["identical_corpus"] = baseline_ids == pipeline_ids
            report["additional_indexed_units"] = len(pipeline_ids - baseline_ids)
            shuffled = positives[:]
            random.Random(args.seed).shuffle(shuffled)
            for index, (group, query, expected) in enumerate(shuffled, 1):
                stage = "positive_retrieval"
                current_query = query
                for label, method in (("plain_bm25", "plain_bm25"), ("enhanced_bm25", "bm25")):
                    tick = time.perf_counter()
                    hits = baseline.search(query, method=method)
                    row = observation(query, expected, hits, (time.perf_counter() - tick) * 1000)
                    row["group"] = group
                    rows_by_method[label].append(row)
                tick = time.perf_counter()
                hits, trace = pipeline.retrieve(query)
                row = observation(query, expected, hits, (time.perf_counter() - tick) * 1000, trace)
                row["group"] = group
                rows_by_method["bge_milvus_bm25_rerank"].append(row)
                if index % 8 == 0:
                    checkpoint()
                    print(f"Actual neural retrieval: {index}/{len(shuffled)}", flush=True)
            for method, observations in rows_by_method.items():
                report["retrieval"][method] = {**metrics(observations), "latency": latency(observations),
                    "groups": {name: metrics([row for row in observations if row["group"] == name]) for name in groups},
                    "observations": observations}
        if args.route_probe:
            for index, (group, query, expected) in enumerate(positives, 1):
                stage = "positive_answer_route"
                current_query = query
                answer = pipeline.answer(query, force_extract=True)
                source_ids = [row["id"] for row in answer["sources"]]
                report["positive_routing"]["observations"].append({"query": query, "group": group,
                    "expected": expected, "mode": answer["mode"], "source_ids": source_ids,
                    "source_id_hit": bool(set(expected) & set(source_ids)),
                    "answer": answer["answer"], "elapsed_ms": answer["elapsed_ms"], "trace": answer["trace"]})
                if index % 8 == 0:
                    checkpoint()
                    print(f"Actual FAQ/RAG routing: {index}/{len(positives)}", flush=True)
            routed = report["positive_routing"]["observations"]
            faq_rows = [row for row in routed if row["mode"] == "faq"]
            report["positive_routing"].update({"n": len(routed),
                "source_id_hits": sum(row["source_id_hit"] for row in routed),
                "faq_direct_answers": len(faq_rows),
                "faq_source_id_hits": sum(row["source_id_hit"] for row in faq_rows),
                "cache_hits": sum(row["trace"]["cache_hit"] for row in routed),
                "latency": latency(routed),
                "meaning": "Source-ID coverage of actual answer routes; not answer factual accuracy or blind FAQ calibration"})
        for index, (group, query) in enumerate(negatives, 1):
            stage = "negative_answer"
            current_query = query
            tick = time.perf_counter()
            answer = pipeline.answer(query, force_extract=True)
            observation_row = {"query": query, "group": group, "mode": answer["mode"],
                               "rejected": answer["mode"] == "insufficient_evidence", "answer": answer["answer"],
                               "source_ids": [row["id"] for row in answer["sources"]],
                               "elapsed_ms": (time.perf_counter() - tick) * 1000, "trace": answer["trace"]}
            report["negative"]["observations"].append(observation_row)
            if index % 8 == 0:
                checkpoint()
                print(f"BERT and evidence-gate negatives: {index}/{len(negatives)}", flush=True)
        rows = report["negative"]["observations"]
        report["negative"].update({"n": len(rows), "rejected": sum(row["rejected"] for row in rows),
                                  "scope": "actual answer route with BERT/rules; no external model; rejection is not retrieval emptiness",
                                  "latency": latency(rows)})
        stage = "component_and_index_audit"
        current_query = None
        report["components"] = {"encoder": pipeline.vector.encoder.status(), "reranker": pipeline.reranker.status(),
                                "bert": pipeline.router.status(), "qwen": pipeline.llm.status()}
        report["milvus_audit"] = pipeline.vector.audit(expected_count=report["fullstack_corpus_units"])
        report["last_stage"] = "retrieval_and_negatives_complete"
        if args.llm_probe:
            report["qwen_probe_output"] = args.output.stem + "_qwen_probe.json"
            routed = report["positive_routing"]["observations"]
            rag_queries = {row["query"] for row in routed if row["mode"] == "retrieval_only"}
            probe_cases = ([case for case in positives if case[1] in rag_queries][:8]
                           if args.route_probe else positives[:8])
            probe = {"scope": "Up to 8 synthetic source-grounded Qwen API smoke questions; not RAGAS or answer-quality benchmark",
                     "configured": settings.llm_configured, "model": settings.llm_model,
                     "model_memory_policy": settings.model_memory_policy,
                     "latency_scope": report["latency_scope"], "observations": [], "complete": False,
                     "sample_count": len(probe_cases),
                     "selection": "First available RAG-bound positive cases from the no-Qwen route audit" if args.route_probe else "First 8 frozen positives; FAQ may intercept",
                     "cache_scope": "new evaluation run namespace; existing answer cache cannot mask provider calls"}
            if settings.llm_configured:
                original_cache_config = pipeline._cache_config
                pipeline._cache_config = hashlib.sha256((original_cache_config + uuid.uuid4().hex).encode()).hexdigest()
                try:
                    for index, (_, query, expected) in enumerate(probe_cases, 1):
                        stage = "qwen_probe"
                        current_query = query
                        before = pipeline.llm.status()
                        result = pipeline.answer(query, force_extract=False)
                        after = pipeline.llm.status()
                        probe["observations"].append({"query": query, "expected": expected, "mode": result["mode"],
                            "answer": result["answer"], "source_ids": [row["id"] for row in result["sources"]],
                            "source_id_hit": bool(set(expected) & {row["id"] for row in result["sources"]}),
                            "elapsed_ms": result["elapsed_ms"], "trace": result["trace"],
                            "provider_calls": after["calls"] - before["calls"],
                            "successful_provider_calls": after["successful_calls"] - before["successful_calls"]})
                        probe["grounded_llm_count"] = sum(row["mode"] == "grounded_llm" for row in probe["observations"])
                        probe["actual_provider_calls"] = sum(row["provider_calls"] for row in probe["observations"])
                        probe["successful_provider_calls"] = sum(row["successful_provider_calls"] for row in probe["observations"])
                        save(args.output.with_name(args.output.stem + "_qwen_probe.json"), probe)
                        print(f"Actual Qwen answer probe: {index}/{len(probe_cases)}", flush=True)
                    probe["complete"] = True
                finally:
                    pipeline._cache_config = original_cache_config
                    probe["provider"] = pipeline.llm.status()
                    save(args.output.with_name(args.output.stem + "_qwen_probe.json"), probe)
            else:
                probe["status"] = "not_configured; no external request made"
            probe["provider"] = pipeline.llm.status()
            save(args.output.with_name(args.output.stem + "_qwen_probe.json"), probe)
        report["complete"] = True
        stage = "complete"
        current_query = None
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["failed_stage"] = stage
        report["failed_query"] = current_query
        raise
    finally:
        checkpoint()
        if pipeline is not None:
            pipeline.close()
    summary = {key: report[key] for key in ("complete", "scope", "sample_counts", "cold_start", "wall_seconds")}
    summary["retrieval"] = {method: {key: value for key, value in result.items() if key not in {"observations", "groups"}}
                            for method, result in report["retrieval"].items()}
    summary["negative"] = {key: value for key, value in report["negative"].items() if key != "observations"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
