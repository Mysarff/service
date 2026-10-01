"""Opt-in real CloudCare workflow acceptance on uniquely owned temporary data.

Run after other GPU processes have finished. This performs real OCR, BGE-M3,
Milvus, BGE-Reranker, BERT, Qwen, MySQL and Redis operations. It creates no real
business ticket and deletes only the exact fixture IDs/cache keys it owns.
No model or service is loaded merely by importing this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_fixture(marker: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    candidates = [Path("C:/Windows/Fonts/arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("A TrueType font is required for the real OCR fixture")
    image = Image.new("RGB", (1900, 650), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 52)
    for number, line in enumerate(("CloudCare support acceptance " + marker,
        "Verify masked incident logs before handoff.",
        "Escalate to duty staff after approval.", "This is synthetic acceptance data.")):
        draw.text((60, 60 + number * 130), line, font=font, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def parse_marker(value: str) -> str:
    normalized = value.strip().upper()
    if not re.fullmatch(r"WF-[A-F0-9]{12}", normalized):
        raise argparse.ArgumentTypeError("Marker must have the form WF- followed by 12 hexadecimal characters")
    return normalized


def run(output: Path, marker: str | None = None) -> int:
    from cloudcare.pipeline import SupportPipeline
    from cloudcare.settings import Settings

    settings = Settings.load()
    explicit_marker = marker is not None
    marker = parse_marker(marker) if explicit_marker else "WF-" + uuid.uuid4().hex[:12].upper()
    category = "合成验收_" + marker
    report = {"schema_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real component workflow on newly generated synthetic acceptance fixtures; not production quality or a business KPI",
        "marker": marker, "explicit_reproduction_marker": explicit_marker,
        "category": category, "synthetic_fixture": True,
        "followup_scope": "Only the existing short-follow-up rule is checked using 然后呢？. Queries with at least three lexical terms, such as 下一步怎么核验？, are not guaranteed to inherit history by this rule.",
        "active_corpus_sha256": sha(ROOT / "data/knowledge.jsonl"), "stages": [],
        "answer_observations": [], "llm_invoke_observations": [], "complete": False}
    core_names = ["cloudcare/pipeline.py", "cloudcare/llm.py", "cloudcare/neural.py",
                  "cloudcare/retrieval.py", "cloudcare/storage.py", "cloudcare/documents.py",
                  "scripts/probe_fullstack_workflow.py"]
    before_sha = {name: sha(ROOT / name) for name in core_names}
    report["code_sha256"] = before_sha
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = None
    planned = []
    sessions = set()
    tickets = {}
    cache_keys = set()
    raw_answers = []

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    def stage(name, function, *, critical=True):
        entry = {"stage": name, "status": "running"}
        report["stages"].append(entry)
        save()
        started = time.perf_counter()
        try:
            result = function()
            entry.update(status="passed", result=result)
            return result
        except Exception as exc:
            entry.update(status="failed", error_type=type(exc).__name__)
            # Only our assertion messages are safe, predetermined fixture checks.
            if isinstance(exc, AssertionError):
                entry["check_failure"] = str(exc)
            if critical:
                raise
            return None
        finally:
            entry["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            save()

    def snapshot():
        return {"mysql": pipeline.sql.counts(), "redis": pipeline.redis.counts(),
                "milvus_rows": pipeline.vector.count(tenant_id="demo")}

    def source_summary(row):
        return {key: row.get(key) for key in ("id", "document_id", "parent_id", "source",
                                             "category", "content_hash", "rerank_score")}

    def answer_summary(answer):
        return {"mode": answer["mode"], "answer": answer["answer"],
                "sources": [source_summary(row) for row in answer["sources"]],
                "cache_hit": answer["trace"]["cache_hit"], "session_id": answer["session_id"],
                "bert": answer["trace"].get("bert"), "rewrite": answer["trace"].get("rewrite"),
                "original_query": answer["trace"].get("original_query"),
                "contextual_query": answer["trace"].get("contextual_query"),
                "generation_error_type": answer["trace"].get("generation_error_type")}

    def observe_answer(label, answer):
        # Capture the real outcome before an assertion can reject its mode,
        # citation or history. Failed acceptance never erases model evidence.
        report["answer_observations"].append({"probe": label, **answer_summary(answer)})
        save()

    try:
        pipeline = SupportPipeline(settings)
        pipeline.start()
        report["before"] = stage("dedicated_services", snapshot)
        # Transparent instrumentation records ownership, then calls the actual
        # Redis/SQL methods. It replaces no database, model or response.
        original_cache = pipeline.redis.set_cached_answer
        original_record = pipeline.sql.record_message
        original_invoke = pipeline.llm._invoke

        def record_cache(key, payload, ttl=None):
            actual_key = pipeline.redis._key("answer", key)
            if not pipeline.redis.client.exists(actual_key):
                cache_keys.add(actual_key)
            return original_cache(key, payload, ttl)

        def record_message(session_id, role, content, metadata=None):
            if session_id not in sessions:
                if pipeline.sql.get_session(session_id):
                    raise ValueError("A generated fixture session collides with an existing session")
                sessions.add(session_id)
            return original_record(session_id, role, content, metadata)

        def record_invoke(prompt, values):
            query_value = values.get("query", "")
            if not isinstance(query_value, str) or marker.lower() not in query_value.lower():
                return original_invoke(prompt, values)
            observation = {"purpose": "generation" if "evidence" in values else "rewrite",
                "status": "running", "synthetic_fixture_only": True,
                "calls_before": pipeline.llm.calls,
                "successful_calls_before": pipeline.llm.successful_calls}
            report["llm_invoke_observations"].append(observation)
            save()
            started = time.perf_counter()
            try:
                response = original_invoke(prompt, values)
                observation.update(status="returned", raw_response=response)
                return response
            except Exception as exc:
                # No exception body, settings, provider URL or credential is saved.
                observation.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                observation.update(calls_after=pipeline.llm.calls,
                    successful_calls_after=pipeline.llm.successful_calls,
                    elapsed_ms=round((time.perf_counter()-started)*1000, 3))
                observation["actual_call_delta"] = observation["calls_after"] - observation["calls_before"]
                save()

        pipeline.redis.set_cached_answer = record_cache
        pipeline.sql.record_message = record_message
        pipeline.llm._invoke = record_invoke
        text = (f"CloudCare合成客服工单升级验收，编号{marker}。\n"
                "这是随机生成的验收资料，不是真实企业政策。\n"
                "客服处理故障工单时，先核验脱敏故障日志和影响范围，记录工单编号。\n"
                "确认需要升级后，将已核验的故障信息提交值班人员审批，再转交人工跟进。\n"
                "验收要求：工单中保留脱敏日志、审批记录和交接结果；不得承诺自动退款或实际执行变更。\n").encode("utf-8")
        fixtures = [("workflow-" + marker + ".txt", text),
                    ("workflow-" + marker + ".png", image_fixture(marker))]
        with tempfile.TemporaryDirectory(prefix="cloudcare-workflow-fixtures-") as directory:
            for filename, raw in fixtures:
                path = Path(directory) / filename
                path.write_bytes(raw)
                processed = pipeline.documents.ingest_path(path, category=category, source_name=filename)
                if pipeline.sql.document_by_hash(processed.sha256):
                    raise ValueError("A generated fixture collides with existing source data")
                planned.append(processed)
                extension = path.suffix[1:]
                if extension == "png":
                    extracted = " ".join(row.page_content for row in processed.documents).lower()
                    assert "masked" in extracted and "handoff" in extracted, "Actual OCR did not recover fixture instructions"
                result = stage("import_" + extension,
                    lambda filename=filename, raw=raw: pipeline.ingest(filename, raw, category))
                assert not result["duplicate"] and result["document_id"] == processed.id, "Fresh import must publish its parsed document ID"
            def publication_counts():
                counts = snapshot()
                expected_units = sum(len(item.chunks) for item in planned)
                assert counts["mysql"]["documents"] == report["before"]["mysql"]["documents"] + 2, "Two sources must be published"
                assert counts["mysql"]["chunks"] == report["before"]["mysql"]["chunks"] + expected_units, "SQL import count mismatch"
                assert counts["milvus_rows"] == report["before"]["milvus_rows"] + expected_units, "Milvus import count mismatch"
                return counts
            after_import = stage("publication_counts", publication_counts)
            report["after_import"] = after_import

            def duplicates():
                results = [pipeline.ingest(filename, raw, category) for filename, raw in fixtures]
                assert all(row["duplicate"] for row in results), "Identical-byte imports must be duplicates"
                assert snapshot() == after_import, "Repeated imports must not increase stored counts"
                return results
            stage("duplicate_hashes", duplicates)

        query = f"编号{marker}的客服故障工单应该如何升级并核验交接？"
        owned_chunks = {row["id"] for item in planned for row in item.chunks}

        def retrieval():
            hits, trace = pipeline.retrieve(query, category)
            assert hits and all(row["id"] in owned_chunks for row in hits), "Category-filtered retrieval must return owned fixture sources"
            for hit in hits:
                parent = pipeline.sql.get_parent(hit["parent_id"])
                assert parent and parent["document_id"] == hit["document_id"], "SQL parent provenance mismatch"
                assert hit["parent_content"] == parent["content"], "Parent evidence must come from actual SQL"
            return {"sources": [source_summary(row) for row in hits], "trace": trace}
        stage("milvus_rerank_sql_parents", retrieval)

        def qwen():
            answer = pipeline.answer(query, category, force_extract=False)
            raw_answers.append(answer)
            observe_answer("qwen_initial", answer)
            assert answer["mode"] == "grounded_llm", "Qwen must return grounded_llm; fallback/extraction is not model acceptance"
            assert answer["sources"] and all(row["id"] in owned_chunks for row in answer["sources"]), "Qwen citations must refer to imported fixtures"
            assert answer["trace"].get("rewrite", {}).get("method") == "qwen_langchain", "Actual Qwen query rewriting must succeed"
            return answer_summary(answer)
        stage("qwen_grounded_answer", qwen, critical=False)

        def cache():
            first = raw_answers[0] if raw_answers and raw_answers[0]["mode"] == "grounded_llm" else pipeline.answer(query, category, force_extract=True)
            observe_answer("cache_setup", first)
            mode = first["mode"]
            calls = pipeline.llm.calls
            second = pipeline.answer(query, category, force_extract=mode != "grounded_llm")
            raw_answers.append(second)
            observe_answer("cache_repeat", second)
            assert second["trace"]["cache_hit"], "Identical initial query must hit the real Redis answer cache"
            assert second["answer"] == first["answer"], "Cached answer must retain grounded content"
            assert pipeline.llm.calls == calls, "A cache hit must avoid a new provider request"
            return answer_summary(second)
        stage("redis_answer_cache", cache, critical=False)

        def persistence():
            first = raw_answers[0]
            identifier = first["session_id"]
            sql_history = pipeline.sql.get_session(identifier)
            redis_history = pipeline.redis.get_session(identifier)
            assert len(sql_history) == 2 and redis_history, "Actual SQL and Redis must both persist the first turn"
            pipeline.redis.clear_session(identifier)
            assert pipeline.redis.get_session(identifier) is None, "Only this owned session cache must be cleared"
            followup = pipeline.answer("然后呢？", category, session_id=identifier, force_extract=True)
            raw_answers.append(followup)
            observe_answer("short_followup_after_redis_miss", followup)
            assert len(pipeline.sql.get_session(identifier)) == 4, "SQL must persist the follow-up after Redis cache miss"
            assert pipeline.redis.get_session(identifier), "SQL history fallback must restore Redis session history"
            assert marker.lower() in followup["trace"]["contextual_query"].lower(), "Follow-up query must preserve original incident identity"
            return {"sql_messages": 4, "redis_restored": True, "followup": answer_summary(followup)}
        stage("sql_session_and_redis_fallback", persistence, critical=False)

        def ticket():
            summary = "合成工作流验收工单 " + marker
            result = pipeline.create_ticket(summary)
            identifier = result["ticket"]["id"]
            tickets[identifier] = summary
            row = pipeline.sql.get_ticket(identifier)
            assert row and row["question"] == summary, "Local support ticket must actually persist in MySQL"
            assert result["ticket"]["local_only"], "The acceptance ticket must be explicitly local-only"
            return {"id": identifier, "summary": summary, "local_only": True}
        stage("local_ticket_persistence", ticket, critical=False)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        if pipeline is not None:
            def cleanup():
                identifiers = {item.id for item in planned}
                chunk_ids = {row["id"] for item in planned for row in item.chunks}
                parent_ids = {row["id"] for item in planned for row in item.parents}
                if chunk_ids:
                    pipeline.vector.delete_documents(sorted(chunk_ids), tenant_id="demo")
                with pipeline.sql.engine.begin() as connection:
                    if sessions:
                        connection.execute(pipeline.sql.messages.delete().where(pipeline.sql.messages.c.session_id.in_(sessions)))
                        connection.execute(pipeline.sql.sessions.delete().where(pipeline.sql.sessions.c.id.in_(sessions)))
                    for identifier, summary in tickets.items():
                        connection.execute(pipeline.sql.tickets.delete().where(
                            (pipeline.sql.tickets.c.id == identifier) & (pipeline.sql.tickets.c.question == summary)))
                    if chunk_ids:
                        connection.execute(pipeline.sql.chunks.delete().where(
                            pipeline.sql.chunks.c.id.in_(chunk_ids) & pipeline.sql.chunks.c.document_id.in_(identifiers)))
                    if parent_ids:
                        connection.execute(pipeline.sql.parents.delete().where(
                            pipeline.sql.parents.c.id.in_(parent_ids) & pipeline.sql.parents.c.document_id.in_(identifiers)))
                    if identifiers:
                        connection.execute(pipeline.sql.documents.delete().where(
                            pipeline.sql.documents.c.id.in_(identifiers) & (pipeline.sql.documents.c.category == category)))
                for identifier in sessions:
                    pipeline.redis.clear_session(identifier)
                if cache_keys:
                    pipeline.redis.client.delete(*sorted(cache_keys))
                pipeline.refresh()
                after = snapshot()
                report["after_cleanup"] = after
                if "before" in report:
                    for key in ("documents", "parents", "chunks", "faq", "sessions", "messages", "tickets"):
                        assert after["mysql"][key] == report["before"]["mysql"][key], "SQL fixture cleanup must restore pre-run counts: " + key
                    assert after["milvus_rows"] == report["before"]["milvus_rows"], "Milvus cleanup must restore pre-run unit count"
                    assert after["redis"] == report["before"]["redis"], "Live FAQ index must remain unchanged"
                return {"owned_documents": sorted(identifiers), "owned_chunks": sorted(chunk_ids),
                        "owned_sessions": sorted(sessions), "owned_tickets": sorted(tickets),
                        "owned_answer_cache_keys": len(cache_keys), "counts_restored": True}
            stage("precise_fixture_cleanup", cleanup, critical=False)
            pipeline.close()
        after_sha = {name: sha(ROOT / name) for name in core_names}
        report["code_sha256_after"] = after_sha
        report["code_stable"] = before_sha == after_sha
        mandatory = {"dedicated_services", "import_txt", "import_png", "publication_counts", "duplicate_hashes",
            "milvus_rerank_sql_parents", "qwen_grounded_answer", "redis_answer_cache",
            "sql_session_and_redis_fallback", "local_ticket_persistence", "precise_fixture_cleanup"}
        passed = {row["stage"] for row in report["stages"] if row["status"] == "passed"}
        report["complete"] = mandatory <= passed and report["code_stable"]
        save()
    print(json.dumps({"report": str(output), "complete": report["complete"],
        "stages": {row["stage"]: row["status"] for row in report["stages"]}}, ensure_ascii=False))
    return 0 if report["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/fullstack_workflow.json")
    parser.add_argument("--marker", type=parse_marker,
                        help="Reuse a cleaned synthetic marker, e.g. WF-E68D8BA8D833, to reproduce a failure")
    args = parser.parse_args()
    return run(args.output.resolve(), marker=args.marker)


if __name__ == "__main__":
    raise SystemExit(main())
