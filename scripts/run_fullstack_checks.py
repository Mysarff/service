"""Run the complete checked-in suite and preserve auditable per-test evidence.

Dedicated MySQL/Redis integration is mandatory here. This runner does not load
the large BGE/BERT checkpoints or call an external LLM: their real acceptance
results belong to fullstack_metrics.json/fullstack_index.json. Baseline tests
remain baseline tests, even though they share this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCOPES = {
    "test_system": "historical_baseline_http",
    "test_review_regressions": "historical_baseline_regressions",
    "test_retrieval_audit": "historical_baseline_corpus_and_metrics",
    "test_frontend_history": "current_frontend_javascript_against_baseline_http",
    "test_documents_fullstack": "real_langchain_office_pdf_and_cpu_rapidocr",
    "test_doctor_release": "doctor_model_resource_lifecycle_with_injected_adapters",
    "test_api_fullstack": "fastapi_transport_with_injected_pipeline",
    "test_neural_initialize": "initialization_and_training_data_fixture_contracts",
    "test_neural_contracts": "injected_neural_and_milvus_clients_with_small_cpu_tensor",
    "test_pipeline_contracts": "pipeline_and_qwen_contracts_with_injected_backends",
    "test_qwen_contracts": "qwen_prompt_and_response_contracts_with_injected_call",
}


def scope_for(name: str) -> str:
    module = name.split(".")[0]
    if module == "test_storage_fullstack":
        return ("real_dedicated_mysql_and_redis" if ".StorageIntegrationTests." in name
                else "real_dedicated_mysql" if ".MySQLIntegrationTests." in name
                else "storage_namespace_unit_contract")
    return SCOPES.get(module, "unclassified_test_requires_review")


def select_node() -> dict:
    candidates = [os.environ.get("CLOUDCARE_NODE"), str(Path.home() / ".cache" /
        "codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"),
        shutil.which("node")]
    inspected = []
    for candidate in dict.fromkeys(value for value in candidates if value):
        try:
            version = subprocess.run([candidate, "--version"], capture_output=True,
                                     text=True, check=True, timeout=10).stdout.strip()
            major = int(re.fullmatch(r"v(\d+)\.\d+\.\d+", version).group(1))
            inspected.append({"path": candidate, "version": version})
            if major >= 18:
                os.environ["PATH"] = str(Path(candidate).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
                return {"ready": True, "path": str(Path(candidate).resolve()), "version": version}
        except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
            inspected.append({"path": candidate, "version": "unavailable"})
    return {"ready": False, "reason": "Node 18+ is required for the frontend fetch check", "inspected": inspected}


def hashes() -> dict[str, str]:
    paths = [*ROOT.glob("cloudcare/*.py"), *ROOT.glob("tests/test*.py"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("evaluation/*.py")]
    paths.extend(ROOT / name for name in ("app.py", "baseline_app.py", "engine.py", "static/index.html",
        "static/baseline.html", "requirements-full.txt", "requirements-fullstack.txt",
        "compose.yaml", "docker-compose.yml"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths)) if path.is_file()}


def storage_snapshot() -> dict:
    """Read actual dedicated-service counts without exposing connection secrets."""
    from cloudcare.settings import Settings
    from cloudcare.storage import MySQLStore, RedisStore
    settings = Settings.load()
    stores = {"mysql": MySQLStore(settings), "redis": RedisStore(settings)}
    snapshot = {}
    for name, store in stores.items():
        try:
            store.health()
            snapshot[name] = {"ready": True, "counts": store.counts()}
        except Exception as exc:
            snapshot[name] = {"ready": False, "error_type": type(exc).__name__}
        finally:
            store.close()
    return snapshot


class AuditedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: dict[str, dict] = {}
        self.started: dict[str, float] = {}

    def _entry(self, test):
        name = test.id()
        return self.records.setdefault(name, {"name": name, "status": "running", "scope": scope_for(name)})

    def startTest(self, test):
        self.started[test.id()] = time.perf_counter()
        self._entry(test)
        super().startTest(test)

    def stopTest(self, test):
        self._entry(test)["elapsed_seconds"] = round(time.perf_counter() - self.started[test.id()], 6)
        super().stopTest(test)

    def addSuccess(self, test):
        self._entry(test)["status"] = "passed"
        super().addSuccess(test)

    def addFailure(self, test, err):
        self._entry(test).update(status="failed", error_type=err[0].__name__)
        super().addFailure(test, err)

    def addError(self, test, err):
        self._entry(test).update(status="error", error_type=err[0].__name__)
        super().addError(test, err)

    def addSkip(self, test, reason):
        self._entry(test).update(status="skipped", reason=reason)
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self._entry(test).update(status="expected_failure", error_type=err[0].__name__)
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self._entry(test)["status"] = "unexpected_success"
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self._entry(test).update(status="failed" if issubclass(err[0], test.failureException) else "error",
                                     error_type=err[0].__name__)
            self._entry(test).setdefault("failed_subtests", []).append(subtest.id())
        super().addSubTest(test, subtest, err)


def run_worker(pattern: str, output: Path) -> int:
    """Keep a native crash in one module from erasing all other evidence."""
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern)
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2, resultclass=AuditedResult).run(suite)
    records = list(result.records.values())
    for record in records:
        record.setdefault("elapsed_seconds", 0.)  # E.g. setUpClass import failure.
    report = {"testsRun": result.testsRun, "tests": records,
              "failed": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
              "unexpected_successes": len(result.unexpectedSuccesses),
              "elapsed_seconds": round(time.perf_counter() - started, 3),
              "wasSuccessful": result.wasSuccessful()}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/fullstack_tests.json")
    parser.add_argument("--pattern", default="test*.py", help="Optional diagnostic subset; only test*.py proves the entire suite")
    parser.add_argument("--module-timeout", type=int, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.module_timeout < 1:
        parser.error("Module timeout must be a positive number of seconds")
    output = args.output.resolve()
    if output == (ROOT / "evaluation/system_results.json").resolve():
        parser.error("Historical system_results.json must remain unchanged")
    if args.worker:
        return run_worker(args.pattern, output)
    # These are deliberate integration requests, not optional skips. Fixture
    # tests touch only dedicated CloudCare services and UUID-owned test rows.
    os.environ["CLOUDCARE_INTEGRATION"] = "1"
    os.environ["CLOUDCARE_MYSQL_INTEGRATION"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    node = select_node()
    before = hashes()
    storage_before = storage_snapshot()
    historical_path = ROOT / "evaluation/system_results.json"
    historical_sha = hashlib.sha256(historical_path.read_bytes()).hexdigest() if historical_path.exists() else None
    modules = sorted((ROOT / "tests").glob(args.pattern))
    if not modules or any(path.parent != ROOT / "tests" or not path.name.startswith("test")
                          or path.suffix != ".py" for path in modules):
        parser.error("Test pattern must select checked-in test*.py files")
    started = time.perf_counter()
    records = []
    module_reports = []
    totals = Counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_directory = ROOT / "runtime" / "validation" / timestamp
    log_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cloudcare-test-reports-") as temporary:
        for module in modules:
            print(f"[validation] {module.name}", flush=True)
            worker_output = Path(temporary) / (module.stem + ".json")
            log_path = log_directory / (module.stem + ".log")
            command = [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
                       "--worker", "--pattern", module.name, "--output", str(worker_output)]
            module_started = time.perf_counter()
            timed_out = False
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                try:
                    returncode = process.wait(timeout=args.module_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    returncode = process.wait()
            module_report = {"module": module.stem, "process_exit_code": returncode,
                             "timed_out": timed_out,
                             "elapsed_seconds": round(time.perf_counter() - module_started, 3),
                             "log": log_path.relative_to(ROOT).as_posix()}
            if worker_output.exists():
                worker = json.loads(worker_output.read_text(encoding="utf-8"))
                records.extend(worker["tests"])
                for key in ("testsRun", "failed", "errors", "skipped", "expected_failures", "unexpected_successes"):
                    totals[key] += worker[key]
                module_report["worker_report_created"] = True
                module_report["wasSuccessful"] = worker["wasSuccessful"]
            else:
                # A Python/native process can stop before unittest creates a result.
                # Never silently turn that into zero tests or a successful module.
                records.append({"name": module.stem + ".__worker_process__", "status": "error",
                                "scope": scope_for(module.stem), "elapsed_seconds": module_report["elapsed_seconds"],
                                "error_type": "WorkerTimeout" if timed_out else "WorkerExitedWithoutReport"})
                totals["errors"] += 1
                module_report["worker_report_created"] = False
                module_report["wasSuccessful"] = False
            if returncode or timed_out:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                print(f"[validation] FAILED {module.name} exit={returncode}; log={log_path}", flush=True)
                print("\n".join(log_text.splitlines()[-25:]), flush=True)
            else:
                print(f"[validation] passed {module.name}", flush=True)
            module_reports.append(module_report)
    after = hashes()
    storage_after = storage_snapshot()
    status_counts = Counter(row["status"] for row in records)
    scope_counts = Counter(row["scope"] for row in records)
    changed = [name for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]
    historical_preserved = historical_sha == (hashlib.sha256(historical_path.read_bytes()).hexdigest()
                                             if historical_path.exists() else None)
    storage_preserved = all(storage_before[name]["ready"] and storage_after[name]["ready"]
        for name in ("mysql", "redis"))
    if storage_preserved:
        storage_preserved = (all(storage_before["mysql"]["counts"][key] == storage_after["mysql"]["counts"][key]
            for key in ("documents", "parents", "chunks", "faq"))
            and storage_before["redis"]["counts"] == storage_after["redis"]["counts"])
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner": "scripts/run_fullstack_checks.py", "python_version": platform.python_version(),
        "node": node, "integration_flags": {"CLOUDCARE_INTEGRATION": "1", "CLOUDCARE_MYSQL_INTEGRATION": "1"},
        "pattern": args.pattern, "complete_suite": args.pattern == "test*.py",
        "testsRun": totals["testsRun"], "passed": status_counts["passed"], "failed": totals["failed"],
        "errors": totals["errors"], "skipped": totals["skipped"],
        "expected_failures": totals["expected_failures"], "unexpected_successes": totals["unexpected_successes"],
        "elapsed_seconds": round(time.perf_counter() - started, 3), "status_counts": dict(status_counts),
        "scope_counts": dict(scope_counts), "tests": records, "modules": module_reports,
        "code_sha256": {name: sha for name, sha in before.items() if not name.startswith("tests/")},
        "code_sha256_after": {name: sha for name, sha in after.items() if not name.startswith("tests/")},
        "test_sha256": {name: sha for name, sha in before.items() if name.startswith("tests/")},
        "changed_files_during_run": changed, "historical_system_results_preserved": historical_preserved,
        "logs_directory": log_directory.relative_to(ROOT).as_posix(),
        "storage_snapshot_before": storage_before, "storage_snapshot_after": storage_after,
        "knowledge_and_live_faq_counts_preserved": storage_preserved,
        "scope_note": "Selected checked-in tests; complete_suite identifies whether all modules were selected. Historical baseline remains explicitly labeled. SQL/Redis tests use real isolated CloudCare services; document checks use real LangChain/PDF/Office/RapidOCR. API, pipeline, Qwen and most neural tests use injected collaborators; the small reranker adapter tensor runs on CPU. This suite is not a large-model quality evaluation, a real Milvus acceptance test, an external Qwen test, or a production business KPI.",
    }
    ok = (all(module["wasSuccessful"] and module["process_exit_code"] == 0 for module in module_reports)
          and node["ready"] and not totals["skipped"] and not changed and historical_preserved
          and storage_preserved and "unclassified_test_requires_review" not in scope_counts)
    report["passed_all_mandatory_checks"] = ok
    report["passed_complete_suite"] = ok and report["complete_suite"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "testsRun": totals["testsRun"], "passed": report["passed"],
        "failed": report["failed"], "errors": report["errors"], "skipped": report["skipped"],
        "passed_all_mandatory_checks": ok}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
