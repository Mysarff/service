"""Re-run the unchanged support BERT checkpoint on its frozen held-out queries.

This measures synthetic intention classification, not FAQ matching or RAG
answer correctness. The original dataset and checkpoint are read-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from cloudcare.neural import SupportBertRouter, SUPPORT_LABELS
    from cloudcare.settings import Settings
    from scripts.train_support_router import evaluate

    settings = Settings.load()
    dataset = ROOT / "evaluation/support_router/dataset.jsonl"
    original = json.loads((settings.bert_model_path / "training_report.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [row for row in rows if row["split"] == "train"]
    heldout = [row for row in rows if row["split"] == "test"]
    overlap = {row["group"] for row in train} & {row["group"] for row in heldout}
    if overlap or sha(dataset) != original["dataset_sha256"]:
        raise ValueError("Frozen evaluation dataset or group split does not match training evidence")
    output = Path(__file__).resolve().parent / "bert_heldout.json"
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current unchanged support BERT checkpoint inference on 432 frozen synthetic held-out examples; no retraining, not answer quality",
        "dataset_sha256": sha(dataset), "checkpoint_sha256": original["checkpoint_sha256"],
        "current_knowledge_sha256": sha(ROOT / "data/knowledge.jsonl"),
        "training_knowledge_sha256": original["knowledge_sha256"],
        "train_count": len(train), "test_count": len(heldout), "group_overlap": len(overlap),
        "labels": list(SUPPORT_LABELS), "training_scope": original["training_scope"],
        "code_sha256": {str(path.relative_to(ROOT).as_posix()): sha(path) for path in
            (Path(__file__).resolve(), ROOT / "cloudcare/neural.py", ROOT / "scripts/train_support_router.py")},
        "observations": [], "complete": False}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    router = SupportBertRouter(settings)
    started = time.perf_counter()
    predicted, expected = [], []
    try:
        for index, row in enumerate(heldout, 1):
            result = router.predict(row["query"])
            expected.append(SUPPORT_LABELS.index(row["label"]))
            predicted.append(SUPPORT_LABELS.index(result["intent"]))
            report["observations"].append({"query": row["query"], "label": row["label"],
                "group": row["group"], "prediction": result["intent"], "confidence": result["confidence"]})
            if index % 64 == 0:
                save()
                print(f"Current BERT held-out inference: {index}/{len(heldout)}", flush=True)
        report.update(evaluate(predicted, expected))
        report["complete"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        router.release_model()
        report["wall_seconds"] = time.perf_counter() - started
        save()
    print(json.dumps({key: report[key] for key in ("complete", "test_count", "accuracy", "macro_f1", "wall_seconds")}))


if __name__ == "__main__":
    main()
