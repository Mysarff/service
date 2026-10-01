"""Resource lifecycle checks use adapters, never real GPU/model inference."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cloudcare_doctor_lifecycle", ROOT / "scripts/doctor.py")
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


class DoctorReleaseTests(unittest.TestCase):
    def check_case(self, kind: str, fail: bool) -> None:
        events = []

        class Adapter:
            def __init__(self, *args, **kwargs):
                events.append("constructed")

            def invoke(self):
                events.append("inference")
                if fail:
                    raise RuntimeError("fixture inference failure")

            def encode_queries(self, _):
                self.invoke()
                return [SimpleNamespace(dense=[0.0] * 1024, sparse={1: 1.0})]

            def rerank(self, *args, **kwargs):
                self.invoke()
                return [{"rerank_score": 0.5}]

            def predict(self, _):
                self.invoke()
                return {"probabilities": {"support_knowledge": 1.0},
                        "intent": "support_knowledge", "artifact_verified": True}

            def release_model(self):
                events.append("released")

        fake_torch = SimpleNamespace(set_num_threads=lambda _: None)
        fake_neural = SimpleNamespace(BGEM3Encoder=Adapter, BGEReranker=Adapter, SupportBertRouter=Adapter)
        with patch.dict(sys.modules, {"torch": fake_torch, "cloudcare.neural": fake_neural}):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "fixture inference failure"):
                    doctor.infer_model(kind, SimpleNamespace(device="cpu"))
            else:
                self.assertEqual(doctor.infer_model(kind, SimpleNamespace(device="cpu"))["status"], "inference_verified")
        self.assertEqual(events, ["constructed", "inference", "released"])

    def test_adapters_released_on_success(self):
        for kind in ("embedding", "reranker", "bert"):
            with self.subTest(kind=kind):
                self.check_case(kind, False)

    def test_adapters_released_on_failure(self):
        for kind in ("embedding", "reranker", "bert"):
            with self.subTest(kind=kind):
                self.check_case(kind, True)


if __name__ == "__main__":
    unittest.main()
