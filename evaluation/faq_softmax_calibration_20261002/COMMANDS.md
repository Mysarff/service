# FAQ threshold calibration and 0.55 boundary checks

```powershell
.venv\Scripts\python.exe evaluation/faq_softmax_calibration_20261002/calibrate.py
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe scripts/run_fullstack_checks.py --worker --pattern test_faq_bm25_routing.py --output evaluation/faq_softmax_calibration_20261002/threshold_boundary_tests.json
```

The first command reads frozen FAQ Softmax observations and scans development-only thresholds; it calls no model and changes no settings. The second command runs the current FAQ-only regression module. Its original 18 mathematical/routing contracts explicitly retain historical 0.85 boundary fixtures, while two additional tests verify the deployed constructor default 0.55 and both sides of the 0.55 gate.

Pipeline dependencies in these threshold tests are injected fixtures. They verify comparison, source checks and cache behavior rather than real BGE/BERT/Qwen quality. The current 0.55 setting is a conservative exploratory trial, not an independently validated optimal threshold.
