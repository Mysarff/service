# 2026-10-02 functional regression commands

This directory preserves new regression evidence without changing the October 1 historical reports.

Environment: customer repository `.venv`, Python 3.11.15. No large BGE/BERT checkpoint is loaded and no Qwen request is made by this runner. Pipeline and model contracts use injected fixtures; dedicated MySQL/Redis and CPU RapidOCR checks are reported separately by scope.

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe scripts/run_fullstack_checks.py --worker --pattern test_faq_bm25_routing.py --output evaluation/validation_20261002/faq_boundary_tests.json
.venv\Scripts\python.exe scripts/run_fullstack_checks.py --output evaluation/validation_20261002/fullstack_tests.json
```

The FAQ tests check BM25 math, question-only indexing, normalization/category/tie behavior, >=8.0 FAQ acceptance, <8.0 fallback even for an exact question, absent/invalid/private sources, empty answers, and cache invalidation after FAQ version/threshold changes. Boundary fixtures explicitly configure 8.0 independently of the final deployed default (32.0). These are correctness contracts rather than measured FAQ accuracy or a calibrated threshold; actual selection/quality evidence is in the separate FAQ evaluation report.
