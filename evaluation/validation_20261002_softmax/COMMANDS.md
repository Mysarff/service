# 2026-10-02 BM25 Softmax functional regression commands

This directory preserves final FAQ Softmax regression evidence independently of earlier raw-BM25 threshold reports and October 1 historical results.

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe scripts/run_fullstack_checks.py --worker --pattern test_faq_bm25_routing.py --output evaluation/validation_20261002_softmax/faq_boundary_tests.json
.venv\Scripts\python.exe scripts/run_fullstack_checks.py --output evaluation/validation_20261002_softmax/fullstack_tests.json
```

Environment: customer repository `.venv`, Python 3.11.15. The complete runner checks real dedicated MySQL/Redis and CPU OCR/Office/PDF/LangChain parsing, injected Pipeline/Qwen/neural contracts, and explicitly labeled historical baseline modules. It does not load the large BERT/BGE models or call external Qwen.

Final FAQ routing consumes a Softmax distribution over BM25 raw scores, with default threshold 0.85. Unmatched eligible FAQs contribute `exp(0)` to the denominator; normalization is never limited to positive matches or Top-K. The normalized value is relative to the corpus and is not a calibrated answer-correctness probability. Separate retrieval/FAQ/Qwen reports provide measured model-quality evidence.
