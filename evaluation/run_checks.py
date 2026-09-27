"""Persist integration test outcomes without contacting any model provider."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import unittest
import io
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
stream=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'passed':result.wasSuccessful(),'scope':'isolated local HTTP and engine; mocked provider only','output':stream.getvalue()}
(ROOT/'evaluation/system_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
print(stream.getvalue()); sys.exit(0 if result.wasSuccessful() else 1)
