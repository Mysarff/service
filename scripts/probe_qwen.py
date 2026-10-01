"""Two bounded, real API calls; never print credentials or raw provider errors."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloudcare.settings import Settings
from cloudcare.llm import QwenGateway

def main():
    settings = Settings.load()
    gateway = QwenGateway(settings)
    row = json.loads((settings.root/'data/knowledge.jsonl').read_text(encoding='utf-8').splitlines()[0])
    report = {'scope':'real DashScope Qwen API, synthetic evidence; rewrite and cited answer only'}
    try:
        report['rewrite'] = gateway.rewrite('账号怎么开通？', [])
        report['answer'] = gateway.generate('账号怎么开通？', [row], [])
        report['success'] = True
    except Exception as exc:
        report.update(success=False,error_type=type(exc).__name__)
    report['qwen'] = gateway.status()
    settings.runtime_dir.mkdir(parents=True,exist_ok=True)
    (settings.runtime_dir/'qwen_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['success'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
