"""Restore packaged local evaluation records without overwriting local work."""
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evaluation/query_upgrade_20261002'
for archive in OUT.glob('*.gz'):
    if archive.name=='source_only_faq.json.gz':
        records=json.loads(gzip.decompress(archive.read_bytes()))
        folder=OUT/'source_only_faq';folder.mkdir(exist_ok=True)
        for name,record in records.items():
            if Path(name).name!=name or not name.endswith('.json'):raise ValueError('Invalid archive member')
            path=folder/name
            if not path.exists():path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    else:
        target=archive.with_suffix('')
        if not target.exists():target.write_bytes(gzip.decompress(archive.read_bytes()))
print('Evaluation archives restored; existing local files preserved.')
