"""Deterministic 64-question retrieval comparison against the 2026-09-27 run.

The historical gzip is read-only. No HTTP timing or external model is involved.
"""
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine

GROUPS = ('cases.json', 'extended_cases.json')
METHODS = ('tfidf', 'plain_bm25', 'bm25')
FROZEN_CORPUS = ROOT / 'evaluation/fixtures/knowledge_20260927.jsonl.gz'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(observations):
    n = len(observations)
    ranks = [row['rank'] for row in observations]
    return {'top1': sum(rank == 1 for rank in ranks), 'hit5': sum(rank > 0 for rank in ranks),
            'mrr5': round(sum(1 / rank for rank in ranks if rank) / n, 4), 'n': n}


def observation(query, expected, ids):
    rank = min((ids.index(item) + 1 for item in expected if item in ids), default=0)
    return {'query': query, 'expected': expected, 'ids': ids, 'rank': rank}


def historical_engine(expected_sha256=None):
    """Replay the lexical algorithm on frozen source bytes, excluding uploads."""
    frozen = gzip.decompress(FROZEN_CORPUS.read_bytes())
    if expected_sha256 is not None and hashlib.sha256(frozen).hexdigest() != expected_sha256:
        raise ValueError('Frozen corpus differs from the historical benchmark')
    with tempfile.TemporaryDirectory(prefix='cloudcare-historical-corpus-') as folder:
        data = Path(folder)
        (data / 'knowledge.jsonl').write_bytes(frozen)
        with patch.object(engine, 'DATA', data), patch.object(engine, 'UPLOADS', data / 'uploads'):
            return engine.Engine()


def audit():
    baseline_path = ROOT / 'evaluation/benchmark_results.json.gz'
    with gzip.open(baseline_path, 'rt', encoding='utf-8') as file:
        baseline = json.load(file)
    groups = {name: json.loads((ROOT / 'evaluation' / name).read_text(encoding='utf-8')) for name in GROUPS}
    frozen = gzip.decompress(FROZEN_CORPUS.read_bytes())
    if hashlib.sha256(frozen).hexdigest() != baseline['corpus_sha256']:
        raise ValueError('Frozen corpus differs from the historical benchmark')
    active_sha = sha256(ROOT / 'data/knowledge.jsonl')
    for name, cases in groups.items():
        for method in METHODS:
            old_queries = [row['query'] for row in baseline['retrieval'][name][method]['observations']]
            if old_queries != [query for query, _ in cases['positive']]:
                raise ValueError(f'Question text/order changed: {name}/{method}')

    # Updating the active OCR runbook must not silently change the historical
    # comparison corpus or overwrite the recorded baseline fingerprint.
    search = historical_engine(baseline['corpus_sha256'])
    if len(search.docs) != baseline['corpus_units']:
        raise ValueError('Corpus unit count differs from the historical benchmark')

    report = {'baseline_commit': '307b9e0a7d46bf9337251f01ffa52b79f607bf51',
              'baseline_file_sha256': sha256(baseline_path),
              'corpus_sha256': baseline['corpus_sha256'], 'corpus_changed': False,
              'historical_scope': 'Current lexical algorithm replayed on frozen 2026-09-27 corpus; not full-stack model quality',
              'frozen_path': FROZEN_CORPUS.relative_to(ROOT).as_posix(),
              'frozen_file_sha256': sha256(FROZEN_CORPUS),
              'active_corpus_sha256': active_sha,
              'active_corpus_differs': active_sha != baseline['corpus_sha256'],
              'question_text_or_order_changed': False, 'positive_questions': 64,
              'original_case_sha256': baseline['cases_sha256'],
              'current_case_sha256': {name: sha256(ROOT / 'evaluation' / name) for name in GROUPS},
              'methods': {}, 'original_bm25_failures': [], 'current_bm25_failures': [], 'top1_changes': []}
    for method in METHODS:
        before = []
        relabeled = []
        after = []
        for name, cases in groups.items():
            old_rows = baseline['retrieval'][name][method]['observations']
            for old, (query, expected) in zip(old_rows, cases['positive']):
                before.append(observation(query, old['expected'], old['ids']))
                relabeled.append(observation(query, expected, old['ids']))
                after.append(observation(query, expected, [x['id'] for x in search.search(query, method=method)]))
        report['methods'][method] = {'historical': score(before), 'labels_only': score(relabeled),
                                     'current': score(after)}
        if method == 'bm25':
            report['original_bm25_failures'] = [row for row in before if row['rank'] != 1]
            report['current_bm25_failures'] = [row for row in after if row['rank'] != 1]
        for old, label, new in zip(before, relabeled, after):
            if old['rank'] != label['rank'] or label['rank'] != new['rank']:
                report['top1_changes'].append({'method': method, 'query': old['query'],
                                               'historical_rank': old['rank'],
                                               'labels_only_rank': label['rank'],
                                               'current_rank': new['rank'],
                                               'historical_top1': old['ids'][0],
                                               'current_top1': new['ids'][0]})
    return report


if __name__ == '__main__':
    report = audit()
    path = ROOT / 'evaluation/retrieval_audit.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['methods'], ensure_ascii=False, indent=2))
