"""Offline FAQ BM25 audit. No BERT, Redis, Milvus or Qwen is loaded.

Run: python evaluation/faq_bm25_20261002/evaluate.py
The original 88 queries are development material, not a blind test.
"""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from faq_raw_snapshot import FAQBM25Index, FAQ_BM25_ALGORITHM, FAQ_BM25_K1, FAQ_BM25_B

SEED = 20261002
PRECISION_TARGET = 0.95
THRESHOLDS = list(range(101))


def read_jsonl(path):
    raw = path.read_bytes() if path.is_file() else gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())
    return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()], raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(text):
    # Duplicate audit only; never used for routing.
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def stats(rows, threshold):
    pos = [r for r in rows if r['kind'] == 'positive']
    neg = [r for r in rows if r['kind'] == 'negative']
    accepted = [r for r in rows if r['score'] is not None and r['score'] >= threshold]
    correct = sum(r['source_id_hit'] for r in accepted)
    return dict(threshold=threshold, sample_count=len(rows), positive_count=len(pos), negative_count=len(neg),
                accepted_total=len(accepted), positive_accepted=sum(r['kind'] == 'positive' for r in accepted),
                positive_correct=correct,
                positive_source_id_miss=sum(r['kind'] == 'positive' and not r['source_id_hit'] for r in accepted),
                negative_accepted=sum(r['kind'] == 'negative' for r in accepted),
                source_id_precision_proxy=correct / len(accepted) if accepted else None,
                positive_correct_direct_coverage=correct / len(pos) if pos else None,
                positive_sent_to_rag=sum(r['score'] is None or r['score'] < threshold for r in pos),
                negative_sent_to_rag=sum(r['score'] is None or r['score'] < threshold for r in neg))


def score_case(case, index):
    r = index.search(case['query'])
    return {**case, 'score': r['score'] if r else None, 'candidate_faq_id': r['id'] if r else None,
            'candidate_question': r['question'] if r else None,
            'candidate_source_id': r['source_id'] if r else None,
            'candidate_answer': r['answer'] if r else None,
            'candidate_count': r['candidate_count'] if r else 0,
            'source_id_hit': bool(r and r['source_id'] in case.get('expected_source_ids', []))}


def frozen_cases(knowledge):
    grouped = defaultdict(list)
    negative_groups, all_cases = [], []
    for filename in ['cases.json', 'extended_cases.json']:
        data = json.loads((ROOT / 'evaluation' / filename).read_text(encoding='utf-8'))
        for number, (query, ids) in enumerate(data['positive']):
            case = dict(id=f'{filename}:positive:{number}', query=query, kind='positive', origin=filename,
                        category=knowledge[ids[0]]['category'], expected_source_ids=ids,
                        group='natural_rewrite_regression')
            grouped[case['category']].append(case)
            all_cases.append(case)
        negatives = [dict(id=f'{filename}:negative:{n}', query=q, kind='negative', origin=filename,
                          expected_source_ids=[], group='non_direct_regression')
                     for n, q in enumerate(data['negative'])]
        negative_groups.append(negatives)
        all_cases.extend(negatives)
    # Class-balanced split: 2/4 positives per category, 6/12 negatives per file.
    # Neither query scores nor predictions are consulted by this split.
    rng = random.Random(SEED)
    dev_ids, retained_ids = set(), set()
    for category in sorted(grouped):
        rows = grouped[category][:]
        assert len(rows) == 4
        rng.shuffle(rows)
        dev_ids.update(r['id'] for r in rows[:2])
        retained_ids.update(r['id'] for r in rows[2:])
    for group in negative_groups:
        rows = group[:]
        assert len(rows) == 12
        rng.shuffle(rows)
        dev_ids.update(r['id'] for r in rows[:6])
        retained_ids.update(r['id'] for r in rows[6:])
    assert len(dev_ids) == 44 and len(retained_ids) == 44 and not dev_ids & retained_ids
    for row in all_cases:
        row['partition'] = 'development' if row['id'] in dev_ids else 'retained'
    return all_cases


def main():
    started = time.perf_counter()
    knowledge_rows, knowledge_bytes = read_jsonl(ROOT / 'data/knowledge.jsonl')
    knowledge = {r['id']: r for r in knowledge_rows}
    faqs, faq_bytes = read_jsonl(ROOT / 'data/faq.jsonl')
    assert len(faqs) == 4736 and all(r['source_id'] in knowledge for r in faqs)
    index = FAQBM25Index([{**r, 'category': knowledge[r['source_id']]['category']} for r in faqs])
    cases = frozen_cases(knowledge)
    (OUT / 'frozen_cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    scored = [score_case(case, index) for case in cases]
    dev = [r for r in scored if r['partition'] == 'development']
    retained = [r for r in scored if r['partition'] == 'retained']
    sweep = [stats(dev, t) for t in THRESHOLDS]
    feasible = [r for r in sweep if r['accepted_total'] > 0 and r['source_id_precision_proxy'] >= PRECISION_TARGET
                and r['negative_accepted'] == 0]
    selected = min(feasible, key=lambda r: (-r['positive_accepted'], r['threshold'])) if feasible else None
    threshold = selected['threshold'] if selected else None
    # Known indexed strings: usability check, never used for selecting threshold.
    exact = []
    for row in faqs:
        r = index.search(row['question'])
        exact.append(dict(id=row['id'], score=r['score'] if r else None,
                          correct=bool(r and r['source_id'] == row['source_id'])))
    challenges = json.loads((OUT / 'challenge_cases.json').read_text(encoding='utf-8'))['cases']
    assert all(s in knowledge for r in challenges for s in r.get('expected_source_ids', []))
    challenge_scored = [score_case(case, index) for case in challenges]
    faq_strings = {canonical(r['question']) for r in faqs}
    token_bags = {tuple(sorted(index.tokenize(r['question']))) for r in faqs}
    misses_at_8 = [r for r in scored if r['kind'] == 'positive' and r['score'] is not None
                   and r['score'] >= 8 and not r['source_id_hit']]
    same_parent_misses = [r for r in misses_at_8 if knowledge[r['candidate_source_id']]['parent_id']
                         in {knowledge[s]['parent_id'] for s in r['expected_source_ids']}]
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                   scope='FAQ question BM25 and threshold only; no BERT/Redis/Milvus/Qwen calls',
                   algorithm=FAQ_BM25_ALGORITHM, k1=FAQ_BM25_K1, b=FAQ_BM25_B,
                   faq_records=len(faqs), knowledge_source_ids=len(knowledge), seed=SEED,
                   selection_protocol=dict(development_positive=32, development_negative=12,
                                           retained_positive=32, retained_negative=12, integer_thresholds=[0, 100],
                                           minimum_source_id_precision_proxy=PRECISION_TARGET,
                                           maximum_development_negative_accepts=0,
                                           objective='maximize development positive accepts, tie lowest threshold'),
                   selected_threshold=threshold,
                   development_at_8=stats(dev, 8), retained_at_8=stats(retained, 8), all_regression_at_8=stats(scored, 8),
                   development_at_selected=stats(dev, threshold) if threshold is not None else None,
                   retained_at_selected=stats(retained, threshold) if threshold is not None else None,
                   all_regression_at_selected=stats(scored, threshold) if threshold is not None else None,
                   source_miss_at_8_parent_analysis=dict(total=len(misses_at_8),
                       same_parent=len(same_parent_misses), cross_parent=len(misses_at_8) - len(same_parent_misses)),
                   indexed_exact_question_diagnostic=dict(queries=len(exact), correct_source_ids=sum(r['correct'] for r in exact),
                       accepted_at_selected=sum(r['score'] >= threshold for r in exact) if threshold is not None else 0,
                       min_score=min(r['score'] for r in exact), max_score=max(r['score'] for r in exact),
                       usage='within-index diagnostic; not generalization, independent knowledge or threshold selection'),
                   challenge_at_8=stats(challenge_scored, 8),
                   challenge_at_selected=stats(challenge_scored, threshold) if threshold is not None else None,
                   challenge_group_at_selected={g: stats([r for r in challenge_scored if r['group'] == g], threshold)
                                                for g in sorted({r['group'] for r in challenge_scored})} if threshold is not None else None,
                   duplicate_audit=dict(regression_canonical_exact_in_faq=sum(canonical(r['query']) in faq_strings for r in scored),
                       regression_token_bag_exact_in_faq=sum(tuple(sorted(index.tokenize(r['query']))) in token_bags for r in scored),
                       challenge_canonical_exact_in_faq=sum(canonical(r['query']) in faq_strings for r in challenge_scored),
                       regression_unique_canonical_queries=len({canonical(r['query']) for r in scored})),
                   input_sha256={'knowledge.jsonl': digest(knowledge_bytes), 'faq.jsonl': digest(faq_bytes),
                                 **{f: digest((ROOT / 'evaluation' / f).read_bytes()) for f in ['cases.json', 'extended_cases.json']},
                                 'faq.py': digest((OUT / 'faq_raw_snapshot.py').read_bytes()),
                                 'challenge_cases.json': digest((OUT / 'challenge_cases.json').read_bytes())},
                   limitations=[
                       'Original regression files are synthetic development material used in previous development.',
                       'Quick all-regression results were inspected before the split; retained is not blind.',
                       'Source-ID hit is a strict relevance proxy, not factual accuracy or answer coverage.',
                       'Same-parent diagnosis/repair may partially answer one another: not every miss is a factual error.',
                       'FAQ-only negative acceptance is not complete-system error: BERT can route actions or out-of-scope first.',
                       'Refund-time negative can be safely denied by caution text; negative labels are routing expectations.',
                       'Challenges are agent-authored after corpus review, not real users or an external benchmark.',
                       'Raw BM25 score depends on corpus/tokenizer/query; it cannot certify answer coverage.',
                       'No neural retrieval quality, generation truthfulness, latency or business benefit is measured.'],
                   elapsed_seconds=time.perf_counter() - started)
    (OUT / 'threshold_sweep.json').write_text(json.dumps(sweep, indent=2) + '\n', encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in scored + challenge_scored), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['selected_threshold', 'all_regression_at_8',
                     'all_regression_at_selected', 'challenge_at_selected', 'indexed_exact_question_diagnostic']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
