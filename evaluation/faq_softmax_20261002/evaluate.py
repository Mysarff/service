"""Audit fixed FAQ whole-index Softmax threshold 0.85 without model calls.

Run: python evaluation/faq_softmax_20261002/evaluate.py
No threshold tuning; preserves the user's education-project score convention.
"""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RAW = ROOT / 'evaluation/faq_bm25_20261002'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RAW))
spec = importlib.util.spec_from_file_location('raw_faq_audit_helpers', RAW / 'evaluate.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
from cloudcare.faq import FAQBM25Index, FAQ_BM25_ALGORITHM, FAQ_BM25_K1, FAQ_BM25_B

THRESHOLD = 0.85


def independent_whole_index_softmax(index, query, category=''):
    """Slow reference over *all* eligible rows, including raw zero scores."""
    terms = set(index.tokenize(query))
    scores = []
    n = len(index.rows)
    for position, row in enumerate(index.rows):
        if category and row.get('category') != category:
            continue
        bag = index.bags[position]
        common = terms & bag.keys()
        norm = FAQ_BM25_K1 * (1 - FAQ_BM25_B + FAQ_BM25_B * index.lengths[position] / max(1, index.avg_length))
        score = sum(math.log(1 + (n - len(index.postings[t]) + .5) / (len(index.postings[t]) + .5))
                    * bag[t] * (FAQ_BM25_K1 + 1) / (bag[t] + norm) for t in sorted(common))
        scores.append(score)
    if not scores:
        return None
    best = max(scores)
    return 1 / math.fsum(math.exp(s - best) for s in scores)


def score_case(case, index):
    row = index.search(case['query'])
    if row:
        assert row['normalization'] == 'softmax_all_faq'
        reference = independent_whole_index_softmax(index, case['query'])
        assert math.isclose(row['score'], reference, rel_tol=1e-12, abs_tol=1e-12), (case['id'], row['score'], reference)
    return {**case, 'score': row['score'] if row else None, 'raw_score': row['raw_score'] if row else None,
            'candidate_faq_id': row['id'] if row else None, 'candidate_question': row['question'] if row else None,
            'candidate_source_id': row['source_id'] if row else None, 'candidate_answer': row['answer'] if row else None,
            'candidate_count': row['candidate_count'] if row else 0,
            'eligible_count': row['eligible_count'] if row else len(index.rows),
            'normalization': row['normalization'] if row else 'softmax_all_faq',
            'accepted': bool(row and row['score'] >= THRESHOLD),
            'source_id_hit': bool(row and row['source_id'] in case.get('expected_source_ids', []))}


def main():
    started = time.perf_counter()
    knowledge_rows, knowledge_bytes = helpers.read_jsonl(ROOT / 'data/knowledge.jsonl')
    knowledge = {r['id']: r for r in knowledge_rows}
    faqs, faq_bytes = helpers.read_jsonl(ROOT / 'data/faq.jsonl')
    index = FAQBM25Index([{**r, 'category': knowledge[r['source_id']]['category']} for r in faqs])
    assert FAQ_BM25_ALGORITHM == 'faq_bm25_softmax_v2'
    cases = helpers.frozen_cases(knowledge)
    challenges = json.loads((RAW / 'challenge_cases.json').read_text(encoding='utf-8'))['cases']
    scored = [score_case(case, index) for case in cases]
    challenge_scored = [score_case(case, index) for case in challenges]
    exact = []
    for row in faqs:
        candidate = index.search(row['question'])
        exact.append(dict(faq_id=row['id'], expected_source_id=row['source_id'],
                          candidate_source_id=candidate['source_id'] if candidate else None,
                          score=candidate['score'] if candidate else None,
                          raw_score=candidate['raw_score'] if candidate else None,
                          accepted=bool(candidate and candidate['score'] >= THRESHOLD),
                          correct=bool(candidate and candidate['source_id'] == row['source_id'])))
    # Cross-check a category-filtered query in every category as well.
    category_checks = []
    for category in sorted({r['category'] for r in index.rows}):
        row = next(r for r in index.rows if r['category'] == category)
        result = index.search(row['question'], category)
        expected = independent_whole_index_softmax(index, row['question'], category)
        assert math.isclose(result['score'], expected, rel_tol=1e-12, abs_tol=1e-12)
        category_checks.append(dict(category=category, eligible_count=result['eligible_count'], score=result['score']))
    assert index.search('') is None
    assert index.search('zqxwjkvbnonexistenttoken') is None
    distribution = [r['score'] for r in exact if r['score'] is not None]
    faq_strings = {helpers.canonical(f['question']) for f in faqs}
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='FAQ BM25 whole-index Softmax only; no BERT, Redis, Milvus, BGE or Qwen calls',
        algorithm=FAQ_BM25_ALGORITHM, k1=FAQ_BM25_K1, b=FAQ_BM25_B,
        normalization='softmax_all_faq', fixed_threshold=THRESHOLD, threshold_tuned=False,
        faq_records=len(faqs), knowledge_source_ids=len(knowledge),
        faq_variants_per_source_id=dict(Counter(Counter(r['source_id'] for r in faqs).values())),
        normalization_crosschecks=dict(all_eligible_including_zero_scores_queries=len(scored) + len(challenge_scored),
                                      category_filtered_queries=len(category_checks), tolerance=1e-12),
        all_regression=helpers.stats(scored, THRESHOLD),
        development_partition=helpers.stats([r for r in scored if r['partition'] == 'development'], THRESHOLD),
        retained_partition=helpers.stats([r for r in scored if r['partition'] == 'retained'], THRESHOLD),
        challenge=helpers.stats(challenge_scored, THRESHOLD),
        challenge_by_group={g: helpers.stats([r for r in challenge_scored if r['group'] == g], THRESHOLD)
                            for g in sorted({r['group'] for r in challenge_scored})},
        indexed_exact_question_diagnostic=dict(query_count=len(exact), correct_top1_source_ids=sum(r['correct'] for r in exact),
            accepted_at_085=sum(r['accepted'] for r in exact), correct_and_accepted=sum(r['accepted'] and r['correct'] for r in exact),
            softmax_min=min(distribution), softmax_median=statistics.median(distribution), softmax_max=max(distribution),
            note='Known indexed questions; Top1 correctness is separate from passing the FAQ routing threshold.'),
        query_duplicate_audit=dict(regression_exact_in_faq=sum(helpers.canonical(r['query']) in faq_strings for r in scored),
                                  challenge_exact_in_faq=sum(helpers.canonical(r['query']) in faq_strings for r in challenge_scored)),
        input_sha256={'knowledge.jsonl': helpers.digest(knowledge_bytes), 'faq.jsonl': helpers.digest(faq_bytes),
                      'faq.py': helpers.digest((ROOT / 'cloudcare/faq.py').read_bytes()),
                      'cases.json': helpers.digest((ROOT / 'evaluation/cases.json').read_bytes()),
                      'extended_cases.json': helpers.digest((ROOT / 'evaluation/extended_cases.json').read_bytes()),
                      'challenge_cases.json': helpers.digest((RAW / 'challenge_cases.json').read_bytes())},
        limitations=['0.85 is the user-requested inherited threshold; no optimum/calibration claim.',
                     'Softmax is relative score concentration, not a correctness probability.',
                     'Eight alternate FAQ phrasings per source compete individually in the denominator.',
                     'Regression and challenge material are synthetic developer-authored, not a blind external benchmark.',
                     'Strict source-ID hit measures relevance, not complete answer coverage or factual accuracy.',
                     'No acceptance means the FAQ branch declines; subsequent RAG effectiveness is not measured here.',
                     'Whole-system BERT routing can intercept actions/out-of-scope queries before FAQ.'],
        elapsed_seconds=time.perf_counter() - started)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in scored + challenge_scored), encoding='utf-8')
    (OUT / 'exact_question_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in exact), encoding='utf-8')
    (OUT / 'category_normalization_checks.json').write_text(json.dumps(category_checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['algorithm', 'fixed_threshold', 'all_regression', 'challenge',
        'indexed_exact_question_diagnostic', 'normalization_crosschecks']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
