"""Scan frozen FAQ Softmax observations; never runs models or changes settings.

Run: python evaluation/faq_softmax_calibration_20261002/calibrate.py
Selection uses only the existing seeded development partition. Retained and
challenge rows are reported after selecting, never consulted by the selector.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUT = ROOT / 'evaluation/faq_softmax_20261002/results.jsonl'
SAFE_DENIAL_ID = 'extended_cases.json:negative:9'
# Developer-reviewed correction, supported by retained source text:
# This question asks if a seven-day promise is guaranteed; RET-09 denies making
# a promise and says qualification/timing depend on order/channel rules.
# Primary strict metrics preserve its original negative label.
REVIEW = {
    SAFE_DENIAL_ID: {
        'classification': 'safe_supported_denial',
        'expected_candidate_source_id': 'RET-09',
        'reason': 'Matched answer explicitly declines a uniform refund-time promise; it does not invent seven days.'
    }
}
UNKNOWN_FACT_IDS = {
    'cases.json:negative:8', 'cases.json:negative:9', 'cases.json:negative:11',
    *{f'extended_cases.json:negative:{i}' for i in [0, 1, 2, 6, 7, 8, 10, 11]},
    *{f'missing-{i:02d}' for i in range(1, 9)},
}


def stats(rows, threshold):
    accepted = [r for r in rows if r['score'] is not None and r['score'] >= threshold]
    correct = sum(r['kind'] == 'positive' and r['source_id_hit'] for r in accepted)
    reviewed = sum(r['id'] in REVIEW and r['candidate_source_id'] == REVIEW[r['id']]['expected_candidate_source_id']
                   for r in accepted)
    positives = sum(r['kind'] == 'positive' for r in rows)
    return dict(threshold=threshold, sample_count=len(rows), positive_count=positives,
                negative_count=len(rows) - positives, accepted_total=len(accepted), strict_correct=correct,
                safe_denial_review_correct=reviewed,
                strict_precision=correct / len(accepted) if accepted else None,
                reviewed_precision=(correct + reviewed) / len(accepted) if accepted else None,
                original_positive_correct_coverage=correct / positives if positives else None,
                positive_source_id_misses=sum(r['kind'] == 'positive' and not r['source_id_hit'] for r in accepted),
                negative_accepts=sum(r['kind'] == 'negative' for r in accepted),
                unknown_fact_negative_accepts=sum(r['id'] in UNKNOWN_FACT_IDS for r in accepted),
                nonreviewed_negative_accepts=sum(r['kind'] == 'negative' and r['id'] not in REVIEW for r in accepted),
                accepted_ids=[r['id'] for r in accepted])


def choose(sweep, precision_target, metric):
    feasible = [r for r in sweep if r['strict_correct'] > 0 and r[metric] >= precision_target
                and r['unknown_fact_negative_accepts'] <= 1]
    # Primary objective: correct original positive coverage; ties prefer better
    # precision then smaller threshold. Labels from retained/challenge are absent.
    return min(feasible, key=lambda r: (-r['strict_correct'], -r[metric], r['threshold'])) if feasible else None


def distribution(rows):
    result = {}
    groups = {
        'correct_positive': [r for r in rows if r['kind'] == 'positive' and r['source_id_hit']],
        'incorrect_positive': [r for r in rows if r['kind'] == 'positive' and not r['source_id_hit']],
        'negative': [r for r in rows if r['kind'] == 'negative'],
        'unknown_fact_negative': [r for r in rows if r['id'] in UNKNOWN_FACT_IDS],
    }
    for group, cases in groups.items():
        values = [r['score'] for r in cases if r['score'] is not None]
        result[group] = dict(count=len(cases), no_candidate=sum(r['score'] is None for r in cases),
                            min=min(values) if values else None,
                            median=statistics.median(values) if values else None,
                            max=max(values) if values else None)
    return result


def main():
    raw = INPUT.read_bytes() if INPUT.is_file() else gzip.decompress(INPUT.with_suffix(INPUT.suffix + '.gz').read_bytes())
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    dev = [r for r in rows if r.get('partition') == 'development']
    held = [r for r in rows if r.get('partition') == 'retained']
    challenge = [r for r in rows if 'partition' not in r]
    assert len(dev) == 44 and len(held) == 44 and len(challenge) == 20
    boundaries = [value for r in dev if r['score'] is not None
                  for value in [r['score'], math.nextafter(r['score'], math.inf)] if 0 < value < 1]
    thresholds = sorted(set([i / 100 for i in range(1, 100)] + boundaries))
    sweep = [stats(dev, t) for t in thresholds]
    strategies = []
    for metric in ['strict_precision', 'reviewed_precision']:
        for target in [.95, .90, .80, .75, .70, .50]:
            selected = choose(sweep, target, metric)
            strategies.append(dict(metric=metric, target=target, selected_development=selected,
                                   retained_at_selected=stats(held, selected['threshold']) if selected else None,
                                   challenge_at_selected=stats(challenge, selected['threshold']) if selected else None))
    diagnostic = [dict(development=stats(dev, t), retained=stats(held, t), all_regression=stats(dev + held, t),
                       challenge=stats(challenge, t))
                  for t in [.99, .85, .80, .75, .70, .60, .55, .50, .45, .44, .40, .30, .20, .10, .05, .01]]
    coarse_grid = [round(i / 20, 2) for i in range(1, 20)]
    coarse_sweep = [stats(dev, t) for t in coarse_grid]
    coarse_feasible = [r for r in coarse_sweep if r['strict_correct'] > 0 and r['strict_precision'] >= .75
                       and r['negative_accepts'] <= 1]
    max_correct = max((r['strict_correct'] for r in coarse_feasible), default=None)
    coarse_best = [r for r in coarse_feasible if r['strict_correct'] == max_correct]
    # A higher grid point with the same accepted IDs is a conservative
    # representative of the same development decision region.
    coarse_selected = max(coarse_best, key=lambda r: (r['strict_precision'], r['threshold'])) if coarse_best else None
    strict90 = next(r for r in strategies if r['metric'] == 'strict_precision' and r['target'] == .90)
    reviewed90 = next(r for r in strategies if r['metric'] == 'reviewed_precision' and r['target'] == .90)
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), algorithm='faq_bm25_softmax_v2',
                   normalization='softmax_all_faq', source_results_sha256=hashlib.sha256(raw).hexdigest(),
                   reran_models=False, altered_core=False,
                   protocol=dict(seed=20261002, development_positive=32, development_negative=12,
                                 retained_positive=32, retained_negative=12, challenge_positive=8, challenge_negative=12,
                                 thresholds='0.01..0.99 plus exact development score boundaries and nextafter(score,+inf)',
                                 threshold_count=len(thresholds), max_unknown_fact_negative_accepts=1,
                                 initial_diagnostic_target=.90, current_development_target=.75,
                                 objective='maximum correct original positive coverage, tie higher precision then lower threshold'),
                   manual_review=REVIEW, unknown_fact_negative_ids=sorted(UNKNOWN_FACT_IDS),
                   strict90_feasible=strict90['selected_development'] is not None,
                   strict90_selected=strict90,
                   reviewed90_selected=reviewed90,
                   coarse_development_selection=dict(precision_target=.75, metric='strict_precision',
                       grid=coarse_grid, maximum_original_negative_accepts=1, objective='maximum strict-correct coverage; same decision region use higher grid point',
                       selected_development=coarse_selected,
                       retained_at_selected=stats(held, coarse_selected['threshold']) if coarse_selected else None,
                       challenge_at_selected=stats(challenge, coarse_selected['threshold']) if coarse_selected else None),
                   conservative_exploratory_055=dict(threshold=.55, development=stats(dev, .55),
                       retained=stats(held, .55), all_regression=stats(dev + held, .55), challenge=stats(challenge, .55),
                       selection_note='Post-audit conservative trial setting, not selected by the development strict-75% rule; does not meet strict-75% on development.'),
                   recommended_trial_threshold=.55,
                   recommendation='Use 0.55 as the user-authorized conservative exploratory trial setting after the additional risk audit. Low coverage and one retained wrong source remain; no observed challenge unknown-fact accepts. Formal development strict-75% grid independently selects 0.45. Neither is a validated optimal production threshold.',
                   strategies=strategies, diagnostic_thresholds=diagnostic,
                   score_distributions=dict(development=distribution(dev), retained=distribution(held), challenge=distribution(challenge)),
                   limitations=[
                       'All observations are frozen results from the current corpus, not live recomputation after code/data changes.',
                       'Current split is synthetic prior development material, not an independent external or blind benchmark.',
                       'Strict source-ID correctness is a relevance proxy. The reviewed metric corrects only one independently checked safe denial label.',
                       'Retained/challenge results are reported after selection and never used to tune threshold.',
                       'Unknown-fact acceptance records non-answering FAQ text, not a generated fictitious fact or full-system error.',
                       'BERT and RAG may change complete-system outcomes; this study only scores the FAQ gate.',
                       'Near-equal and shared-template FAQ candidates lower concentration; rare boundary bigrams can raise wrong candidates.',
                       'A low-count development precision of 100% is not evidence of stable 90% generalization.'
                   ])
    (OUT / 'development_sweep.json').write_text(json.dumps(sweep, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'strict90_feasible':summary['strict90_feasible'], 'reviewed90_selected':reviewed90,
                      'strict_strategies':[r for r in strategies if r['metric']=='strict_precision']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
