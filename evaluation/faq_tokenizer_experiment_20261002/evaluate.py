"""Two predeclared question-only Jieba tokenizers; core remains unchanged.

Run: .venv/Scripts/python.exe evaluation/faq_tokenizer_experiment_20261002/evaluate.py
Only the frozen development partition selects tokenizer/threshold. Retained
and challenge outcomes are reported afterward and do not enter selection.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import jieba
from cloudcare.faq import FAQBM25Index, FAQ_BM25_K1, FAQ_BM25_B


class JiebaCutFAQIndex(FAQBM25Index):
    @staticmethod
    def tokenize(text):
        # Exact original education preprocessing convention, read-only reference:
        # mysql_qa/utils/preprocess.py uses jieba.lcut(text.lower()).
        # No hand-authored dictionary, stopword list, or query expansion.
        return jieba.lcut(str(text).lower())


class JiebaSearchFAQIndex(FAQBM25Index):
    @staticmethod
    def tokenize(text):
        return jieba.lcut_for_search(str(text).lower())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path):
    raw = path.read_bytes() if path.is_file() else gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())
    return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()], raw


def score(case, index):
    result = index.search(case['query'])
    return {**case, 'score': result['score'] if result else None,
            'raw_score': result['raw_score'] if result else None,
            'candidate_faq_id': result['id'] if result else None,
            'candidate_question': result['question'] if result else None,
            'candidate_source_id': result['source_id'] if result else None,
            'candidate_answer': result['answer'] if result else None,
            'candidate_count': result['candidate_count'] if result else 0,
            'eligible_count': result['eligible_count'] if result else len(index.rows),
            'source_id_hit': bool(result and result['source_id'] in case.get('expected_source_ids', []))}


def stats(rows, threshold):
    accepted = [row for row in rows if row['score'] is not None and row['score'] >= threshold]
    positives = [row for row in rows if row['kind'] == 'positive']
    correct = sum(row['kind'] == 'positive' and row['source_id_hit'] for row in accepted)
    return dict(threshold=threshold, sample_count=len(rows), positive_count=len(positives),
                negative_count=len(rows) - len(positives), accepted_total=len(accepted),
                strict_correct=correct,
                strict_precision=correct / len(accepted) if accepted else None,
                correct_positive_coverage=correct / len(positives) if positives else None,
                positive_source_id_misses=sum(row['kind'] == 'positive' and not row['source_id_hit'] for row in accepted),
                negative_accepts=sum(row['kind'] == 'negative' for row in accepted),
                accepted_ids=[row['id'] for row in accepted])


def top1(rows):
    positives = [row for row in rows if row['kind'] == 'positive']
    return dict(positive_count=len(positives), source_id_hits=sum(row['source_id_hit'] for row in positives),
                source_id_hit_rate=sum(row['source_id_hit'] for row in positives) / len(positives),
                no_candidate=sum(row['score'] is None for row in rows))


def choose(sweep, target, require_no_negative=True):
    feasible = [row for row in sweep if row['strict_correct'] > 0
                and row['strict_precision'] >= target
                and (not require_no_negative or row['negative_accepts'] == 0)]
    return min(feasible, key=lambda row: (-row['strict_correct'], -row['strict_precision'], row['threshold'])) if feasible else None


def pareto(sweep):
    distinct = {}
    for row in sweep:
        if row['accepted_total']:
            key = (row['strict_precision'], row['strict_correct'])
            distinct.setdefault(key, row)
    return [row for key, row in distinct.items()
            if not any(other[0] >= key[0] and other[1] >= key[1] and other != key for other in distinct)]


def distribution(rows):
    groups = {'correct_positive': [row for row in rows if row['kind'] == 'positive' and row['source_id_hit']],
              'incorrect_positive': [row for row in rows if row['kind'] == 'positive' and not row['source_id_hit']],
              'negative': [row for row in rows if row['kind'] == 'negative']}
    output = {}
    for name, group in groups.items():
        values = [row['score'] for row in group if row['score'] is not None]
        output[name] = dict(count=len(group), no_candidate=len(group) - len(values),
                            min=min(values) if values else None,
                            median=statistics.median(values) if values else None,
                            max=max(values) if values else None)
    return output


def main():
    started = time.perf_counter()
    core_path = ROOT / 'cloudcare/faq.py'
    core_before = digest(core_path.read_bytes())
    old_preprocess = ROOT.parent / 'mysql_qa/utils/preprocess.py'
    old_before = digest(old_preprocess.read_bytes())
    knowledge, knowledge_raw = read_jsonl(ROOT / 'data/knowledge.jsonl')
    by_id = {row['id']: row for row in knowledge}
    faqs, faq_raw = read_jsonl(ROOT / 'data/faq.jsonl')
    faq_rows = [{**row, 'category': by_id[row['source_id']]['category']} for row in faqs]
    case_path = ROOT / 'evaluation/faq_bm25_20261002/frozen_cases.json'
    challenge_path = ROOT / 'evaluation/faq_bm25_20261002/challenge_cases.json'
    cases = json.loads(case_path.read_text(encoding='utf-8'))
    challenge = json.loads(challenge_path.read_text(encoding='utf-8'))['cases']
    assert len(cases) == 88 and len(challenge) == 20 and len(faq_rows) == 4736
    assert sum(row['partition'] == 'development' for row in cases) == 44
    assert sum(row['partition'] == 'retained' for row in cases) == 44
    variants = [('jieba_lcut_original', JiebaCutFAQIndex), ('jieba_lcut_for_search', JiebaSearchFAQIndex)]
    reports = {}
    selections = []
    for name, cls in variants:
        print(f'[tokenizer] {name}: question-only index, {len(faq_rows)} records', flush=True)
        index = cls(faq_rows)
        scored = [score(case, index) for case in cases]
        challenged = [score(case, index) for case in challenge]
        dev = [row for row in scored if row['partition'] == 'development']
        held = [row for row in scored if row['partition'] == 'retained']
        boundaries = [value for row in dev if row['score'] is not None
                      for value in [row['score'], math.nextafter(row['score'], math.inf)] if 0 < value < 1]
        thresholds = sorted(set([i / 100 for i in range(1, 100)] + boundaries))
        sweep = [stats(dev, threshold) for threshold in thresholds]
        strategies = []
        for target in [.95, .90, .80, .70, .50]:
            selected = choose(sweep, target)
            strategies.append(dict(target=target, require_no_development_negative_accepts=True,
                                   selected_development=selected,
                                   retained_at_selected=stats(held, selected['threshold']) if selected else None,
                                   challenge_at_selected=stats(challenged, selected['threshold']) if selected else None))
        primary = next(row for row in strategies if row['target'] == .90)
        if primary['selected_development']:
            selections.append((name, primary['selected_development']))
        reports[name] = dict(top1=dict(development=top1(dev), retained=top1(held),
                                      all_regression=top1(scored), challenge=top1(challenged)),
                             fixed_085=dict(development=stats(dev, .85), retained=stats(held, .85),
                                            challenge=stats(challenged, .85)),
                             threshold_count=len(thresholds), strict90=primary, strategies=strategies,
                             development_pareto=pareto(sweep),
                             score_distributions=dict(development=distribution(dev), retained=distribution(held),
                                                      challenge=distribution(challenged)))
        (OUT / (name + '_development_sweep.json')).write_text(json.dumps(sweep, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raw_rows = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in scored + challenged).encode('utf-8')
        (OUT / (name + '_results.jsonl.gz')).write_bytes(gzip.compress(raw_rows, mtime=0))
        print(json.dumps({'tokenizer':name, 'top1':reports[name]['top1'], 'strict90':primary}, ensure_ascii=True), flush=True)
    selected_variant = min(selections, key=lambda pair: (-pair[1]['strict_correct'], -pair[1]['strict_precision'],
                           pair[1]['threshold'], [name for name, _ in variants].index(pair[0]))) if selections else None
    selected_summary = dict(tokenizer=selected_variant[0], threshold=selected_variant[1]['threshold']) if selected_variant else None
    baseline, baseline_raw = read_jsonl(ROOT / 'evaluation/faq_softmax_20261002/results.jsonl')
    summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                   scope='FAQ tokenizer-only experiment; no core edits, data augmentation, neural models, databases or Qwen calls',
                   jieba_version=jieba.__version__, changed_bm25_parameters=False,
                   k1=FAQ_BM25_K1, b=FAQ_BM25_B, normalization='softmax_all_faq', faq_records=len(faq_rows),
                   protocol=dict(development=44, retained=44, challenge=20,
                                 thresholds='0.01..0.99 plus development score and nextafter(score,+inf)',
                                 primary_target=.90, require_no_development_negative_accepts=True,
                                 objective='maximize development correct positives, tie precision then lowest threshold',
                                 tried_word_tokenizers=2,
                                 segmentation_only='default Jieba; no custom dictionaries, stopword lists or query expansions'),
                   variants=reports, development_selected=selected_summary,
                   frozen_bigram_baseline_top1=dict(all_regression=top1([row for row in baseline if 'partition' in row]),
                       development=top1([row for row in baseline if row.get('partition') == 'development']),
                       retained=top1([row for row in baseline if row.get('partition') == 'retained']),
                       challenge=top1([row for row in baseline if 'partition' not in row])),
                   input_sha256={'faq.py':core_before, 'faq.jsonl':digest(faq_raw), 'knowledge.jsonl':digest(knowledge_raw),
                                 'frozen_cases.json':digest(case_path.read_bytes()),
                                 'challenge_cases.json':digest(challenge_path.read_bytes()),
                                 'original_preprocess.py':old_before,
                                 'frozen_bigram_results.jsonl':digest(baseline_raw)},
                   core_unchanged=core_before == digest(core_path.read_bytes()),
                   original_education_preprocess_unchanged=old_before == digest(old_preprocess.read_bytes()),
                   elapsed_seconds=time.perf_counter() - started,
                   limitations=['Prior synthetic development and retained material are not a blind external benchmark.',
                                'This is a targeted experiment after prior bad-score analysis; thresholds alone cannot fix missing FAQ coverage.',
                                'Source-ID correctness is a strict relevance proxy and does not prove answer completeness.',
                                'No negative relabeling or safe-denial credit is applied to strict selection.',
                                'Retained and challenge observations do not enter tokenizer/threshold selection.',
                                'Two predeclared variants are tried; no stopword tuning or question-derived FAQ generation.',
                                'The relative Softmax score changes with corpus/tokenization and is not calibrated correctness.',
                                'Whole-system BERT routing/RAG fallback can change final outcomes and is not run here.'])
    assert summary['core_unchanged'] and summary['original_education_preprocess_unchanged']
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'development_selected':selected_summary, 'elapsed_seconds':summary['elapsed_seconds']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
