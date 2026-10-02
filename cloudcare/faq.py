"""Question-only BM25, normalized against a same-query reference score."""
from __future__ import annotations

from collections import Counter
import math
import hashlib
from typing import Any, Iterable
import unicodedata

from engine import terms


FAQ_BM25_ALGORITHM = "faq_bm25_query_reference_v4"
FAQ_BM25_K1 = 1.2
FAQ_BM25_B = 0.75


class FAQBM25Index:
    """Immutable question index rebuilt when the Redis FAQ version changes.

    Chinese character bigrams are tokens, as in the knowledge BM25 branch.
    BM25 uses their frequencies and corpus IDF; no Dice score is computed.
    Only the question is indexed, never the answer or knowledge body.
    """

    def __init__(self, rows: Iterable[dict[str, Any]], *, group_answers: bool = True, normalization: str = 'query_reference'):
        self.rows = tuple(dict(row) for row in rows)
        if normalization not in ('query_reference', 'softmax'):
            raise ValueError('Unsupported BM25 normalization')
        self.normalization = normalization
        self.group_answers = group_answers
        self.groups = tuple((row.get('source_id'), hashlib.sha256(str(row.get('answer', '')).encode()).hexdigest())
                            if group_answers else (row['id'],) for row in self.rows)
        self.bags = tuple(Counter(self.tokenize(row["question"])) for row in self.rows)
        self.lengths = tuple(sum(bag.values()) for bag in self.bags)
        self.avg_length = sum(self.lengths) / max(1, len(self.rows))
        self.postings: dict[str, list[int]] = {}
        for index, bag in enumerate(self.bags):
            for term in bag:
                self.postings.setdefault(term, []).append(index)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return terms(unicodedata.normalize("NFKC", str(text)))

    def search(self, query: str, category: str = "") -> dict[str, Any] | None:
        query_bag = Counter(self.tokenize(query))
        query_terms = set(query_bag)
        candidate_indices: set[int] = set()
        for term in query_terms:
            candidate_indices.update(self.postings.get(term, ()))
        ranked = []
        count = len(self.rows)
        for index in sorted(candidate_indices):
            row = self.rows[index]
            if category and row.get("category") != category:
                continue
            bag = self.bags[index]
            common = query_terms & bag.keys()
            length_norm = FAQ_BM25_K1 * (
                1 - FAQ_BM25_B
                + FAQ_BM25_B * self.lengths[index] / max(1, self.avg_length)
            )
            score = sum(
                math.log(1 + (count - len(self.postings[term]) + 0.5)
                         / (len(self.postings[term]) + 0.5))
                * bag[term] * (FAQ_BM25_K1 + 1) / (bag[term] + length_norm)
                for term in sorted(common)
            )
            ranked.append({**row, "score": score, "method": "bm25", "matches": len(common), '_group': self.groups[index]})
        if not ranked:
            return None
        ranked.sort(key=lambda row: (-row["score"], row["id"]))
        # Match the education system's whole-index Softmax convention. Every
        # eligible FAQ participates, including questions with no shared terms
        # (BM25 score 0). Normalizing only the retrieved candidate(s), or dividing
        # by the maximum score, would make weak singleton matches look certain.
        eligible_count = (len(self.rows) if not category else
                          sum(row.get("category") == category for row in self.rows))
        candidate_count = len(ranked)
        groups = {}
        for row in ranked:
            groups.setdefault(row['_group'], row)
        ranked = list(groups.values())
        eligible_groups = {group for row, group in zip(self.rows, self.groups)
                           if not category or row.get('category') == category}
        best = ranked[0]
        raw_score = best["score"]
        zero_count = len(eligible_groups) - len(ranked)
        denominator = math.fsum(math.exp(row["score"] - raw_score) for row in ranked)
        denominator += zero_count * math.exp(-raw_score)
        # Query-relative BM25: use the same corpus IDFs and length formula,
        # treating the query as a hypothetical perfectly matching question.
        # Unseen query terms still contribute to that reference, penalizing
        # partial matches. Clip because BM25 term saturation is not a cosine.
        reference_norm = FAQ_BM25_K1 * (1 - FAQ_BM25_B + FAQ_BM25_B * sum(query_bag.values()) / max(1, self.avg_length))
        reference_score = math.fsum(math.log(1 + (count - len(self.postings.get(term, ())) + .5)
            / (len(self.postings.get(term, ())) + .5)) * query_bag[term] * (FAQ_BM25_K1 + 1)
            / (query_bag[term] + reference_norm) for term in sorted(query_terms))
        normalized = min(1., max(0., raw_score / reference_score)) if reference_score else 0.
        return {**{key: value for key, value in best.items() if key != '_group'},
                "raw_score": raw_score, "reference_score": reference_score,
                "score": normalized if self.normalization == 'query_reference' else 1.0 / denominator,
                'softmax_diagnostic': 1.0 / denominator,
                "normalization": 'bm25_query_reference' if self.normalization == 'query_reference' else
                    ("softmax_answer_groups" if self.group_answers else "softmax_all_faq"),
                "candidate_count": candidate_count, 'candidate_group_count': len(ranked),
                "eligible_count": eligible_count, 'eligible_group_count': len(eligible_groups)}
