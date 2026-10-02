"""BM25 over FAQ questions; raw scores are not correctness probabilities."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable
import unicodedata

from engine import terms


FAQ_BM25_ALGORITHM = "faq_bm25_v1"
FAQ_BM25_K1 = 1.2
FAQ_BM25_B = 0.75


class FAQBM25Index:
    """Immutable question index rebuilt when the Redis FAQ version changes.

    Chinese character bigrams are tokens, as in the knowledge BM25 branch.
    BM25 uses their frequencies and corpus IDF; no Dice score is computed.
    Only the question is indexed, never the answer or knowledge body.
    """

    def __init__(self, rows: Iterable[dict[str, Any]]):
        self.rows = tuple(dict(row) for row in rows)
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
        query_terms = set(self.tokenize(query))
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
            ranked.append({**row, "score": score, "method": "bm25", "matches": len(common)})
        if not ranked:
            return None
        ranked.sort(key=lambda row: (-row["score"], row["id"]))
        return {**ranked[0], "candidate_count": len(ranked)}
