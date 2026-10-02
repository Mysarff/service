"""Bounded query preparation. Generated search text is never answer evidence."""
from __future__ import annotations

import re
import unicodedata


def denoise_query(text: str) -> str:
    """Normalize presentation noise without deleting numbers or negations."""
    value = unicodedata.normalize('NFKC', text)
    value = re.sub(r'[\u200b-\u200d\ufeff]', '', value)
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)
    value = re.sub(r'\s+', ' ', value).strip()
    value = re.sub(r'([!?！？。])\1+', r'\1', value)
    # Only an isolated greeting/polite prefix; business text stays intact.
    value = re.sub(r'^(?:你好|您好|请问|麻烦问一下)[,，:：\s]+', '', value)
    return value or text.strip()


def protected_terms(text: str) -> list[str]:
    return re.findall(r'[A-Za-z0-9][A-Za-z0-9_./:-]*|不(?:能|要|是|允许)|未(?:开通|支付|收到)', text)


def explicit_subqueries(query: str) -> list[str]:
    """Conservative fallback for explicitly separated, independently asked clauses.

    Do not split simple punctuation or a single question into fragments. Keep
    the complete original query as the primary route regardless of this result.
    """
    parts = [part.strip(' ,，;；?？。') for part in re.split(r'[,，;；]|以及|另外', query)]
    if not 2 <= len(parts) <= 3:
        return []
    if not all(len(part) >= 6 and re.search(r'怎么|如何|什么|多少|是否|怎么办|该怎样',part) for part in parts):
        return []
    return parts


def validate_rewrite(original: str, rewritten: str) -> None:
    if not isinstance(rewritten, str) or not 1 <= len(rewritten.strip()) <= 2000:
        raise ValueError('Invalid rewritten query')
    if any(token.lower() not in rewritten.lower() for token in protected_terms(original)):
        raise ValueError('Rewrite changed a query constraint')


def search_routes(original: str, plan: dict, *, max_queries: int = 7,
                  max_subqueries: int = 3, hyde_enabled: bool = True) -> list[dict]:
    """Original/rewrite/subqueries use hybrid retrieval; keywords lexical; HyDE dense.

    Subqueries share a total RRF weight of one. More model-generated queries
    cannot increase that group's aggregate voting budget without limit.
    """
    routes = [{'kind': 'original', 'query': original, 'weight': 1.0}]
    seen = {original}

    def add(kind, text, weight):
        if isinstance(text, str) and text.strip() and text.strip() not in seen:
            text = text.strip()
            routes.append({'kind': kind, 'query': text, 'weight': weight})
            seen.add(text)

    add('rewrite', plan.get('query'), 1.0)
    subqueries = plan.get('subqueries', [])
    if not isinstance(subqueries, list):
        subqueries = []
    unique_subqueries = list(dict.fromkeys(q.strip() for q in subqueries
        if isinstance(q, str) and 1 <= len(q.strip()) <= 500))[:max_subqueries]
    for subquery in unique_subqueries:
        add('subquery', subquery, 1 / max(1, len(unique_subqueries)))
    keywords = plan.get('keywords', [])
    if isinstance(keywords, list):
        valid_keywords = [k.strip() for k in keywords
                          if isinstance(k, str) and 1 <= len(k.strip()) <= 32][:4]
        add('keywords', ' '.join(valid_keywords), .5)
    if hyde_enabled:
        add('hyde', plan.get('hyde_document'), .5)
    return routes[:max(1, min(max_queries, 7))]
