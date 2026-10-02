"""Optional RAGAS 0.4.3 adapters. Runtime service does not import this module."""
from __future__ import annotations
import asyncio
import math
from ragas.embeddings.base import BaseRagasEmbedding


class BGERagasEmbedding(BaseRagasEmbedding):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def embed_text(self, text, **kwargs):
        return self.encoder.encode_queries([text])[0].dense

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text)


async def score_record(metrics, record):
    """Do not convert provider/parse/NaN errors to invented zero scores."""
    inputs = {
        'faithfulness': {k: record[k] for k in ('user_input','response','retrieved_contexts')},
        'answer_relevancy': {k: record[k] for k in ('user_input','response')},
        'context_precision': {k: record[k] for k in ('user_input','reference','retrieved_contexts')},
        'context_recall': {k: record[k] for k in ('user_input','reference','retrieved_contexts')},
    }
    result = {}
    for name, metric in metrics.items():
        try:
            score = float((await asyncio.wait_for(metric.ascore(**inputs[name]), timeout=180)).value)
            if not math.isfinite(score): raise ValueError('Nonfinite metric')
            result[name] = {'value': score, 'status': 'ok'}
        except Exception as exc:
            result[name] = {'value': None, 'status': 'error', 'error_type': type(exc).__name__}
    return result


def summarize_scores(records):
    summary = {}
    for name in ('faithfulness','answer_relevancy','context_precision','context_recall'):
        values = [r['metrics'][name]['value'] for r in records if r.get('metrics',{}).get(name,{}).get('status')=='ok']
        summary[name] = {'mean':sum(values)/len(values) if values else None,
                         'valid':len(values),'total':len(records),'errors':len(records)-len(values)}
    return summary
