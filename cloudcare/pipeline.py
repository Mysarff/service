"""Customer-support routing, real retrieval, persistence and grounded generation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from langchain_core.runnables import RunnableLambda, RunnableParallel

from engine import Engine, terms
from .documents import DocumentProcessor
from .llm import QwenGateway
from .neural import BGEReranker, SupportBertRouter
from .retrieval import MilvusStore
from .settings import Settings
from .storage import MySQLStore, RedisStore
from .query import denoise_query, search_routes


class BM25Index:
    """A lexical control and complementary exact-business-term retrieval branch."""
    def __init__(self, rows):
        self.rows = rows
        self.bags = [Counter(terms(row['content']) + terms(row['title']) * 2) for row in rows]
        self.df = Counter(word for bag in self.bags for word in bag)
        self.lengths = [sum(bag.values()) for bag in self.bags]
        self.avg = sum(self.lengths) / max(1, len(rows))

    def search(self, query, category='', limit=20):
        words = set(terms(query)); result = []
        for row, bag, size in zip(self.rows, self.bags, self.lengths):
            if category and row['category'] != category:
                continue
            common = words & bag.keys()
            if not common:
                continue
            score = sum(math.log(1 + (len(self.rows) - self.df[word] + .5) / (self.df[word] + .5))
                        * bag[word] * 2.2 / (bag[word] + 1.2 * (.25 + .75 * size / max(1, self.avg)))
                        for word in common)
            result.append({**row, 'score': score, 'retrieval_method': 'bm25', 'matches': len(common)})
        return sorted(result, key=lambda row: (-row['score'], row['id']))[:limit]


class SupportPipeline:
    def __init__(self, settings=None):
        self.settings = settings or Settings.load()
        self.sql = MySQLStore(self.settings)
        self.redis = RedisStore(self.settings)
        self.vector = MilvusStore(self.settings)
        self.reranker = BGEReranker(self.settings)
        self.router = SupportBertRouter(self.settings)
        self.llm = QwenGateway(self.settings)
        self.documents = DocumentProcessor(self.settings)
        self._lock = threading.RLock()
        self._ingest_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._sessions_lock = threading.RLock()
        self._revision = ''
        self._started = False
        self._legacy = Engine()
        self._legacy.key = ''  # Only context/gap rules; never use its optional provider.
        self._cache_config = hashlib.sha256(json.dumps({
            'model': self.settings.llm_model, 'endpoint': self.settings.llm_base_url,
            'rewrite': self.settings.query_rewrite_enabled,
            'query_expansion': self.settings.query_expansion_enabled,
            'hyde': self.settings.hyde_enabled,
            'query_budget': [self.settings.max_subqueries, self.settings.max_retrieval_queries],
            'faq_bm25_threshold': self.settings.faq_bm25_threshold,
            'embedding': str(self.settings.embedding_model_path),
            'reranker': str(self.settings.reranker_model_path),
            'code': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ('pipeline.py','llm.py','query.py','neural.py','retrieval.py','faq.py','storage.py')},
            'bert_artifact': self.router.status().get('dataset_sha256'),
            'bert_checkpoint': self.router.status().get('checkpoint_sha256'),
        }, sort_keys=True).encode()).hexdigest()
        self.bm25 = BM25Index([])
        self.retrieval_chain = RunnableParallel(
            neural=RunnableLambda(lambda value: self.vector.hybrid_search(value['query'],
                category=value['category'], tenant_id='demo', visibility='public', limit=self.settings.retrieval_k)),
            lexical=RunnableLambda(lambda value: self.bm25.search(value['query'], value['category'], self.settings.retrieval_k)))

    def start(self):
        with self._lock:
            if self._started:
                return
            self.sql.initialize()
            self.sql.health(); self.redis.health()
            self.vector.ensure_collection()
            self.refresh()
            self._started = True

    def refresh(self):
        rows = self.sql.list_chunks(tenant_id='demo', visibility='public')
        self.bm25 = BM25Index(rows)
        self._revision = hashlib.sha256(json.dumps(
            [{key: row.get(key) for key in ('id','content_hash','title','category','version','tags',
              'source','parent_id','parent_content','tenant_id','visibility','synthetic')}
              for row in rows], sort_keys=True).encode()).hexdigest()

    def health(self):
        components = {}
        counts = {}
        for name, probe in [('mysql', self.sql.health), ('redis', self.redis.health)]:
            try:
                components[name] = {'ready': True, **probe()}
            except Exception as exc:
                components[name] = {'ready': False, 'error_type': type(exc).__name__}
        try:
            counts = self.sql.counts()
        except Exception:
            pass
        try:
            milvus_count = self.vector.count(tenant_id='demo')
            components['milvus'] = {'ready': milvus_count == counts.get('chunks', -1) and milvus_count > 0,
                                    'indexed_rows': milvus_count, 'collection': self.settings.milvus_collection}
        except Exception as exc:
            milvus_count = None
            components['milvus'] = {'ready': False, 'error_type': type(exc).__name__}
        components.update(bert=self.router.status(), bge_m3=self.vector.encoder.status(),
                          bge_reranker=self.reranker.status(), qwen=self.llm.status())
        manifest = json.loads((self.settings.root / 'data/manifest.json').read_text(encoding='utf-8'))
        return {'status': 'ok' if all(components[name]['ready'] for name in ('mysql','redis','milvus')) else 'degraded',
                'knowledge_units': counts.get('chunks', 0), 'milvus_rows': milvus_count, 'milvus_units': milvus_count,
                'counts': {'source_documents': counts.get('documents',0), 'knowledge_units': counts.get('chunks',0),
                           'faq_variants': counts.get('faq',0),
                           'simulated_tickets': manifest['counts']['simulated_tickets'],
                           'local_tickets': counts.get('tickets',0)},
                'categories': sorted({row['category'] for row in self.bm25.rows}),
                'model_enabled': self.settings.llm_configured, 'model': self.settings.llm_model,
                'model_state': ('verified' if self.llm.successful_calls else 'configured_unverified') if self.settings.llm_configured else 'unconfigured',
                'retriever': 'BGE-M3 dense + learned sparse / BM25 / BGE-Reranker',
                'components': components, 'synthetic': all(row.get('synthetic') for row in self.bm25.rows),
                'synthetic_units': sum(bool(row.get('synthetic')) for row in self.bm25.rows),
                'model_memory_policy': self.settings.model_memory_policy,
                'corpus_revision': self._revision}

    def knowledge(self):
        return {'items': self.sql.list_chunks(tenant_id='demo', visibility='public')}

    def retrieve(self, query, category='', additional_queries=None, query_plan=None):
        with self._inference_lock:
            try:
                return self._retrieve_serial(query, category, additional_queries, query_plan)
            except Exception:
                if self.settings.model_memory_policy == 'sequential':
                    for component in (self.router, self.vector.encoder, self.reranker):
                        self._release(component)
                raise

    @staticmethod
    def _release(component):
        release = getattr(component, 'release_model', None)
        if release:
            release()

    def _predict_route(self, query):
        with self._inference_lock:
            if self.settings.model_memory_policy == 'sequential':
                self._release(self.vector.encoder); self._release(self.reranker)
            try:
                return self.router.predict(query)
            finally:
                if self.settings.model_memory_policy == 'sequential':
                    self._release(self.router)

    def _retrieve_serial(self, query, category='', additional_queries=None, query_plan=None):
        """Filter all branches before RRF union and true cross-encoder reranking."""
        if self.settings.model_memory_policy == 'sequential':
            self._release(self.router); self._release(self.reranker)
        if query_plan is not None:
            routes = search_routes(query, query_plan,
                max_queries=self.settings.max_retrieval_queries,
                max_subqueries=self.settings.max_subqueries, hyde_enabled=self.settings.hyde_enabled)
        else:
            routes = [{'kind': 'original' if i == 0 else 'rewrite', 'query': value, 'weight': 1.0}
                      for i, value in enumerate(list(dict.fromkeys([query, *(additional_queries or [])]))[:2])]
        queries = [route['query'] for route in routes]
        merged = {}; route_counts = []; unpublished_skipped = 0
        for route in routes:
            value = route['query']
            if route['kind'] == 'hyde':
                branches = {'hyde_dense': self.vector.dense_search(value, category=category,
                    tenant_id='demo', visibility='public', limit=self.settings.retrieval_k)}
            elif route['kind'] == 'keywords':
                branches = {'keywords_bm25': self.bm25.search(value, category, self.settings.retrieval_k)}
            else:
                branches = self.retrieval_chain.invoke({'query': value, 'category': category})
            route_counts.append({'kind': route['kind'], 'weight': route['weight'],
                                 **{name: len(rows) for name, rows in branches.items()}})
            for branch, rows in branches.items():
                for rank, row in enumerate(rows,1):
                    identifier = row['id']
                    if identifier not in merged:
                        canonical = self.sql.get_chunk(identifier)
                        if not canonical:
                            unpublished_skipped += 1
                            continue
                        if canonical['tenant_id'] != 'demo' or canonical['visibility'] != 'public':
                            raise ValueError('Retrieved source is outside the demo corpus')
                        if category and canonical['category'] != category:
                            raise ValueError('Retrieved source is outside the requested module')
                        if row.get('content_hash') and row['content_hash'] != canonical['content_hash']:
                            raise ValueError('Milvus content is stale relative to MySQL')
                        merged[identifier] = {**canonical, 'fusion_score': 0., 'branches': []}
                    hit = merged[identifier]
                    hit['fusion_score'] += route['weight'] / (60 + rank)
                    hit['branches'].append(branch)
        candidates = sorted(merged.values(),key=lambda row: (-row['fusion_score'],row['id']))[:self.settings.retrieval_k]
        if self.settings.model_memory_policy == 'sequential':
            self._release(self.vector.encoder)
        ranked = self.reranker.rerank(query, candidates)
        for hit in ranked:
            parent = self.sql.get_parent(hit['parent_id'])
            if parent:
                if parent.get('document_id') != hit.get('document_id'):
                    raise ValueError('Parent context belongs to another source document')
                hit['parent_content'] = parent['content']
        return ranked, {'queries': queries, 'query_routes': routes, 'branch_candidates': route_counts,
                        'merged_candidates': len(candidates), 'reranked': len(ranked),
                        'unpublished_skipped': unpublished_skipped,
                        'fusion': 'Milvus WeightedRanker dense/sparse + BM25 RRF(k=60)',
                        'reranker': 'BGE-Reranker cross-encoder'}

    def _history(self, session_id, supplied):
        if session_id:
            saved = self.redis.get_session(session_id)
            if saved is None:
                saved = self.sql.get_session(session_id, limit=8)
            if saved:
                return [{'role': row['role'], 'content': row['content'][:2000]} for row in saved][-8:]
            return []
        return supplied or []

    def answer(self, query, category='', session_id=None, history=None, force_extract=False):
        self.start()
        start = time.perf_counter()
        if session_id and not re.fullmatch(r'[a-f0-9]{32}', session_id):
            raise ValueError('Invalid session identifier')
        history = self._history(session_id, history)
        session_id = session_id or uuid.uuid4().hex
        cleaned = denoise_query(query)
        contextual = self._legacy.contextual_query(cleaned, history)
        trace = {'original_query': query, 'cleaned_query': cleaned,
                 'contextual_query': contextual, 'cache_hit': False}
        # Run actual BERT even when a capability rule or FAQ will select the final route.
        route = self._predict_route(contextual)
        trace['bert'] = route
        gap = self._legacy.capability_gap(contextual)
        result = None
        if gap:
            result = {'answer': gap, 'sources': [], 'mode': 'insufficient_evidence'}
        elif re.fullmatch(r'(你好|您好|hi|hello)[！!。\s]*',query,re.I):
            result = {'answer':'您好，我是云栈客服助手。请描述业务模块和遇到的问题。','sources':[], 'mode':'greeting'}
        elif route['intent'] == 'handoff' and route['confidence'] >= .9 and re.search(r'人工|真人|投诉|客服人员', query):
            result = {'answer': '可以提交人工工单，请描述问题并提供脱敏信息。此项目创建演示工单，尚未连接真人客服。',
                      'sources': [], 'mode': 'handoff'}
        elif route['intent'] == 'out_of_scope' and route['confidence'] >= .9:
            lexical_hint = self.bm25.search(contextual, category, 1)
            if not lexical_hint or lexical_hint[0]['matches'] < 2:
                result = {'answer':'当前企业客服资料不覆盖这个问题，请咨询对应专业人员。',
                          'sources': [], 'mode':'insufficient_evidence'}
        faq_version = self.redis.faq_index_version()
        cache_key = hashlib.sha256(json.dumps({'q':contextual,'category':category,'history':history,
            'revision':self._revision,'config':self._cache_config,
             'faq_version':faq_version,'faq_bm25_threshold':self.settings.faq_bm25_threshold,
            'model':self.settings.llm_model,'generation':self.settings.llm_configured and not force_extract},
            sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if result is None:
            cached = self.redis.get_cached_answer(cache_key)
            if cached is not None:
                result = cached
                trace['cache_hit'] = True
        if result is None:
            faq = self.redis.find_faq(contextual, category=category, min_score=0.0,
                                      index_version=faq_version)
            threshold = self.settings.faq_bm25_threshold
            trace['faq'] = {'method':'bm25', 'score':faq['score'] if faq else None,
                            'raw_score':faq.get('raw_score') if faq else None,
                            'normalization':faq.get('normalization', 'bm25_query_reference') if faq else 'bm25_query_reference',
                            'reference_score':faq.get('reference_score') if faq else None,
                            'threshold':threshold, 'accepted':False,
                            'reason':'below_threshold' if faq else 'no_candidate',
                            'index_version':faq_version,
                            'candidate_count':faq.get('candidate_count',0) if faq else 0,
                            'eligible_count':faq.get('eligible_count',0) if faq else 0}
            if faq:
                trace['faq'].update({key:faq.get(key) for key in ('id','question','source_id','matches')})
            if faq and faq['score'] >= threshold:
                source = self.sql.get_chunk(faq.get('source_id',''))
                if source and source['tenant_id']=='demo' and source['visibility']=='public' and (not category or source['category']==category):
                    if isinstance(faq.get('answer'),str) and faq['answer'].strip():
                        result = {'answer': f"{faq['answer']} [{source['id']}]", 'sources':[source], 'mode':'faq'}
                        trace['faq'].update(accepted=True, reason='accepted')
                    else:
                        trace['faq']['reason'] = 'empty_answer'
                else:
                    trace['faq']['reason'] = 'invalid_source'
        if result is None:
            rewritten = contextual
            query_plan = None
            trace['rewrite'] = {'method':'context_rules','query':contextual}
            if self.settings.llm_configured and self.settings.query_rewrite_enabled and not force_extract:
                try:
                    if self.settings.query_expansion_enabled:
                        query_plan = self.llm.plan(contextual, history)
                        trace['rewrite'] = query_plan
                    else:
                        trace['rewrite'] = self.llm.rewrite(contextual,history)
                    rewritten = trace['rewrite']['query']
                except Exception as exc:
                    trace['rewrite']['error_type'] = type(exc).__name__
            extra = [rewritten] if rewritten != contextual else []
            hits, retrieval_trace = self.retrieve(contextual,category,extra,query_plan=query_plan)
            trace['retrieval'] = retrieval_trace
            lexical = self.bm25.search(contextual,category,5)
            # Reranker score is a heuristic, never a calibrated correctness probability.
            has_terms = bool(lexical and lexical[0]['matches'] >= 2)
            sufficient = bool(hits and hits[0]['rerank_score'] >= .01 and has_terms)
            if not sufficient:
                result = {'answer':'当前资料不足以回答。请补充业务模块、操作步骤或脱敏错误信息，必要时提交人工工单。',
                          'sources':[], 'mode':'insufficient_evidence'}
            else:
                answer = f"根据当前演示资料：\n{hits[0]['content']} [{hits[0]['id']}]"
                result = {'answer':answer,'sources':hits,'mode':'retrieval_only'}
                if self.settings.llm_configured and not force_extract:
                    try:
                        result['answer'] = self.llm.generate(contextual,hits,history)
                        result['mode'] = 'grounded_llm'
                        result['model'] = self.settings.llm_model
                    except Exception as exc:
                        result['mode'] = 'retrieval_fallback'
                        result['notice'] = '模型调用或引用校验失败，当前展示检索到的原文。'
                        trace['generation_error_type'] = type(exc).__name__
        result = json.loads(json.dumps(result, ensure_ascii=False, default=str))
        if not trace['cache_hit'] and result['mode'] in ('faq','retrieval_only','grounded_llm'):
            self.redis.set_cached_answer(cache_key,result,self.settings.cache_ttl_seconds)
        with self._sessions_lock:
            self.sql.record_message(session_id,'user',query,{'category':category})
            self.sql.record_message(session_id,'assistant',result['answer'],{'mode':result['mode'],'source_ids':[s['id'] for s in result['sources']]})
            next_history = self._legacy.next_history(query,result['answer'],history)
            self.redis.set_session(session_id,next_history,self.settings.session_ttl_seconds)
        return {**result,'trace':trace,'session_id':session_id,'history':next_history,
                'elapsed_ms':round((time.perf_counter()-start)*1000,2)}

    def ingest(self, filename, raw: bytes, category='uploaded'):
        with self._ingest_lock:
            return self._ingest_serial(filename, raw, category)

    def _ingest_serial(self, filename, raw: bytes, category='uploaded'):
        self.start()
        if Path(filename).name != filename or not raw or len(raw)>self.settings.max_upload_bytes:
            raise ValueError('Invalid filename or upload size')
        temporary_dir = self.settings.runtime_dir/'uploads'; temporary_dir.mkdir(parents=True,exist_ok=True)
        path = temporary_dir/(uuid.uuid4().hex+Path(filename).suffix.lower())
        path.write_bytes(raw)
        try:
            processed = self.documents.ingest_path(path,category=category,source_name=filename)
            existing = self.sql.document_by_hash(processed.sha256)
            if existing:
                return {'document_id':existing['id'],'duplicate':True,'knowledge_units':existing['chunk_count']}
            # Index before publication. A failed vector write never advertises an imported document.
            try:
                with self._inference_lock:
                    if self.settings.model_memory_policy == 'sequential':
                        self._release(self.router); self._release(self.reranker)
                    try:
                        self.vector.upsert_documents(processed.chunks)
                    finally:
                        if self.settings.model_memory_policy == 'sequential':
                            self._release(self.vector.encoder)
                self.sql.publish_ingested_document(processed.document_record(), processed.parents, processed.chunks)
            except Exception:
                # Only IDs from this unpublished upload; no collection/database-wide cleanup.
                self.vector.delete_documents([row['id'] for row in processed.chunks], tenant_id='demo')
                raise
            with self._lock:
                self.refresh()
            return {'document_id':processed.id,'duplicate':False,'knowledge_units':len(processed.chunks),
                    'parents':len(processed.parents),'metadata':processed.metadata}
        finally:
            path.unlink(missing_ok=True)

    def create_ticket(self, summary):
        self.start()
        row = {'id':'CC-'+uuid.uuid4().hex[:12].upper(),'summary':summary,'status':'待处理','local_only':True}
        self.sql.create_ticket(row)
        return {'ticket':row,'message':'已存入本地演示工单库，尚未连接真人客服。'}

    def close(self):
        self.sql.close(); self.redis.close(); self.llm.close()
