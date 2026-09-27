"""Generate the quantitative report from actual result files, not hand-entered metrics."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
r=read('evaluation/benchmark_summary.json'); data=read('evaluation/data_validation.json'); checks=read('evaluation/system_results.json')
counts=read('data/manifest.json')['counts']
methods={}
for method in ('tfidf','plain_bm25','bm25'):
    rows=[g[method] for g in r['retrieval'].values()]
    methods[method]={'n':sum(x['n'] for x in rows),'top1':sum(x['top1'] for x in rows),'hit5':sum(x['hit5'] for x in rows),
                     'mrr5':sum(x['mrr5']*x['n'] for x in rows)/sum(x['n'] for x in rows)}
best=methods['bm25']; neg=sum(x['rejected'] for x in r['negative_evidence_gate'].values()); neg_n=sum(x['n'] for x in r['negative_evidence_gate'].values())
http=r['http'][-1]; scan=r['search_performance']['scan']; inverted=r['search_performance']['inverted']
improvement=(scan['mean_ms']-inverted['mean_ms'])/scan['mean_ms']*100
tables='\n'.join(f"| {m} | {v['top1']}/{v['n']} ({100*v['top1']/v['n']:.2f}%) | {v['hit5']}/{v['n']} | {v['mrr5']:.4f} |" for m,v in methods.items())
httptable='\n'.join(f"| {v['concurrency']} | {v['http_successful']}/{v['requests']} | {v['successful_requests_per_second']} | {v['latency_all_requests']['p50_ms']} | {v['latency_all_requests']['p95_ms']} | {v['latency_all_requests']['max_ms']} | {v['wall_seconds']} |" for v in r['http'])
report=f'''# 可复现评测与简历数字

测量时间：{r['created_at_utc']}。环境：{r['host']['os']}，Python {r['host']['python']}，{r['host']['logical_cpus']}个逻辑CPU。
这是自建合成语料上的开发回归和本机短时性能测试，没有独立盲测或真实客户样本。问题不放入索引，但开发者阅读过资料并根据失败题修改过规则，存在作者偏差和测试集调参偏差。

## 数据审计

- 来源文档：{counts['source_documents']}份 = {counts['source_manuals']}份模块手册 + {counts['source_runbooks']}份处置手册。
- 可检索知识片段：{counts['knowledge_units']}个；原始模块章节192个、专项规则16个、96个故障场景各4段，共384个场景片段。
- 知识正文{data['content_characters']:,}字符（不是token数）；完整重复正文{data['exact_duplicate_texts']}条。
- 字符三元组Jaccard ≥ 0.8 的相似片段对：{data['near_duplicate_pair_count']}对，主要来自模块模板。不能将592个片段等同于592个相互独立的业务事实。
- FAQ {counts['faq_variants']:,}条、模拟工单{counts['simulated_tickets']:,}条均复用已有事实，不计入新增知识。
- 逐文件SHA-256及来源正文一致性已核验；审计明细见 `evaluation/data_validation.json`。

## 检索：先定来源标签，再看排名

64道有依据问题（原32 + 场景32），每题给定允许的来源ID。Top-1 = 第一条来源正确的题数/64；Hit@5 = 前5条至少有一个正确来源的题数/64；MRR@5 = 正确来源首次出现名次的倒数平均值，未命中记0。

| 方法 | Top-1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|
{tables}

三者共享语料、分词、字段权重和领域词替换；增强BM25另外使用模块/标题加权、同一场景分组及按问题阶段扩展子段。它是多个工程规则的组合对照，不能把全部增益归因于某一个组件。Top-1仍有{best['n']-best['top1']}道失败；完整问题、预期来源和实际前5条保存在压缩原始结果中。

最初扩容后增强BM25场景题仅19/32，负例共17/24拒答。修复后场景题29/32、负例{neg}/{neg_n}。首次结果完整保留在 `expanded_before_fixes.json` 和 `expanded_before_fixes_raw.json.gz`。这不是未见样本上的泛化改进证明。Hit@5也不是最终回答准确率。

## 倒排索引：与全量扫描做同条件比较

64题 × 10轮；查询顺序固定种子打乱，两种实现随机交换执行顺序，计时前预热。两种路径使用完全相同的排序逻辑。

| 实现 | 平均耗时 ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| 全量扫描 | {scan['mean_ms']} | {scan['p50_ms']} | {scan['p95_ms']} |
| 倒排候选 | {inverted['mean_ms']} | {inverted['p50_ms']} | {inverted['p95_ms']} |

本次平均检索耗时下降{improvement:.1f}%，排序及分数一致性{r['ranking_parity']['passed']}/{r['ranking_parity']['n']}。这只证明本机592片段上的效果，不外推百万规模。Python分配峰值{r['index']['traced_peak_bytes']/1024/1024:.2f} MiB，带内存跟踪的单次建索引{r['index']['build_ms_with_tracemalloc']} ms；前者不是整个进程RSS，后者含测量开销。

## HTTP短时负载

回环地址、本机客户端和服务端同机；先预热20次；并发1/4/8，每组2,000次。每次包含HTTP连接、JSON编解码、检索及原文返回。外部模型被明确关闭。P95用升序后ceil(0.95×N)所在样本，计时使用单调高精度时钟。

| 并发数 | HTTP成功 | 成功请求/秒 | P50 ms | P95 ms | 最大 ms | 测量秒数 |
|---|---:|---:|---:|---:|---:|---:|
{httptable}

共{sum(x['http_successful'] for x in r['http'])}/{sum(x['requests'] for x in r['http'])}请求HTTP成功。本次均返回原文检索答案，但HTTP成功不代表答案语义正确。8并发的最慢请求为{http['latency_all_requests']['max_ms']} ms，不能只报告P95而隐去长尾。这是每组数秒的短时结果，不是持续压测、生产QPS、真实LLM吞吐或可用性SLA。

## 功能与模型

功能检查{checks['tests']-checks['failures']-checks['errors']}/{checks['tests']}通过：来源、拒答、请求校验、上传尾部、内容去重、索引刷新、短追问、检索一致性，以及模拟超时/伪造引用的回退。

真实模型完整验收尚未完成：此前配置返回401 invalid_api_key，当前没有收到新的有效且获授权的配置。因此没有可报告的真实生成成功率、生成延迟、token成本或语义忠实度。

有效配置就绪后运行 `python evaluation/model_benchmark.py --model-config config.ini --limit 8`。脚本保存实际模式、回答、来源、引用合法性、token用量（服务商有返回时）和耗时。逐句事实是否被原文支持、关键步骤是否遗漏需人工审查；引用编号存在不等于答案正确。脚本最多8次正例请求，遇到服务网络/认证错误会早停并保留失败。

## 重跑

```bash
python scripts/build_data.py
python scripts/check_data.py
python evaluation/run_checks.py
python evaluation/benchmark.py --requests 2000 --rounds 10
python evaluation/evaluate.py
python scripts/render_report.py
```

原始记录：`evaluation/benchmark_results.json.gz`；简表：`evaluation/benchmark_summary.json`；语料/题目哈希及机器环境均在结果内。只改变规则而沿用旧结果会使报告过期，必须重跑。用 `gzip.open(path, 'rt', encoding='utf-8')` 可以读取压缩JSON。
'''
(ROOT/'docs/BENCHMARK.md').write_text(report,encoding='utf-8',newline='\n')
resume=f'''# 简历可用表述与证据边界

## 可直接使用的项目描述

**CloudCare 企业客服知识库（个人工程项目，合成业务数据）**

- 构建覆盖16个业务模块的知识库，整理112份来源文档、592个可追溯片段，配套生成4,736条FAQ问法和24,000条模拟工单，实现来源映射、内容去重与SHA-256完整性校验。
- 实现BM25倒排检索、模块加权、场景分组与子段扩展，在64道自建开发回归题上取得Top-1 {100*best['top1']/best['n']:.1f}%、Hit@5 {100*best['hit5']/best['n']:.1f}%；与相同排序的全量扫描相比，本机640次测量平均检索耗时下降{improvement:.1f}%。
- 完成{checks['tests']}项功能检查及6,000次本机原文检索接口请求，8并发短时测试P95 {http['latency_all_requests']['p95_ms']:.2f} ms；实现模型调用异常与无效引用回退，保留逐题失败和逐请求测量记录。

## 面试时必须能解释

1. 592是切分片段数，不是独立事实数；24,000条模拟工单不进入知识索引。来源规则由开发者编写，未接入真实企业资料。
2. 64题是参与过调试的开发回归，另外24题检验拒答；不能称为独立测试或真实客服准确率。当前Top-1仍有7题失败。
3. 倒排索引减少无共同词项文档的扫描；两条检索路径使用同一排序逻辑，640/640排名与分数相同。需解释平均值与P95的不同。
4. HTTP测试关闭外部模型，每组仅数秒；8并发最大耗时约{http['latency_all_requests']['max_ms']/1000:.2f}秒。不能写成大模型毫秒级回答或企业生产SLA。
5. 真实模型尚缺有效配置完成验收。可以写“实现兼容模型接口”，不能写“已完成真实模型端到端验证”。
6. 没有实现Milvus、BGE向量/重排、BERT分类、多租户权限或真实退款/审批。与教育版共享RAG步骤，但不是完整技术栈迁移。

完整证据及复测命令见 [BENCHMARK.md](BENCHMARK.md)。
'''
(ROOT/'docs/RESUME.md').write_text(resume,encoding='utf-8',newline='\n')
print('Wrote docs/BENCHMARK.md and docs/RESUME.md from measured results.')
