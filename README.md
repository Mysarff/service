# CloudCare 云栈——企业客服 RAG 知识库问答系统

面向账号权限、订单售后、订阅账单、工单和开放接口等 **16 类业务知识主题**，支持多格式知识导入、连续咨询、带出处回答与客服演示工单。基于原教育 RAG 的技术链路进行客服业务改造，默认入口使用真实数据库、本地神经检索和模型重排。

仓库：[Mysarff/service](https://github.com/Mysarff/service)。企业、产品政策、手册、FAQ 和模拟工单均为**合成演示资料**；没有真实企业上线或客服成本下降的证明。

## 技术栈与功能

| 技术 | 当前代码中的实际作用 |
|---|---|
| Python、FastAPI、Uvicorn | HTTP 服务、严格请求模型、线程池执行推理与数据库访问 |
| LangChain | Document 与递归父子分块、并行检索链、Query 改写及证据回答链 |
| BERT | 使用独立客服分类头预测知识咨询、人工跟进和服务范围外 |
| BGE-M3 | 本地生成 1024 维稠密向量与学习得到的稀疏词项权重 |
| Milvus | 独立集合保存真实向量，稠密／稀疏双路召回及加权融合 |
| BM25、RRF | 补充精确词项召回，融合神经与词项候选 |
| BGE-Reranker | 对候选进行真实 CrossEncoder 二次排序 |
| MySQL、SQLAlchemy、PyMySQL | 连接池和短事务，保存文档、父子知识、FAQ、会话和工单 |
| Redis | 答案及会话缓存、FAQ记录与发布版本；各工作进程按版本构建本地FAQ BM25快照 |
| RapidOCR、ONNX Runtime | 识别图片、扫描 PDF 及 Office 嵌入图片中的文字 |
| Qwen API | 兼容接口调用：Query 改写、选择证据步骤并校验逐字引用 |
| HTML、CSS、JavaScript | 客服对话、多格式上传、出处浏览与实际服务状态展示 |

代码存在、包已安装、模型配置齐全、真实推理通过、数据库实际入库和端到端验收分别记录。服务或模型不可用会明确报错或显示原文回退，不用内存字典冒充 MySQL/Redis，也不用规则分数冒充神经向量或重排。

## 启动全栈版本

使用 Python 3.11 和可用的 Docker Engine。Windows 在仓库目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -CreateEnv -CreateVenv -InstallDependencies -StartServices -SkipDoctor
.venv\Scripts\python.exe scripts/train_support_router.py --device auto
.venv\Scripts\python.exe scripts/initialize.py
.venv\Scripts\python.exe scripts/doctor.py --probe-models --probe-ocr --probe-llm --output evaluation/fullstack_doctor.json
.venv\Scripts\python.exe app.py
```

打开 <http://127.0.0.1:8088/>，也可以运行 `start.ps1`。首次运行需要准备 BGE-M3、BGE-Reranker 和中文 BERT 权重；模型目录可以用环境变量指定。已有教育项目模型只作只读基础权重，客服训练结果写入本项目 `runtime/models/support-router/`，不提交 Git。

默认有可用 CUDA 时在 GPU 运行 BGE 和 BERT，没有 CUDA 时使用 CPU。首次安装先跳过完整诊断，训练、入库及本地 Qwen 配置完成后再进行验收。需要 CUDA 12.6 环境时给安装脚本增加 `-TorchRuntime cu126`。默认依次加载、释放模型以适配本机内存，实测延迟包含重载成本。部署及镜像构建见 [DEPLOYMENT.md](docs/DEPLOYMENT.md)，格式与诊断见 [FULLSTACK.md](docs/FULLSTACK.md)。

### 独立存储

| 服务 | 默认位置 | 独立命名空间 |
|---|---|---|
| MySQL | `127.0.0.1:3308` | `cloudcare_support` |
| Redis | `127.0.0.1:6381/0` | `cloudcare:v2:` |
| Milvus | `127.0.0.1:19531` | `cloudcare_support_v2` |

`compose.yaml` 使用本项目容器、网络和数据卷。初始化创建缺失表、写入本项目数据，不 DROP 教育表、不清空共享 Redis、不删除教育向量集合。MySQL 表为 `documents`、`knowledge_parents`、`knowledge_chunks`、`faq`、`sessions`、`messages`、`tickets`。

### Qwen 与本地配置

`scripts/bootstrap.ps1 -CreateEnv` 生成忽略提交的 `.env`，保存独立数据库密码。Qwen 使用 `CLOUDCARE_QWEN_API_KEY`、`CLOUDCARE_QWEN_BASE_URL`、`CLOUDCARE_QWEN_MODEL`，也支持客服项目自己的 `config.ini`；默认示例为 DashScope 兼容接口。密钥不得加入手册、源码或 Git。

真实外部调用会发送问题、近期对话和选中的知识片段，并可能产生费用。`scripts/doctor.py --probe-llm` 检查实际调用；网页和回答区分未配置、已配置未验证、调用成功、生成失败回退等状态。Qwen 选择原文步骤，服务校验每步引用 ID 和原文连续子串，拒绝没有依据的补写；这仍不代表资料本身真实或答案业务准确率已验证。

`app.py --offline` 关闭外部模型调用，仍要求真实数据库、向量索引和本地神经模型可用。

## 数据规模与知识单元

| 数据 | 样例数量 | 用途 |
|---|---:|---|
| 模块操作手册 | 16 份 | 可阅读业务来源 |
| 故障处置手册 | 96 份 | 每类主题 6 个场景，来源文档共 112 份 |
| 来源父块 | 112 个 | 样例按来源保留完整上下文 |
| 可检索子片段／知识单元 | **592 条** | 208 个模块章节及专项规则＋384 个场景阶段片段 |
| FAQ 问法 | 4,736 条 | 每个知识片段 8 种模板表达 |
| 模拟会话工单 | 24,000 条 | 演示会话记录，不进入知识索引 |
| 开发回归题 | 88 道 | 64 道有来源问题＋24 道无依据／越界题 |

**一个知识单元是一条带出处的检索子片段。**每条 Milvus 记录有稠密和稀疏两种向量表示，592 条记录仍是 592 个知识单元，不能加成 1,184 个独立知识。FAQ 和模拟工单复用这些资料，也不能累计成更多独立知识。独立业务事实数没有人工统计，资料规模属于个人工程演示。

实际 MySQL 和 Milvus 行数从服务器查询，初始化结果保存在 [fullstack_index.json](evaluation/fullstack_index.json)。新上传资料会增加实时数量；“知识库概况”展示库存，不代表准确率或大规模系统能力。

数据来源为 `scripts/build_data.py` 与 `scripts/scenarios.py`，`data/manifest.json` 保存数量、版本、种子和 SHA-256。FAQ 与模拟工单以完整 `.jsonl.gz` 提交，初始化直接支持解压读取。数据定义见 [DATASET.md](docs/DATASET.md)。

## 检索与回答流程

```text
新文档／图片 → 文本解析或 OCR → LangChain 父子分块
             → BGE-M3 编码 → Milvus 写入 → MySQL 单事务发布来源与父子记录
样例初始化：种子文件 → MySQL／Redis → BGE-M3／Milvus → 实际数量与哈希核验

问题＋服务端会话 → 上下文补全＋BERT 路由 → 能力边界与 Redis 答案缓存
                → FAQ BM25：Top-1达阈值且出处有效则直答，否则继续RAG
                → 可选 Qwen Query 改写
                → LangChain 并行：Milvus 稠密／稀疏混合召回＋BM25
                → RRF 候选融合 → BGE-Reranker → MySQL 来源与父块核验
                → 证据门槛 → Qwen 选择并引用原文步骤／原文回退
                → MySQL 消息与工单持久化＋Redis 会话与答案缓存
```

FAQ问题使用BM25（k1=1.2、b=0.75）排名，再对全部符合分类条件的FAQ执行Softmax，默认以归一化分数 **0.55** 作为保守试用门槛，沿用教育系统的评分流程。原教育默认0.85并不是新客服语料的最优值；本轮比较后采用0.55，选择依据与失败见[阈值评测](evaluation/faq_softmax_calibration_20261002/README.md)。可通过`CLOUDCARE_FAQ_BM25_THRESHOLD`或INI的`[faq] bm25_threshold`调整，取值须在0–1之间。所有无词项匹配的FAQ以原始0分参与分母；不是只对Top-1归一化，也没有完全匹配绕过。Softmax反映候选分数的相对集中程度，不是回答正确率；旧8.0／32.0原始分数诊断与本口径不能直接比较。Redis保存记录和版本，各工作进程维护本地BM25快照；缓存键包含FAQ版本与阈值，`trace.faq`同时保存`score`（归一化分）、`raw_score`、阈值及分流原因。

当前网页使用固定演示租户 `demo` 的公开资料。向量分支携带租户、可见性和分类过滤，取回 MySQL 来源后再次核对范围与内容哈希；固定演示租户不等于企业账号认证或完整多租户权限系统。工单没有连接真人客服，没有查询真实订单或执行退款。详细流程见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

支持 TXT、Markdown、PDF、DOCX、PPTX、PNG、JPEG 和 WebP，上传上限 10 MB。扫描 PDF／图片执行真实 OCR，新文件默认父块 1,200 字、子块 300 字、重叠 50 字，保留来源、页码和内容哈希。旧 DOC/PPT 需要先转换。

## 验证与量化指标

- [2026-10-02复测报告](docs/RESUME_EVAL_20261002.md)：BM25→全库Softmax评分、0.85全栈实测与后续0.55试用分流的结果、失败和命令。
- [本轮指标摘要](evaluation/resume_eval_20261002/summary.json)：分类、检索、真实回答路径及Qwen检查分别统计；逐题原始报告以`.json.gz`归档。
- [FULLSTACK_VALIDATION.md](docs/FULLSTACK_VALIDATION.md)：当前全栈各组件的实际运行、失败及验收范围。
- [fullstack_index.json](evaluation/fullstack_index.json)：实际 MySQL、Redis、Milvus 数量和向量抽样，未完成时记录失败。
- [fullstack_metrics.json](evaluation/fullstack_metrics.json)：全栈检索及生成指标，以报告中的语料、模型、题目和运行配置为准。
- [BERT 分类报告](evaluation/support_router/training_report.json)：客服合成分组样本上的分类结果，不等于问答准确率。

### 2026-10-02 复测与当前0.55试用

模型/全栈数字先在0.85门槛下实测。后续只调整FAQ默认门槛为0.55，新增FAQ扫描及20项专项边界检查；没有重做整套神经检索、Qwen或113项全套运行。

| 项目 | 结果与范围 |
|---|---|
| 实际库存 | 112份合成文档、592条带出处知识单元、4,736条FAQ；双向量仍是592条知识 |
| BERT重新推理 | 原432条分组留出样本，Accuracy 92.59%、Macro-F1 0.9265；未重新训练 |
| 神经混合检索＋重排 | 64道开发题：Top-1 60/64（93.75%）、Hit@5 64/64、MRR@5 0.9661；基础／增强BM25为46/64、59/64 |
| 当前FAQ评分分流 | 0.55下64自然正例达到门槛4题，标注出处3题正确、1题错；60题交给RAG；20新挑战均未达到门槛；不是最优或盲测准确率 |
| Qwen实际调用 | 8题、16次成功API调用；6题证据回答校验通过，2题原文回退；缓存0命中 |
| 功能检查 | 0.85版全套113/113；随后0.55专项20/20，验证等于门槛直答、低于门槛进入RAG；不是模型准确率 |
| 本机耗时 | 检索P95 14.717秒；8题Qwen端到端P95 16.246秒；包含顺序加载／释放模型 |

开发题参与过调试，检索来源命中、引用有效和答案业务正确分别衡量。FAQ原样索引自测的99.92%阈值通过率不能解释为自然问法准确率或自动解决率。详情见[本轮复测报告](docs/RESUME_EVAL_20261002.md)。

### 2026-10-01 历史代码实测（FAQ BM25调整前）

下列指标保留10月1日原验收记录；当前版本和FAQ分流应以10月2日复测为准。

| 项目 | 结果与范围 |
|---|---|
| 真实入库 | 112份来源、592条知识片段、4,736条FAQ；Milvus实际592行 |
| BERT分类 | 432条合成分组留出样本，Accuracy 92.59%、Macro-F1 0.9265；仅训练分类头 |
| 检索回归 | 同语料64题，Top-1 60/64、Hit@5 64/64、MRR@5 0.9661；基础／增强BM25分别46/64、59/64 |
| 无依据模式 | 24题中23题进入资料不足模式，另外1题返回否定无条件退款保证的政策原文 |
| Qwen实际调用 | 8题、16次接口调用成功；7题证据回答校验通过，1题生成校验后原文回退；答案缓存0次 |
| 功能检查 | 95/95通过，涵盖契约、真实OCR、实际SQL／Redis及历史回归；不是95道模型质量题 |
| 本机耗时 | 顺序加载释放模型，检索P95 10.76秒；8题Qwen端到端P95 17.48秒，包含加载／重载与服务访问 |

这些是合成资料上的开发回归及功能验证，题目参与过调试，没有独立盲测、RAGAS评分或真实业务上线收益。完整边界、失败记录和复测方式见 [FULLSTACK_VALIDATION.md](docs/FULLSTACK_VALIDATION.md)，项目简历见 [CloudCare_项目经历_简历版.md](docs/CloudCare_项目经历_简历版.md)。

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
$env:CLOUDCARE_MYSQL_INTEGRATION='1'
$env:CLOUDCARE_INTEGRATION='1'
.venv\Scripts\python.exe -m unittest discover -s tests -p test_storage_fullstack.py -v
```

存储集成检查使用真实服务，只清理测试自身随机 ID 的记录。前端处理器检查要求 Node.js 18+。`test_api_fullstack.py` 的传输夹具验证输入和网络边界，不用它宣称真实模型或数据库通过。

### 历史 BM25 基线

旧客服词项检索器保留在 `engine.py`，旧 HTTP 服务保留在 `baseline_app.py`。需要复现旧版本可执行：

```powershell
.venv\Scripts\python.exe baseline_app.py --offline --port 8089
.venv\Scripts\python.exe evaluation/benchmark.py
```

[旧基线报告](docs/BENCHMARK.md)中的 Top-1、纯 BM25 检索耗时与原文 HTTP P95 都属于历史方案，不能当作当前向量检索、重排和 Qwen 全流程成果。开发回归题参与过调试，不能当作真实客户盲测；不同版本的指标需要在同一题目、标注、硬件及缓存条件下比较。

## 文件导航

| 路径 | 内容 |
|---|---|
| `app.py`、`cloudcare/api.py` | 默认 FastAPI 服务与入口 |
| `cloudcare/pipeline.py`、`llm.py` | 客服业务链路、LangChain、模型改写与证据回答 |
| `cloudcare/neural.py`、`retrieval.py` | BERT、BGE-M3、BGE-Reranker、真实 Milvus |
| `cloudcare/documents.py`、`storage.py` | OCR、多格式解析、父子切块、MySQL 与 Redis |
| `cloudcare/settings.py` | 独立客服配置；秘密字段不出现在配置展示中 |
| `compose.yaml`、`requirements-full.txt`、`scripts/bootstrap.ps1` | 独立服务与依赖安装 |
| `scripts/train_support_router.py`、`initialize.py`、`doctor.py` | 客服 BERT 训练、真实入库及实际状态诊断 |
| `static/index.html` | 客服聊天、来源浏览、文件上传和实时库存 |
| `data/manuals/`、`data/runbooks/`、`data/knowledge.jsonl` | 合成来源文档与检索记录 |
| `baseline_app.py`、`engine.py` | 保留的历史 BM25 对照 |
| `evaluation/`、`tests/` | 原始结果、回归检查与集成检查 |
| `runtime/`、`.env`、`.venv/` | 本机运行状态、凭据与依赖，不提交 |

本项目代码和自行生成的合成资料采用 MIT。第三方模型和服务需要遵守各自许可证及条款。
