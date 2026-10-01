# CloudCare 客服 RAG 架构

默认入口 `app.py` 使用 FastAPI 和 `cloudcare/` 中的完整客服链路。原客服 BM25 版本保存在 `baseline_app.py`、`engine.py`，用于旧结果回归与检索对照。完整部署和计数定义见 [FULLSTACK.md](FULLSTACK.md)，本次运行证据见 [FULLSTACK_VALIDATION.md](FULLSTACK_VALIDATION.md)。

## 模块分工

| 模块 | 实际责任 |
|---|---|
| `api.py` | FastAPI 请求界限、Host/Origin 校验、线程池、上传与页面 |
| `settings.py` | 本项目配置、模型路径、数据库命名空间、忽略提交的凭据 |
| `documents.py` | 多格式文本／OCR、LangChain Document、可追溯递归父子分块 |
| `storage.py` | 真实 MySQL 连接池与短事务；真实 Redis 缓存、会话与 FAQ 索引 |
| `neural.py` | 本地 BGE-M3、CrossEncoder 重排及独立客服 BERT |
| `retrieval.py` | Milvus 独立集合、真实稠密／稀疏向量、过滤与融合 |
| `pipeline.py` | 客服路由、并行召回、候选合并、父块核验、会话和工单 |
| `llm.py` | LangChain Qwen 改写链与证据步骤选择、逐字引用校验 |

## 离线构建

样例语料是 16 份模块手册、96 份场景手册、112 个来源父块及 592 条子知识单元。初始化保留原章节／场景 ID，与回归题来源标签对应。

`scripts/initialize.py` 把文档与父子知识写入 MySQL，把 4,736 条 FAQ 及其来源写入 MySQL/Redis，调用 BGE-M3 把每条知识编码为稠密和学习稀疏两种表示，写入独立 Milvus 集合。它再查询真实服务器数量和向量样本，把结果保存到 `evaluation/fullstack_index.json`。两种向量字段属于同一条知识记录，不能把知识量乘二。

新上传文档经过文件签名和大小检查，解析正文／表格或运行 OCR，再由 LangChain 递归父子分块。默认父块 1,200 字、子块 300 字、重叠 50 字。来源、页码、父块关系、文件哈希和片段哈希随记录保留。上传与原样例构建是两条可追溯入库入口，不能宣称所有样例都是此次新切分器产生。

客服 BERT 使用 `support_knowledge`、`handoff`、`out_of_scope` 三个标签，冻结基础 BERT 后训练分类头。合成样本按父来源、表达模板或主题组划分；教育标签不直接充当客服标签。训练报告和数据哈希保存到 `evaluation/support_router/`。

## 在线问答

1. 接收问题及会话 ID，从 Redis 或 MySQL 恢复服务端对话，使用已验证的上下文规则补全短追问。
2. 调用真实客服 BERT，结合业务能力规则处理寒暄、无依据实时业务请求和人工诉求。
3. 检查 Redis 答案缓存与规范化精确 FAQ。缓存键包含语料版本、业务分类、对话和生成模式。
4. 配置有效 Qwen 时通过 LangChain 改写查询，校验编号、金额等约束；失败保留原问题。
5. LangChain 并行运行 Milvus 和 BM25。Milvus 分别检索 BGE-M3 稠密／学习稀疏向量并加权融合；神经与 BM25 结果再通过 RRF 合并。
6. BGE-Reranker 对 Query 与候选片段进行真实二次排序；MySQL 核对来源、范围和内容哈希，补回父块上下文。
7. 检查证据门槛；Qwen 输出选择的原文步骤，服务验证引用 ID 和原文连续子串；异常明确返回原文。没有证据时提示补充信息。
8. MySQL 持久化用户／助手消息，Redis 更新有期限的会话缓存；人工工单写本项目数据库。

BERT 分数和重排分数都不等于答案正确概率。引用与子串校验限制模型补写，但无法证明合成资料是真实政策、也无法保证选取的段落完整回答了问题。

## 范围与状态

默认只绑定本机，HTTP 每一路由检查 Host，带 Origin 的请求必须来自当前本机地址，上传和对话有长度限制。演示网页固定使用 `demo` 租户的公开资料，向量分支先过滤、MySQL 再核对；这不等于已完成企业登录认证、多租户身份授权或生产审计后台。

MySQL 使用 `cloudcare_support`，Redis 使用 `cloudcare:v2:`，Milvus 使用 `cloudcare_support_v2`，由独立端口、容器和卷提供。任何组件失败都记录真实失败，不自动降成假数据库或假神经模型。Qwen 的配置齐全、请求成功、生成引用验证和整套问答质量分别判断。

历史 BM25 评测继续运行在保留的旧服务；当前全栈需单独报告检索、分类、生成、缓存及接口延迟，不能继承旧词项检索的 Top-1 或毫秒级性能。
