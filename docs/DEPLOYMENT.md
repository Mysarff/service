# 本机部署、镜像修复与服务验收

## 当前验收范围

2026 年 10 月 1 日，在 Windows、Docker Desktop 4.89.0、Docker Engine 29.7.2 上检查了本项目的独立服务。原教育项目的数据库、向量集合与文件不参与写入。

| 服务 | 本项目版本／镜像 | 本机位置 | 实际检查 |
|---|---|---|---|
| MySQL | 8.0.43 | `127.0.0.1:3308`，`cloudcare_support` | 真实 SQL 查询，112 份文档、112 个父块、592 个子块、4,736 条 FAQ |
| Redis | 7.2.12 | `127.0.0.1:6381/0`，`cloudcare:v2:` | PING、服务器版本、仅本项目键前缀计数 |
| Milvus | 2.6.24，`cloudcare/milvus:v2.6.24-jammy-rebuilt` | `127.0.0.1:19531`，`cloudcare_support_v2` | Strong consistency 实际计数 592；抽样 8 条均有 1024 维稠密向量、非空稀疏向量，内容 SHA-256 一致 |
| etcd | 3.5.25 | Docker 内部网络 | 容器健康检查通过 |
| MinIO | 固定官方 2024-05-28 源码构建 | Docker 内部网络 | 容器健康检查通过 |

五个 `cloudcare-v2-*` 容器均为 healthy，Milvus 的 `http://127.0.0.1:9092/healthz` 返回 200。使用的 PyMilvus SDK 为 3.0.0；服务端与 SDK 主版本不同，但此次真实连接、计数与向量字段查询通过。不能仅凭版本号推断所有 API 都兼容；混合检索另由模型评测报告验收。

服务检查报告为 [`evaluation/fullstack_services_20261001.json`](../evaluation/fullstack_services_20261001.json)，资料修正重新入库后以无 API 调用方式刷新。单独保存的 [`qwen_connection_20261001.json`](../evaluation/qwen_connection_20261001.json) 记录实际 DashScope 模型 `qwen3.7-flash` 的一次非空连接测试响应。此次连接检查不代表 RAG 回答质量，也不代表 API 的 P95 延迟。服务报告刻意保留 `full_stack_verified=false`：该次检查没有加载本地模型或执行 OCR，模型推理与完整问答应查看各自报告。

24,000 条合成会话工单保存在演示数据文件中，不能把它们当成数据库中已有的真实客服工单。MySQL 中 `sessions/messages/tickets` 的实时数量取决于实际演示请求，不会预先写入虚构的客户活动。

## 镜像为什么要构建

### MinIO

旧 MinIO 社区镜像在本次拉取时不可用。官方仓库说明社区版本按源码分发，所以 `docker/minio/Dockerfile` 从固定官方提交 `f79a4ef4d0dc3e6562cad0d1d1db674bc8c75531` 构建，不依赖来历不明的镜像替代品。

源码、Go 编译器镜像和最终 Alpine 镜像均固定到提交／digest，Go 模块按该版本的 `go.sum` 验证；保留 MinIO 的 AGPL 许可证。官方仓库现已停止维护，本项目固定版本用于本机演示，不能据此声称已完成生产版本维护或安全审核。参见 [MinIO 官方仓库及源码分发说明](https://github.com/minio/minio)。

### Milvus

此次本机拉取／缓存的官方 Milvus 镜像中，Ubuntu 运行时层出现零字节共享库，导致健康检查所用的 `curl` 等程序不能正常执行。这个观察不能证明所有机器上的官方镜像都有同样问题。

`docker/milvus/Dockerfile` 保留固定官方 Milvus 2.6.24 镜像的 `/milvus` 和 `/tini`，从固定的官方 Ubuntu 22.04 镜像重新安装运行时依赖。依赖与环境变量按 [Milvus 2.6.24 官方 Dockerfile](https://github.com/milvus-io/milvus/blob/v2.6.24/build/docker/milvus/ubuntu22.04/Dockerfile) 设置。构建时验证关键库非空、`curl`、`tini` 可执行以及 Milvus 共享库没有零字节文件。

实际切换只替换 `cloudcare-v2-milvus` 容器的运行镜像，沿用 `cloudcare-v2-milvus-data` 卷及原 etcd、MinIO 卷。切换后仍查询到 592 条向量记录，未重新创建或清空数据卷。

## 启动与重启

第一次启动需要能够访问官方镜像、Ubuntu 软件源、GitHub 和 Go 模块源，源码构建可能持续数分钟。准备模型和 Python 环境的步骤见 [`FULLSTACK.md`](FULLSTACK.md)。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -CreateEnv -StartServices -SkipDoctor
```

该命令使用本项目 `.env` 中的独立凭据，不覆盖已存在的值。端口已被其他服务占用时停止并报告冲突。它不会对教育项目运行 `compose down`、`DROP`、`FLUSHDB` 或删除集合。

已构建好本项目镜像时，可使用以下方式跳过重复构建和拉取：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -StartServices -UseExistingImages -SkipDoctor
```

`-SkipDoctor` 只结束环境和容器准备；首次还需要训练客服分类头和初始化知识，不能把这个退出码当作系统验收。完成入库后，仅检查实际服务和知识数量：

```powershell
.venv\Scripts\python.exe scripts/doctor.py --services-only --output evaluation/services_check.json
```

该命令不写测试记录、不加载 GPU 模型、不调用外部模型。`--services-only` 的退出码表示服务检查与文件／MySQL／Milvus 数量一致性，不把未执行的模型推理算作通过。若需检查所有本地模型，另运行 `--probe-models --probe-ocr`；若需一次外部连接检查，显式增加 `--probe-llm`，它会产生一次 API 请求。

## Docker Desktop 启动失败的本机记录

本次 Windows 恢复时，Docker Desktop 在创建引擎前失败，日志为 `sailor-ingest.sock` 无法重命名为 `.stale`，错误 `The file cannot be accessed by the system`。这与 [Docker 官方问题记录 #554](https://github.com/docker/desktop-feedback/issues/554) 描述的失效 Windows AF_UNIX 运行时端点一致。

本次恢复先停止失败的 Desktop 进程，确认两个运行时目录仅含零字节端点，再把 `%LOCALAPPDATA%\Docker\run` 和 `%LOCALAPPDATA%\docker-secrets-engine` **原地改名保留**，重新建立空运行时目录并启动 Desktop。未删除原目录、持久化 VHDX、镜像或卷；原有客服服务和 592 条向量记录仍可读。

这是此次故障的恢复记录，不是自动执行的全局修复脚本。`bootstrap.ps1` 不更改这些宿主机目录。只有日志和目录检查确认同类问题后，才应按实际状态处理，不能把重置 Docker 或清空卷当成启动步骤。

## Python 环境与资源

`requirements-full.txt` 的 29 项固定版本已与本次运行环境逐项对照。实际 PyTorch 为 `2.12.0+cu126`，TorchVision 为 `0.27.0+cu126`，CUDA 12.6，RTX 3060 Laptop GPU（约 6 GiB）。GPU 版安装先使用该文件注明的官方 PyTorch CUDA 索引，再安装其余依赖；这次版本对照不等于对所有 Windows/Linux 机器完成了全新安装测试。

本机客服 `.venv` 入口指向本任务自己的环境，旧教育环境只作为已有依赖和基础模型的只读来源。本项目训练产物、运行时文件、环境、配置和密钥不提交 Git。默认模型内存策略为 `sequential`，逐个加载和释放本地模型，适用于本机显存较小的情况；不能把这种演示配置的响应时间写成生产并发能力。
