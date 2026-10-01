"""CloudCare full-stack FastAPI launcher; historical baseline: baseline_app.py."""
from __future__ import annotations

import argparse
from dataclasses import replace

from cloudcare.api import create_app
from cloudcare.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudCare 企业客服 RAG 全栈服务")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model-config", help="客服项目自己的 INI 配置路径")
    parser.add_argument("--offline", action="store_true", help="停用外部模型调用；仍使用真实数据库与本地神经检索")
    args = parser.parse_args()
    settings = Settings.load(args.model_config)
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            parser.error("端口必须介于 1 和 65535")
        settings = replace(settings, port=args.port)
    if args.offline:
        settings = replace(settings, llm_api_key="", query_rewrite_enabled=False)
    import uvicorn
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.port,
                workers=1, timeout_keep_alive=5, access_log=True)


if __name__ == "__main__":
    main()
