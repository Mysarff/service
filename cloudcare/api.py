"""FastAPI routes with bounded inputs and loopback Host/Origin checks."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator
from starlette.concurrency import run_in_threadpool

from .documents import DocumentValidationError
from .settings import Settings, get_settings

LOGGER = logging.getLogger("cloudcare.api")


class LocalRequestBoundary:
    """Validate hosts before every route and cap streamed request bodies."""

    def __init__(self, app: Any, *, port: int, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.origins = {f"http://{host}" for host in self.hosts}
        if port == 80:
            self.hosts.update({"127.0.0.1", "localhost"})
            self.origins.update({"http://127.0.0.1", "http://localhost"})

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        hosts = [value.decode("latin-1").lower() for key, value in headers if key.lower() == b"host"]
        origins = [value.decode("latin-1") for key, value in headers if key.lower() == b"origin"]
        if len(hosts) != 1 or hosts[0] not in self.hosts:
            await JSONResponse({"error": "主机不受支持"}, status_code=403)(scope, receive, send)
            return
        if len(origins) > 1 or any(origin not in self.origins for origin in origins):
            await JSONResponse({"error": "来源不受支持"}, status_code=403)(scope, receive, send)
            return
        lengths = [value for key, value in headers if key.lower() == b"content-length"]
        if len(lengths) > 1:
            await JSONResponse({"error": "请求长度格式无效"}, status_code=400)(scope, receive, send)
            return
        if lengths:
            try:
                declared = int(lengths[0])
                if declared < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse({"error": "请求长度格式无效"}, status_code=400)(scope, receive, send)
                return
            if declared > self.max_bytes:
                await JSONResponse({"error": "请求体超过上传限制"}, status_code=413)(scope, receive, send)
                return
        if scope.get("method") in {"POST", "PUT", "PATCH"}:
            parts = []
            body_size = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                piece = message.get("body", b"")
                body_size += len(piece)
                if body_size > self.max_bytes:
                    await JSONResponse({"error": "请求体超过上传限制"}, status_code=413)(scope, receive, send)
                    return
                parts.append(piece)
                if not message.get("more_body", False):
                    break
            sent_body = False

            async def bounded_receive():
                nonlocal sent_body
                if not sent_body:
                    sent_body = True
                    return {"type": "http.request", "body": b"".join(parts), "more_body": False}
                return await receive()

            next_receive = bounded_receive
        else:
            next_receive = receive

        async def secured_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers", [])) + [(b"x-content-type-options", b"nosniff")]
            await send(message)

        await self.app(scope, next_receive, secured_send)


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(InputModel):
    role: Literal["user", "assistant"]
    content: StrictStr = Field(min_length=1, max_length=2000)


class ChatRequest(InputModel):
    query: StrictStr = Field(min_length=1, max_length=2000)
    category: StrictStr = Field(default="", max_length=128)
    session_id: StrictStr | None = Field(default=None, min_length=1, max_length=96,
                                       pattern=r"^[A-Za-z0-9_-]+$")
    history: list[HistoryMessage] = Field(default_factory=list, max_length=8)
    force_extract: StrictBool = False

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空白")
        return value.strip()


class TextUploadRequest(InputModel):
    filename: StrictStr = Field(min_length=1, max_length=255)
    content: StrictStr = Field(min_length=1)
    category: StrictStr = Field(default="uploaded", max_length=128)


class TicketRequest(InputModel):
    summary: StrictStr = Field(min_length=3, max_length=2000)

    @field_validator("summary")
    @classmethod
    def valid_summary(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("工单摘要至少需要三个字符")
        return value.strip()


def create_app(settings: Settings | None = None, *, pipeline: Any = None) -> FastAPI:
    configuration = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if pipeline is None:
            from .pipeline import SupportPipeline
            service = SupportPipeline(configuration)
        else:
            service = pipeline
        application.state.pipeline = service
        application.state.start_error = None
        try:
            await run_in_threadpool(service.start)
        except Exception as exc:
            # Do not include backend URLs, passwords or provider error strings.
            application.state.start_error = type(exc).__name__
            LOGGER.error("CloudCare initialization blocked (%s)", type(exc).__name__)
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    application = FastAPI(title="CloudCare 企业客服 RAG", version="2.0", lifespan=lifespan,
                          docs_url=None, redoc_url=None)
    application.add_middleware(LocalRequestBoundary, port=configuration.port,
                               max_bytes=configuration.max_upload_bytes + 1_000_000)

    def service(request: Request):
        current = getattr(request.app.state, "pipeline", None)
        if current is None or getattr(request.app.state, "start_error", None):
            raise HTTPException(status_code=503, detail="服务组件尚未就绪，请检查数据库及初始化状态")
        return current

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, _exc: RequestValidationError):
        return JSONResponse({"error": "输入无效，请检查字段类型、长度与历史对话格式"}, status_code=400)

    @application.exception_handler(HTTPException)
    async def expected_error(_request: Request, exc: HTTPException):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @application.exception_handler(DocumentValidationError)
    async def document_error(_request: Request, exc: DocumentValidationError):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @application.exception_handler(ValueError)
    async def value_error(_request: Request, _exc: ValueError):
        return JSONResponse({"error": "输入或上下文无效，请检查问题、文件与会话长度"}, status_code=400)

    @application.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception):
        LOGGER.error("CloudCare request failed (%s)", type(exc).__name__)
        return JSONResponse({"error": "服务处理失败，请检查组件状态"}, status_code=503)

    @application.get("/")
    @application.get("/index.html")
    async def index():
        return FileResponse(configuration.root / "static/index.html", media_type="text/html")

    @application.get("/api/health")
    async def health(request: Request):
        current = getattr(request.app.state, "pipeline", None)
        if current is None:
            return {"status": "blocked", "model_enabled": False, "knowledge_units": 0,
                    "counts": {}, "categories": [], "notice": "服务正在初始化"}
        try:
            result = await run_in_threadpool(current.health)
        except Exception as exc:
            LOGGER.error("CloudCare health probe failed (%s)", type(exc).__name__)
            result = {"status": "blocked", "model_enabled": False, "knowledge_units": 0,
                      "counts": {}, "categories": [], "notice": "数据库或向量索引未就绪"}
        if getattr(request.app.state, "start_error", None):
            result = {**result, "status": "blocked", "notice": "初始化未完成，请检查独立服务配置"}
        return result

    @application.get("/api/knowledge")
    async def knowledge(request: Request):
        result = await run_in_threadpool(service(request).knowledge)
        return result if isinstance(result, dict) else {"items": result}

    @application.post("/api/chat")
    async def chat(payload: ChatRequest, request: Request):
        started = time.perf_counter()
        result = await run_in_threadpool(service(request).answer,
            query=payload.query, category=payload.category, session_id=payload.session_id,
            history=[message.model_dump() for message in payload.history], force_extract=payload.force_extract)
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    @application.post("/api/upload")
    async def upload_text(payload: TextUploadRequest, request: Request):
        if Path(payload.filename).suffix.lower() not in {".txt", ".md"}:
            raise HTTPException(status_code=400, detail="JSON 上传仅支持 TXT/MD；其他格式请使用文件上传")
        raw = payload.content.encode("utf-8")
        if len(raw) > configuration.max_upload_bytes:
            raise HTTPException(status_code=413, detail="文件超过上传大小限制")
        return await run_in_threadpool(service(request).ingest, payload.filename, raw, payload.category)

    @application.post("/api/upload-file")
    async def upload_file(request: Request, file: UploadFile = File(...), category: str = Form("uploaded")):
        if not file.filename or len(file.filename) > 255 or len(category) > 128:
            raise HTTPException(status_code=400, detail="文件名或业务分类格式无效")
        raw = await file.read(configuration.max_upload_bytes + 1)
        await file.close()
        if not raw or len(raw) > configuration.max_upload_bytes:
            raise HTTPException(status_code=413, detail="文件为空或超过上传大小限制")
        return await run_in_threadpool(service(request).ingest, file.filename, raw, category)

    @application.post("/api/tickets")
    async def create_ticket(payload: TicketRequest, request: Request):
        result = await run_in_threadpool(service(request).create_ticket, payload.summary)
        if isinstance(result, dict) and "ticket" in result:
            return result
        return {"ticket": result, "message": "已保存客服演示工单，尚未接入真人客服。"}

    return application
