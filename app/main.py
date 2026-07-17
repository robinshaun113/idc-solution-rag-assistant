"""RAG API 服务 — FastAPI 把 RAG 能力暴露为 HTTP 接口。

三个新概念（先理解再往下看）：
  FastAPI  : 把 Python 函数变成可被 HTTP 请求触发的接口
  Pydantic : 自动校验请求/响应的格式，不合格直接拒绝
  uvicorn  : 跑 Web 服务的，监听端口等请求进来

启动方式（项目根目录、已激活 .venv）：
  uvicorn app.main:app --reload --port 8000

测试：
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"question\":\"机房温度过高怎么处理\"}"
"""

import sys
import time
from pathlib import Path

# 确保 src/ 和 app/ 下的模块可以被导入
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

# ── FastAPI 核心概念 1: FastAPI() ──
# app 是这个服务的"总开关"。后面所有的接口都挂在 app 上。
from fastapi import FastAPI, HTTPException, UploadFile, File, Request        # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse                # noqa: E402

# ── Pydantic 核心概念: BaseModel ──
# 定义一个类，Pydantic 自动：
#   1. 校验请求里有没有这个字段
#   2. 类型对不对（str 不能传成 int）
#   3. 把请求体 JSON 转成 Python 对象
from pydantic import BaseModel, Field                                       # noqa: E402

# ── RAG 核心逻辑（你已有的模块）──
from rag_chain import RagResult                                             # noqa: E402
from generator import generate, generate_stream                              # noqa: E402
from retriever import retrieve                                               # noqa: E402
from evidence import evidence_payload                                        # noqa: E402

# Day 19: 日志中间件
from middleware import log_request_middleware                                 # noqa: E402

app = FastAPI(
    title="IDC 解决方案 RAG 助手",
    description="面向 IDC 行业客户的 AI 解决方案知识库问答系统",
    version="1.0.0",
)

# Day 19: 注册日志中间件（每个请求自动记录 request_id + 耗时 + token 数）
app.middleware("http")(log_request_middleware)


# ═══════════════════════════════════════════════════════════════
# Pydantic 模型：定义请求和响应的格式
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """用户发问题时的请求格式。

    如果一个请求发来 {"question": 123}，Pydantic 直接拒绝
    （question 应该是 str 不是 int），不用你写校验逻辑。
    """
    question: str = Field(..., min_length=1, max_length=1000,
                          description="用户的问题")


class ChatResponse(BaseModel):
    """系统返回答案时的响应格式。

    统一包装成带 answer + sources 的结构，前端/其他系统直接解包用。
    """
    question: str
    answer: str
    sources: list[dict]  # evidence_id/source/page/chunk_index/preview


# ═══════════════════════════════════════════════════════════════
# 接口 ①: 健康检查（确认服务活着）
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """GET /health → 返回服务状态。

    为什么需要：Docker/监控系统靠这个接口判断容器是否正常运转。
    如果服务挂了，这个接口无响应 → 自动重启。
    """
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════
# 接口 ②: 问答（RAG 核心能力）
# ═══════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    """POST /chat -> RAG 问答。

    请求体 JSON：{"question": "机房温度过高怎么处理"}
    响应体 JSON：{"question": "...", "answer": "...", "sources": [...]}

    Day 19 改动：分离检索/生成调用以便各自计时，
    耗时 + token 数写入 request.state 供中间件统一记日志。
    """
    request.state.question = req.question

    try:
        # ── 检索计时 ──
        t0 = time.perf_counter()
        docs = retrieve(req.question)
        request.state.retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── 生成计时 ──
        t0 = time.perf_counter()
        answer = generate(req.question, docs)
        request.state.generation_ms = (time.perf_counter() - t0) * 1000

        # ── Token 估算（tiktoken cl100k_base，兼容中英文）──
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        request.state.token_count = len(enc.encode(answer))

        sources = [evidence_payload(d) for d in docs]
        return ChatResponse(
            question=req.question,
            answer=answer,
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# 接口 ③: 流式问答（逐字返回答案）
# ═══════════════════════════════════════════════════════════════

@app.post("/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """POST /chat/stream -> 流式 RAG 问答（SSE）。

    Day 19 改动：检索计时记入 request.state；
    流式生成完成后单独写一行日志（token 数在流结束后才确定）。
    """
    request.state.question = req.question

    # ── 检索计时 ──
    t0 = time.perf_counter()
    docs = retrieve(req.question)
    request.state.retrieval_ms = (time.perf_counter() - t0) * 1000

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    def _generate():
        """流式生成 + 完成后补记 token 日志。"""
        gen_start = time.perf_counter()
        all_tokens = []
        try:
            for token in generate_stream(req.question, docs):
                all_tokens.append(token)
                yield f"data: {token}\n\n"
        finally:
            from loguru import logger
            gen_ms = (time.perf_counter() - gen_start) * 1000
            token_count = len(enc.encode("".join(all_tokens)))
            logger.bind(request_id=request.state.request_id).info(
                f"POST /chat/stream | stream_done | "
                f"retrieval={request.state.retrieval_ms:.0f}ms | "
                f"generation={gen_ms:.0f}ms | "
                f"tokens={token_count}"
            )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════
# 接口 ④: 文件上传（扩展知识库）
# ═══════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_knowledge(file: UploadFile = File(...)):
    """POST /upload → 上传新文档追加到知识库。

    上传文件用 multipart/form-data 格式。
    接收后保存到 data/raw/，后续需手动重建索引（rebuild_index.py）。
    """
    # 安全检查：只接受 .txt 和 .pdf
    if not file.filename or not file.filename.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="仅支持 .txt 和 .pdf 文件")

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_name = Path(file.filename).name
    if safe_name != file.filename or safe_name in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="文件名不安全")

    # Demo service limit: avoid unbounded memory use and oversized untrusted files.
    max_bytes = 20 * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="文件不能超过 20MB")

    # 保存到 data/raw/；索引构建仍是显式离线步骤，避免请求线程长时间阻塞。
    dest = ROOT / "data" / "raw" / safe_name
    dest.write_bytes(content)

    return {"status": "ok", "filename": safe_name,
            "size": len(content), "saved_to": str(dest.relative_to(ROOT))}
