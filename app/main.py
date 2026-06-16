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
from pathlib import Path

# 确保 src/ 下的模块可以被导入
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── FastAPI 核心概念 1: FastAPI() ──
# app 是这个服务的"总开关"。后面所有的接口都挂在 app 上。
from fastapi import FastAPI, HTTPException, UploadFile, File                 # noqa: E402
from fastapi.responses import JSONResponse                                  # noqa: E402

# ── Pydantic 核心概念: BaseModel ──
# 定义一个类，Pydantic 自动：
#   1. 校验请求里有没有这个字段
#   2. 类型对不对（str 不能传成 int）
#   3. 把请求体 JSON 转成 Python 对象
from pydantic import BaseModel, Field                                       # noqa: E402

# ── RAG 核心逻辑（你已有的模块）──
from rag_chain import answer_question, DEFAULT_K                            # noqa: E402

app = FastAPI(
    title="IDC 解决方案 RAG 助手",
    description="面向 IDC 行业客户的 AI 解决方案知识库问答系统",
    version="1.0.0",
)


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
    sources: list[dict]  # [{"source": "GB50174.pdf", "preview": "..."}, ...]


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
def chat(req: ChatRequest):
    """POST /chat → RAG 问答。

    请求体 JSON：{"question": "机房温度过高怎么处理"}
    响应体 JSON：{"question": "...", "answer": "...", "sources": [...]}

    ── FastAPI 核心概念 2: @app.post("/chat") ──
    这个装饰器告诉 FastAPI：当有人向 /chat 发 POST 请求时，
    执行这个函数。req 参数 FastAPI 自动从请求体 JSON 里解析、
    用 Pydantic 校验。

    ── FastAPI 核心概念 3: response_model ──
    FastAPI 自动把函数返回值转成 JSON 响应。
    Pydantic 检查返回格式是否符合 ChatResponse 定义。
    """
    try:
        result = answer_question(req.question)
        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "preview": d.page_content[:100].replace("\n", " "),
            }
            for d in result.sources
        ]
        return ChatResponse(
            question=result.question,
            answer=result.answer,
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# 接口 ③: 文件上传（扩展知识库）
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

    # 保存到 data/raw/
    dest = ROOT / "data" / "raw" / file.filename
    content = await file.read()
    dest.write_bytes(content)

    return {"status": "ok", "filename": file.filename,
            "size": len(content), "saved_to": str(dest.relative_to(ROOT))}
