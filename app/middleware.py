"""Day 19 · 可观测性 — loguru 日志中间件。

三个新概念（先理解再往下看）：
  loguru     : 比 print 强 100 倍的日志库，自动带时间戳、级别、文件切割
  中间件      : FastAPI 的"门卫"——每个请求进来先经过它，走时也经过它
  结构化日志  : 不是写作文，是写"字段=值"对，方便日后 grep / 分析

为什么不在每个接口里写日志：
  你 4 个接口 × 各写一遍 = 4 倍代码 + 格式不统一。中间件写一次，全覆盖。

面试关联 — 可观测性三件套：
  Logging（日志） ← Day 19 做这个
  Metrics（指标） → Prometheus  不在本期计划
  Tracing（追踪） → request_id  ← 本期用 request_id 模拟全链路追踪
"""

import sys
import time
import uuid
from pathlib import Path

from loguru import logger
from starlette.requests import Request

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── loguru 全局配置 ──
# logger.remove() 关掉 loguru 自带的默认 handler，
# 然后手动 add 两个：一个打终端（给人看），一个写文件（给排查用）。
logger.remove()

# 终端输出：带颜色，适合开发时盯着看
logger.add(
    sys.stderr,
    level="INFO",
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[request_id]}</cyan> | "
        "<level>{message}</level>"
    ),
)

# 文件输出：结构化格式，方便 grep / 导入 ELK
# rotation="00:00" → 每天午夜切一个新文件
# retention="7 days" → 只保留最近 7 天的日志
logger.add(
    LOG_DIR / "app.log",
    level="INFO",
    rotation="00:00",
    retention="7 days",
    encoding="utf-8",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[request_id]} | "
        "{message}"
    ),
)

# 设置默认的 request_id，防止不带 request_id 的 logger.info() 调用报 KeyError
logger.configure(extra={"request_id": "--------"})


async def log_request_middleware(request: Request, call_next):
    """FastAPI HTTP 中间件：每个请求自动经过这里。

    做的事：
      1. 请求进来 → 生成 request_id、记录开始时间
      2. 交给业务处理（call_next）
      3. 响应出去 → 从 request.state 收集耗时/token 数据，写一行结构化日志

    ── call_next 是什么？──
    call_next 是 FastAPI 的"下一站"——调用它就是把请求交给后面的中间件
    （或最终的处理函数），拿到响应后继续往下走。不调用 = 请求被拦截。
    """
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()

    # 把 request_id 绑到 request.state 上，这样业务代码（/chat 等）也能用它
    request.state.request_id = request_id

    # 交给业务处理
    response = await call_next(request)

    # ── 收集耗时数据 ──
    # 这些字段由业务端点（/chat、/chat/stream）在处理过程中填入 request.state。
    # 中间件只负责"收菜"——不主动计时检索/生成，让各端点自己计时最准。
    total_ms = (time.perf_counter() - start) * 1000
    retrieval_ms = getattr(request.state, "retrieval_ms", None)
    generation_ms = getattr(request.state, "generation_ms", None)
    token_count = getattr(request.state, "token_count", None)
    question = getattr(request.state, "question", None)

    # ── 拼日志行 ──
    parts = [
        f"{request.method} {request.url.path}",
        f"status={response.status_code}",
        f"total={total_ms:.0f}ms",
    ]
    if retrieval_ms is not None:
        parts.append(f"retrieval={retrieval_ms:.0f}ms")
    if generation_ms is not None:
        parts.append(f"generation={generation_ms:.0f}ms")
    if token_count is not None:
        parts.append(f"tokens={token_count}")
    if question:
        q = question[:80] + "…" if len(question) > 80 else question
        parts.append(f'q="{q}"')

    # bind 把 request_id 注入到 loguru 的 extra 里，format 中的 {extra[request_id]} 就能用
    logger.bind(request_id=request_id).info(" | ".join(parts))

    return response
