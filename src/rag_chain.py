"""RAG 主链路：把"检索 + 生成"串成一个端到端函数。

这是 V1 的核心编排。结构刻意保持简单（检索→生成），
方便 W2/W3 在中间插入查询改写、Reranker、流式输出等环节。
"""
from dataclasses import dataclass

from langchain_core.documents import Document

from generator import generate
from retriever import DEFAULT_K, retrieve


@dataclass
class RagResult:
    """一次问答的完整结果，便于上层（CLI / API）取用。"""
    question: str
    answer: str
    sources: list[Document]


def answer_question(question: str, k: int = DEFAULT_K) -> RagResult:
    """端到端：检索 → 生成 → 返回答案与来源。"""
    docs = retrieve(question, k=k)
    answer = generate(question, docs)
    return RagResult(question=question, answer=answer, sources=docs)
