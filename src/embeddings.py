"""Embedding 封装：把文本转成向量，用于语义检索。

V1 用百炼 text-embedding-v2（通过 OpenAI 兼容接口）。
以后想换 BGE 本地模型，只需改这一个文件，上层代码不用动 —— 这就是"可插拔"。
"""
from langchain_openai import OpenAIEmbeddings

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    assert_api_key,
)


def get_embeddings() -> OpenAIEmbeddings:
    """返回配置好的 embedding 实例。"""
    assert_api_key()
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        # 百炼 text-embedding-v2 单次最多 25 条，分批避免超限
        chunk_size=25,
        # 关键：langchain-openai 默认会把文本先 tiktoken 切成 token 数组再发，
        # 但百炼兼容接口只认字符串原文，关掉这个预处理（否则报 400 InvalidParameter）
        check_embedding_ctx_length=False,
    )
