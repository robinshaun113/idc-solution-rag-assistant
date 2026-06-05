"""向量库封装：用 Chroma 持久化存储向量，支持构建与加载。

持久化的意义：把语料向量化一次后存到磁盘（chroma_db/），
之后查询直接加载，不用每次重新调 embedding API（省时省钱）。
"""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_PERSIST_DIR
from embeddings import get_embeddings

COLLECTION_NAME = "idc_ops_kb"


def build_vector_store(chunks: list[Document]) -> Chroma:
    """从切片构建向量库并持久化到磁盘。"""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def load_vector_store() -> Chroma:
    """加载已持久化的向量库（查询时用，不重新 embedding）。"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )
