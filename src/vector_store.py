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


# ============================================================
# 以下两个函数供 Day8 切片实验用：每组参数建一个【独立 collection】，
# 互不串味，且【绝不触碰】日常 baseline 用的 idc_ops_kb（96% 的资产库）。
# ============================================================
def build_vector_store_named(
    chunks: list[Document],
    collection_name: str,
    persist_directory: str | None = None,
) -> Chroma:
    """把切片灌进【指定名字】的 collection。

    与 build_vector_store 的唯一区别：collection_name 可外部传入，
    这样实验里每组参数用一个独立 collection（如 exp_cs300_ov50），
    避免不同 chunk 的向量混进同一个库里「串味」。
    persist_directory=None 时落到默认 CHROMA_PERSIST_DIR；
    实验建议传一个临时目录，跑完即弃，不留垃圾在主库目录。
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=persist_directory or CHROMA_PERSIST_DIR,
    )


def load_vector_store_named(
    collection_name: str,
    persist_directory: str | None = None,
) -> Chroma:
    """加载【指定名字】的 collection（与 build_vector_store_named 配对）。"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory or CHROMA_PERSIST_DIR,
    )
