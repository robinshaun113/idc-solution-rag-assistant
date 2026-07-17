"""Production retriever.

The default path intentionally remains direct dense retrieval: experiments showed
that BM25, reranking and query rewriting did not improve the current small corpus.
Those implementations remain under ``experiments/`` as reproducible comparisons.
"""
from functools import lru_cache

from langchain_core.documents import Document

from evidence import backfill_legacy_evidence_metadata
from vector_store import load_vector_store

DEFAULT_K = 4


@lru_cache(maxsize=1)
def get_vector_store():
    """Load Chroma once per process instead of rebuilding the client per request."""
    return load_vector_store()


def retrieve(query: str, k: int = DEFAULT_K) -> list[Document]:
    """返回与 query 最相关的 k 个切片。"""
    store = get_vector_store()
    docs = store.similarity_search(query, k=k)
    # Old indexes may predate evidence metadata. Backfill it at read time so the
    # API contract stays stable; rebuilding the index persists the same fields.
    missing = [d for d in docs if not d.metadata.get("evidence_id")]
    if missing:
        backfill_legacy_evidence_metadata(missing)
    return docs


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "机房温度过高怎么处理"
    results = retrieve(q)
    print(f"问题：{q}\n命中 {len(results)} 个切片：")
    for i, d in enumerate(results, 1):
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  [{i}] ({d.metadata['source']}) {preview}...")
