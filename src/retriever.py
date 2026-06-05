"""检索器：根据问题从向量库取回最相关的 k 个切片。

V1 用最朴素的向量相似度检索。W2 会升级为混合检索 + Reranker（见 PLAN.md）。
"""
from langchain_core.documents import Document

from vector_store import load_vector_store

DEFAULT_K = 4


def retrieve(query: str, k: int = DEFAULT_K) -> list[Document]:
    """返回与 query 最相关的 k 个切片。"""
    store = load_vector_store()
    return store.similarity_search(query, k=k)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "机房温度过高怎么处理"
    results = retrieve(q)
    print(f"问题：{q}\n命中 {len(results)} 个切片：")
    for i, d in enumerate(results, 1):
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  [{i}] ({d.metadata['source']}) {preview}...")
