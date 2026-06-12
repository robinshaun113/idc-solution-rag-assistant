"""混合检索器：BM25（关键词稀疏）+ 向量（语义稠密），加权 RRF 融合。

为什么需要混合检索：
- 向量检索擅长语义相近（"温度高"≈"过热"），但对精确关键词（"GB50174"）不敏感；
- BM25 擅长精确关键词匹配（"UPS 故障"一定命中含 UPS 的文档），但不理解同义词。
- RRF（Reciprocal Rank Fusion）把两边排名融合：排名越靠前的文档得分越高，
  无需归一化不同来源的原始分数（BM25 的 TF-IDF 和向量的 cosine 尺度完全不同）。

用法：
    from hybrid_retriever import HybridRetriever, create_hybrid_retriever
    retriever = create_hybrid_retriever(chunks, vector_store, bm25_weight=0.3)
    docs = retriever.invoke("机房温度过高怎么处理")
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever


class HybridRetriever(BaseRetriever):
    """加权 RRF 混合检索器。

    把 BM25 和向量检索的结果用 RRF 加权融合，返回 top-k。
    两个检索器搜的是同一套 chunks，所以 content hash 可做去重 key。

    实现要点：BM25Retriever 和 Chroma vector retriever 都需要在 invoke 前设好 k 值
    （因为 invoke 第二参数是 config，不能传 top_k）。我们在 __init__ 后统一设 fetch_k，
    invoke 只传 query 字符串。
    """

    bm25_retriever: BM25Retriever
    vector_retriever: BaseRetriever   # Chroma.as_retriever() 返回值
    bm25_weight: float = 0.5          # BM25 权重，向量权重 = 1 - bm25_weight
    rrf_k: int = 60                   # RRF 平滑常数，标准值 60
    top_n: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        """执行加权 RRF 混合检索。"""
        # 1. 两边各自检索
        #    fetch_k 在 __init__ 后已设好（见 create_hybrid_retriever），
        #    这里直接 invoke(query) 即可拿到 fetch_k 条结果。
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        # 2. 加权 RRF 计分
        scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        def _add(docs: list[Document], weight: float) -> None:
            for rank, doc in enumerate(docs):
                key = hash(doc.page_content)
                scores[key] = scores.get(key, 0.0) + weight / (self.rrf_k + rank + 1)
                doc_map[key] = doc

        _add(bm25_docs, self.bm25_weight)
        _add(vector_docs, 1.0 - self.bm25_weight)

        # 3. 按 RRF 得分降序，取 top_n
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[: self.top_n]
        return [doc_map[k] for k in sorted_keys]


def create_hybrid_retriever(
    chunks: list[Document],
    vector_store,
    bm25_weight: float = 0.5,
    top_k: int = 4,
) -> HybridRetriever:
    """工厂函数：用同一套 chunks 创建 HybridRetriever。

    Args:
        chunks: 切好的文档块（BM25 和向量库用同一套）。
        vector_store: 已构建的 Chroma 向量库（含相同 chunks 的向量）。
        bm25_weight: BM25 在 RRF 融合中的权重，0=纯向量，1=纯 BM25。
        top_k: 最终返回的文档数。

    Returns:
        可直接 .invoke(query) 的 HybridRetriever 实例。
    """
    # RRF 融合前两边各自多取一些：候选池越大，RRF 区分度越高
    fetch_k = max(top_k * 5, 20)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = fetch_k

    vector = vector_store.as_retriever(search_kwargs={"k": fetch_k})

    return HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        bm25_weight=bm25_weight,
        top_n=top_k,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_ROOT / "src"))

    from loaders import load_documents
    from splitter import split_documents
    from vector_store import load_vector_store

    docs = load_documents()
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    vs = load_vector_store()

    retriever = create_hybrid_retriever(chunks, vs, bm25_weight=0.3)

    q = sys.argv[1] if len(sys.argv) > 1 else "机房温度过高怎么处理"
    results = retriever.invoke(q)
    print(f"问题：{q}\n命中 {len(results)} 个切片：")
    for i, d in enumerate(results, 1):
        preview = d.page_content[:80].replace("\n", " ")
        print(f"  [{i}] ({d.metadata['source']}) {preview}...")
