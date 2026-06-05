"""文档切片：把长文档切成适合检索的小块（chunk）。

为什么要切片：embedding 模型和 LLM 的上下文有长度限制；而且检索时
"小块"能更精准命中问题相关的段落，避免把无关内容也塞进上下文。
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# V1 默认参数，W2 会做网格实验找最优（见 PLAN.md Day 8）
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """把文档列表切成 chunk。

    RecursiveCharacterTextSplitter 会优先按段落/换行/句子等"自然边界"切，
    尽量不把一句话拦腰截断。overlap 让相邻块有重叠，避免边界信息丢失。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 中文友好的分隔符优先级：先按段落，再按换行、句号，最后才按字符
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    from loaders import load_documents

    chunks = split_documents(load_documents())
    print(f"切片完成，共 {len(chunks)} 块。前 3 块预览：")
    for c in chunks[:3]:
        preview = c.page_content[:60].replace("\n", " ")
        print(f"  [{c.metadata['source']}] {preview}...")
