"""文档加载：把 data/raw 下的多格式语料统一加载为 LangChain Document 列表。

V1 支持 .txt 和 .pdf。每个 Document 带 metadata.source（文件名），用于后续引用溯源。
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# 项目根目录下的默认语料目录
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_documents(raw_dir: Path | str = DEFAULT_RAW_DIR) -> list[Document]:
    """加载 raw_dir 下所有 .txt / .pdf 文件。

    Args:
        raw_dir: 语料目录，默认 data/raw。

    Returns:
        Document 列表，每个文档的 metadata["source"] 为文件名。
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"语料目录不存在: {raw_dir}")

    docs: list[Document] = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            continue  # 跳过 .gitkeep 等非语料文件

        loaded = loader.load()
        # 统一把 source 设为文件名（PyPDFLoader 默认是完整路径，这里规范化）
        for d in loaded:
            d.metadata["source"] = path.name
        docs.extend(loaded)

    return docs


if __name__ == "__main__":
    documents = load_documents()
    print(f"加载到 {len(documents)} 个文档：")
    for d in documents:
        print(f"  - {d.metadata['source']}  ({len(d.page_content)} 字符)")
