"""离线索引构建：加载语料 → 切片 → 向量化 → 持久化到 Chroma。

用法（在项目根目录、已激活 .venv 后）：
    python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

# 把 src 加入模块搜索路径，这样脚本能 import loaders/splitter 等
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loaders import load_documents       # noqa: E402
from splitter import split_documents     # noqa: E402
from vector_store import build_vector_store  # noqa: E402


def main() -> None:
    t0 = time.time()

    print("[1/3] 加载语料 ...")
    docs = load_documents()
    print(f"      共 {len(docs)} 个文档")

    print("[2/3] 切片 ...")
    chunks = split_documents(docs)
    print(f"      共 {len(chunks)} 个切片")

    print("[3/3] 向量化并写入 Chroma（首次会调用 embedding API，稍等）...")
    build_vector_store(chunks)

    elapsed = time.time() - t0
    print(f"\n✅ 索引构建完成：{len(chunks)} 个向量，耗时 {elapsed:.1f}s")
    print("   下一步：python main.py \"机房温度过高怎么处理\"")


if __name__ == "__main__":
    main()
