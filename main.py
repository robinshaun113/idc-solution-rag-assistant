"""CLI 入口：命令行直接提问。

用法（项目根目录、已激活 .venv）：
    python main.py "机房温度过高怎么处理"

需先运行 python scripts/build_index.py 构建索引。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_chain import answer_question  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python main.py "你的问题"')
        sys.exit(1)

    question = sys.argv[1]
    print(f"\n❓ 问题：{question}\n")
    print("🔎 检索 + 生成中 ...\n")

    result = answer_question(question)

    print("=" * 60)
    print(result.answer)
    print("=" * 60)
    print(f"\n📚 本次检索命中 {len(result.sources)} 个片段，来源：")
    seen = []
    for d in result.sources:
        src = d.metadata["source"]
        if src not in seen:
            seen.append(src)
    for s in seen:
        print(f"   - {s}")


if __name__ == "__main__":
    main()
