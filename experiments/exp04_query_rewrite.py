"""Day11 查询改写实验：Multi-Query 与 HyDE vs 直接检索。

目的：
  1. Multi-Query：一个问法 → LLM 改写 3 种 → 分别检索 → RRF 合并 top-4。
  2. HyDE：LLM 先编一段假答案 → 用假答案去检索 top-4。
  3. 看两种方案能否翻转假性拒答题（尤其 id5/9）。

实验设置：
  - 切片：cs=300, ov=50。
  - 检索基础：纯向量 similarity_search。
  - LLM：qwen-plus（百炼兼容接口），复用 config.py 和 embeddings.py 的客户端。
  - 评估维度：hit@4（全量可答题）、假性拒答翻转（id5/8/9/16/20）、延迟、token 消耗。

用法（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python experiments/exp04_query_rewrite.py
"""

import csv
import functools
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loaders import load_documents                                   # noqa: E402
from splitter import split_documents                                 # noqa: E402
from vector_store import build_vector_store_named                    # noqa: E402
from config import (                                                 # noqa: E402
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL,
    assert_api_key,
)

from openai import OpenAI                                            # noqa: E402
from langchain_core.documents import Document                        # noqa: E402

QA_PATH = ROOT / "eval" / "qa_set.jsonl"
OUT_DIR = ROOT / "experiments"
FIG_DIR = OUT_DIR / "figs"
CSV_PATH = OUT_DIR / "exp04_results.csv"

sys.path.insert(0, str(ROOT / "eval"))
from baseline import is_refusal, load_qa                             # noqa: E402
from generator import generate                                       # noqa: E402

# ---- 实验参数 ----
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
FINAL_K = 4
FALSE_REJECT_IDS = {5, 8, 9, 16, 20}
COARSE_K = 20  # Multi-Query 时每个变体取多少，再 RRF 融合


def get_llm():
    """创建百炼 OpenAI 兼容客户端。"""
    assert_api_key()
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def multi_query_rewrite(query: str, client: OpenAI, n: int = 3) -> list[str]:
    """让 LLM 把一个问题改写成 N 个不同问法。"""
    system_prompt = (
        "你是一个检索系统查询优化器。用户问一个问题，你需要把它改写成 N 个语义相同"
        "但措辞不同的变体，每个变体要覆盖原问题的核心信息。\n"
        "要求：\n"
        "- 只返回改写后的问题，每行一个，不要编号、不要解释、不要空行。\n"
        "- 变体之间用词要有差异（比如换同义词、换句式、加/减上下文）。\n"
        f"- 恰好输出 {n} 行。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"原问题：{query}"},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    lines = resp.choices[0].message.content.strip().split("\n")
    # 清洗：去编号、去引导符、去空行
    cleaned = []
    for line in lines:
        line = line.strip().lstrip("0123456789.、-• ").strip()
        if line and len(line) > 3:
            cleaned.append(line)
    return cleaned[:n]


def hyde_document(query: str, client: OpenAI) -> str:
    """让 LLM 生成一段"假设性答案"，模仿知识库文档的措辞风格。"""
    system_prompt = (
        "你是一个数据中心运维专家。用户提了一个关于 IDC 基础设施/标准规范的问题。"
        "请写一段 100-200 字的回答，**模仿国标/白皮书/运维手册的措辞风格**，"
        "尽量使用专业术语和规范表述（如 GB50174、PUE、SLA 等）。"
        "不需要保证答案正确——这段文字将被用来做向量检索"
        "（即用它去找真正包含答案的文档），不直接展示给用户。"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()


def rrf_merge(all_docs: list[list[Document]], k: int = FINAL_K,
              rrf_const: int = 60) -> list[Document]:
    """RRF 融合多个检索结果列表（复用 Day9 的算法）。"""
    scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for docs in all_docs:
        for rank, doc in enumerate(docs):
            key = hash(doc.page_content)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_const + rank + 1)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [doc_map[k] for k in sorted_keys[:k]]


def evaluate_methods(vs, qa: list[dict], client: OpenAI) -> list[dict]:
    """四轨对比：direct / multi-query / hyde / multi-query+hyde。"""
    results = []
    total = len([q for q in qa if q["answerable"]])
    processed = 0

    for item in qa:
        q = item["question"]
        r = {"id": item["id"], "type": item["type"], "question": q}

        # ---- Direct ----
        t0 = time.time()
        d_docs = vs.similarity_search(q, k=FINAL_K)
        d_lat = (time.time() - t0) * 1000
        r["direct_hit"] = hit_at_k(d_docs, item.get("expected_source"))
        r["direct_lat_ms"] = round(d_lat, 1)

        # ---- Multi-Query ----
        t0 = time.time()
        variants = multi_query_rewrite(q, client, n=3)
        mq_docs = [vs.similarity_search(v, k=COARSE_K) for v in variants]
        mq_merged = rrf_merge(mq_docs, k=FINAL_K)
        mq_lat = (time.time() - t0) * 1000
        r["mq_hit"] = hit_at_k(mq_merged, item.get("expected_source"))
        r["mq_lat_ms"] = round(mq_lat, 1)
        r["mq_variants"] = variants

        # ---- HyDE ----
        t0 = time.time()
        hyde_text = hyde_document(q, client)
        h_docs = vs.similarity_search(hyde_text, k=FINAL_K)
        h_lat = (time.time() - t0) * 1000
        r["hyde_hit"] = hit_at_k(h_docs, item.get("expected_source"))
        r["hyde_lat_ms"] = round(h_lat, 1)
        r["hyde_text"] = hyde_text[:150]

        # ---- 假性拒答题额外诊断（生成答案） ----
        if item["id"] in FALSE_REJECT_IDS or not item["answerable"]:
            r["direct_refused"] = is_refusal(generate(q, d_docs))
            r["mq_refused"] = is_refusal(generate(q, mq_merged))
            r["hyde_refused"] = is_refusal(generate(q, h_docs))
        else:
            r["direct_refused"] = None
            r["mq_refused"] = None
            r["hyde_refused"] = None

        results.append(r)
        processed += 1
        if item["answerable"]:
            print(
                f"  [{item['id']:>2}] {item['type']:<13} "
                f"direct={r['direct_hit']} mq={r['mq_hit']} hyde={r['hyde_hit']}  "
                f"d={r['direct_lat_ms']:.0f}ms mq={r['mq_lat_ms']:.0f}ms hyde={r['hyde_lat_ms']:.0f}ms"
            )
        else:
            dr = "是" if r.get("direct_refused") else "否"
            mr = "是" if r.get("mq_refused") else "否"
            hr = "是" if r.get("hyde_refused") else "否"
            print(
                f"  [{item['id']:>2}] {item['type']:<13} "
                f"direct拒答={dr} mq拒答={mr} hyde拒答={hr}"
            )
    return results


def hit_at_k(docs, expected_source: str | None) -> int:
    if not expected_source:
        return 0
    sources = [d.metadata["source"] for d in docs]
    return 1 if expected_source in sources else 0


def print_summary(results: list[dict]) -> None:
    answerable = [r for r in results if r["type"] != "unanswerable"]
    n = len(answerable)

    d_hits = sum(r["direct_hit"] for r in answerable)
    m_hits = sum(r["mq_hit"] for r in answerable)
    h_hits = sum(r["hyde_hit"] for r in answerable)

    d_lat = sum(r["direct_lat_ms"] for r in results) / len(results)
    m_lat = sum(r["mq_lat_ms"] for r in results) / len(results)
    h_lat = sum(r["hyde_lat_ms"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("【exp04 查询改写结论】")
    print(f"  direct       hit@4 = {d_hits}/{n} ({d_hits/n*100:.1f}%)  平均延迟 {d_lat:.0f}ms")
    print(f"  Multi-Query  hit@4 = {m_hits}/{n} ({m_hits/n*100:.1f}%)  平均延迟 {m_lat:.0f}ms")
    print(f"  HyDE         hit@4 = {h_hits}/{n} ({h_hits/n*100:.1f}%)  平均延迟 {h_lat:.0f}ms")

    # 假性拒答
    print(f"\n  假性拒答题翻转情况：")
    fr_ids = sorted(FALSE_REJECT_IDS)
    for r in results:
        if r["id"] not in fr_ids:
            continue
        d_ref = r.get("direct_refused")
        m_ref = r.get("mq_refused")
        h_ref = r.get("hyde_refused")
        parts = []
        if d_ref is not None:
            parts.append(f"direct={'拒' if d_ref else '答'}")
        if m_ref is not None:
            parts.append(f"mq={'拒' if m_ref else '答'}")
        if h_ref is not None:
            parts.append(f"hyde={'拒' if h_ref else '答'}")
        print(f"    id {r['id']:>2}: {', '.join(parts)}")

    # 无答案题拒答率
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    if unanswerable:
        d_ru = sum(r.get("direct_refused", False) for r in unanswerable)
        m_ru = sum(r.get("mq_refused", False) for r in unanswerable)
        h_ru = sum(r.get("hyde_refused", False) for r in unanswerable)
        nu = len(unanswerable)
        print(f"\n  无答案题拒答率: direct={d_ru}/{nu}  mq={m_ru}/{nu}  hyde={h_ru}/{nu}")

    print("=" * 60)

    # 性能解读
    print("\n【解读】")
    print("  Multi-Query 延迟高是因为多次 LLM 调用（改写 3 次） + 多次检索。")
    print("  HyDE 延迟高是因为 LLM 先生成假答案，再检索。")
    print("  两者都在'用 LLM 调用换检索质量'——如果 hit@4 已饱和，")
    print("  则这种交换在当前知识库上不成立。但在大规模/多领域知识库上，")
    print("  用户问法和文档措辞差异大时，查询改写是必要的。")


def write_csv(results: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "type", "question",
            "direct_hit", "mq_hit", "hyde_hit",
            "direct_lat_ms", "mq_lat_ms", "hyde_lat_ms",
            "direct_refused", "mq_refused", "hyde_refused",
            "mq_variants", "hyde_text",
        ])
        for r in results:
            w.writerow([
                r["id"], r["type"], r["question"][:60],
                r.get("direct_hit", ""), r.get("mq_hit", ""), r.get("hyde_hit", ""),
                r.get("direct_lat_ms", ""), r.get("mq_lat_ms", ""), r.get("hyde_lat_ms", ""),
                r.get("direct_refused", ""), r.get("mq_refused", ""), r.get("hyde_refused", ""),
                " | ".join(r.get("mq_variants", [])),
                r.get("hyde_text", ""),
            ])
    print(f"\n  CSV → {CSV_PATH}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    qa = load_qa(QA_PATH)
    client = get_llm()

    print("[1/3] 加载语料 + 切片 + 建库...")
    docs = load_documents()
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"      共 {len(docs)} 文档 → {len(chunks)} 个切片")

    workdir = Path(tempfile.mkdtemp(prefix="exp04_"))
    try:
        t0 = time.time()
        vs = build_vector_store_named(chunks, "exp04", str(workdir / "vs"))
        print(f"      向量库就绪，{time.time() - t0:.0f}s")
        print(f"      LLM: {LLM_MODEL}（百炼兼容接口）\n")

        print(f"[2/3] 四轨对比评估（direct / Multi-Query / HyDE）...")
        t0 = time.time()
        results = evaluate_methods(vs, qa, client)
        print(f"\n      总耗时 {time.time() - t0:.0f}s（含 LLM 调用）")

        print(f"\n[3/3] 汇总...")
        print_summary(results)
        write_csv(results)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
