"""Day9 混合检索实验：BM25+向量加权 RRF，扫 BM25 权重找最优。

目的：
  1. 验证混合检索是否比纯向量/纯 BM25 更好（hit@4 提升）。
  2. 看假性拒答题在混合检索下是否有翻转（诊断信号，为 Day10/13 提供靶子）。

实验设置：
  - 切片参数固定 Day8 最优：chunk_size=300, overlap=50。
  - 向量库只建一次（各组权重用同一个库，省 API 调用）。
  - BM25 权重 ∈ {0, 0.3, 0.5, 0.7, 1.0}（0=纯向量, 1=纯BM25）。

用法（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python experiments/exp02_hybrid.py
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

from loaders import load_documents                         # noqa: E402
from splitter import split_documents                       # noqa: E402
from vector_store import (                                 # noqa: E402
    build_vector_store_named,
    load_vector_store_named,
)
from hybrid_retriever import create_hybrid_retriever       # noqa: E402

QA_PATH = ROOT / "eval" / "qa_set.jsonl"
OUT_DIR = ROOT / "experiments"
FIG_DIR = OUT_DIR / "figs"
CSV_PATH = OUT_DIR / "exp02_results.csv"

sys.path.insert(0, str(ROOT / "eval"))
from baseline import is_refusal  # noqa: E402
from generator import generate   # noqa: E402

# ---- 实验参数 ----
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 4
BM25_WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]

FALSE_REJECT_IDS = {5, 8, 9, 16, 20}


def load_qa() -> list[dict]:
    items = []
    with open(QA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def hit_at_k_from_docs(docs, expected_source: str) -> int:
    sources = [d.metadata["source"] for d in docs]
    return 1 if expected_source in sources else 0


def diagnose_one(hybrid, item: dict) -> dict:
    docs = hybrid.invoke(item["question"])
    answer = generate(item["question"], docs)
    return {
        "id": item["id"],
        "refused": is_refusal(answer),
        "answer": answer,
    }


def run_one_weight(
    chunks,
    vector_store,
    weight: float,
    qa: list[dict],
) -> dict:
    """测试一个 BM25 权重（复用已建好的 vector_store）。"""
    label = "纯向量" if weight == 0 else ("纯BM25" if weight == 1.0 else f"混合(w={weight:.1f})")

    t0 = time.time()
    hybrid = create_hybrid_retriever(
        chunks,
        vector_store,
        bm25_weight=weight,
        top_k=TOP_K,
    )
    build_s = time.time() - t0

    # (1) 全量 hit@4
    answerable = [q for q in qa if q["answerable"]]
    hits = sum(
        hit_at_k_from_docs(hybrid.invoke(q["question"]), q["expected_source"])
        for q in answerable
    )
    hit_rate = hits / len(answerable) if answerable else 0

    # (2) 假性拒答题诊断
    fr_items = [q for q in qa if q["id"] in FALSE_REJECT_IDS]
    t1 = time.time()
    fr_results = [diagnose_one(hybrid, q) for q in fr_items]
    gen_s = time.time() - t1
    still_refused = [r["id"] for r in fr_results if r["refused"]]

    # (3) 检索延迟（3 次采样平均）
    sample_q = answerable[0]["question"]
    latencies = []
    for _ in range(3):
        tq = time.time()
        hybrid.invoke(sample_q)
        latencies.append((time.time() - tq) * 1000)
    avg_lat_ms = sum(latencies) / len(latencies)

    print(
        f"  [{label:<14}] hit@4={hit_rate:.1%}  "
        f"仍拒答={sorted(still_refused) or '无'}  "
        f"延迟≈{avg_lat_ms:.0f}ms  build={build_s:.0f}s"
    )

    return {
        "bm25_weight": weight,
        "hit_rate": hit_rate,
        "hits": hits,
        "total": len(answerable),
        "still_refused": still_refused,
        "avg_lat_ms": round(avg_lat_ms, 1),
        "build_s": round(build_s, 1),
        "gen_s": round(gen_s, 1),
        "fr_results": fr_results,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    qa = load_qa()

    print("[1/3] 加载语料 + 切片（cs=300, ov=50，Day8 最优）...")
    docs = load_documents()
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"      共 {len(docs)} 文档 → {len(chunks)} 个切片")

    # 向量库只建一次，各组权重复用（省 4x embedding API 调用）
    print("      建向量库（仅一次，各组权重复用）...")
    workdir = Path(tempfile.mkdtemp(prefix="exp02_"))
    try:
        t0 = time.time()
        vs = build_vector_store_named(
            chunks,
            collection_name="exp02_shared",
            persist_directory=str(workdir / "vs"),
        )
        print(f"      向量库就绪，耗时 {time.time() - t0:.0f}s\n")

        print(f"[2/3] 扫 BM25 权重 {{0, 0.3, 0.5, 0.7, 1.0}}（复用同一向量库）...")
        results = [run_one_weight(chunks, vs, w, qa) for w in BM25_WEIGHTS]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n[3/3] 汇总写 CSV + 画图 + 结论 ...")
    write_csv(results)
    make_plot(results)
    print_conclusion(results)


def write_csv(results: list[dict]) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["bm25_weight", "hit@4", "hits", "total", "still_refused", "avg_lat_ms"])
        for r in results:
            w.writerow([
                r["bm25_weight"],
                f"{r['hit_rate']:.4f}",
                r["hits"],
                r["total"],
                "|".join(map(str, sorted(r["still_refused"]))),
                r["avg_lat_ms"],
            ])
    print(f"  CSV → {CSV_PATH}")


def print_conclusion(results: list[dict]) -> None:
    print("\n" + "=" * 56)
    print("【exp02 混合检索结论】")

    best = max(results, key=lambda r: r["hit_rate"])
    pure_vector = next(r for r in results if r["bm25_weight"] == 0.0)
    pure_bm25 = next(r for r in results if r["bm25_weight"] == 1.0)

    print(f"  纯向量 hit@4 = {pure_vector['hit_rate']:.1%}  (cs300/ov50 baseline)")
    print(f"  纯 BM25  hit@4 = {pure_bm25['hit_rate']:.1%}")
    print(f"  最优混合 hit@4 = {best['hit_rate']:.1%}  (bm25_weight={best['bm25_weight']:.1f})")

    print(f"\n  假性拒答题对比：")
    print(f"    纯向量仍拒答:    {sorted(pure_vector['still_refused']) or '无'}")
    print(f"    最优混合仍拒答:  {sorted(best['still_refused']) or '无'}")
    print(f"    纯 BM25 仍拒答:  {sorted(pure_bm25['still_refused']) or '无'}")

    flips = set(pure_vector["still_refused"]) - set(best["still_refused"])
    if flips:
        print(f"    ✅ 混合检索翻转入: id {sorted(flips)}（向量拒答 → 混合答出）")
    else:
        print(f"    ⚠️ 混合检索未能翻转任何假性拒答题")

    print(f"\n  延迟: 纯向量 {pure_vector['avg_lat_ms']:.0f}ms → "
          f"混合(最优) {best['avg_lat_ms']:.0f}ms "
          f"(+{best['avg_lat_ms'] - pure_vector['avg_lat_ms']:.0f}ms)")
    print("=" * 56)


def make_plot(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    weights = [r["bm25_weight"] for r in results]
    hit_rates = [r["hit_rate"] * 100 for r in results]
    lats = [r["avg_lat_ms"] for r in results]

    color1, color2 = "#2c7fb8", "#e6550d"
    ax1.set_xlabel("BM25 权重（0=纯向量, 1=纯BM25）")
    ax1.set_ylabel("hit@4 (%)", color=color1)
    ax1.plot(weights, hit_rates, marker="o", color=color1, linewidth=2, label="hit@4")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("检索延迟 (ms)", color=color2)
    ax2.plot(weights, lats, marker="s", color=color2, linewidth=1.5, linestyle="--", label="延迟")
    ax2.tick_params(axis="y", labelcolor=color2)

    best = max(results, key=lambda r: r["hit_rate"])
    best_w, best_h = best["bm25_weight"], best["hit_rate"] * 100
    ax1.annotate(
        f"最优 w={best_w:.1f}\nhit@4={best_h:.0f}%",
        xy=(best_w, best_h),
        xytext=(best_w + 0.15, best_h - 3 if best_h > 90 else best_h + 3),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )

    fig.suptitle("exp02: BM25+向量混合检索 — hit@4 与延迟随 BM25 权重的变化")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp02_hybrid.png", dpi=130)
    plt.close(fig)
    print(f"  图 → {FIG_DIR / 'exp02_hybrid.png'}")


if __name__ == "__main__":
    main()
