"""Day8 切片网格实验：chunk_size × overlap 调参 + 假性拒答根因诊断。

这个实验同时回答两个问题（B 方案的灵魂）：
  (1) 调参：哪组 chunk_size/overlap 让 hit@4 最高？
  (2) 根因：5 道「假性拒答」题(id 5/8/9/16/20)，在不同切片粒度下，
      拒答会不会「翻转」成正常回答？
        - 翻转了 → 大切片把真答案带进了上下文 → 检索侧(粒度)问题，Day8 能救
        - 怎么调都不翻 → 上下文喂进去仍拒答     → 生成侧(阈值/prompt)问题，甩给 Day13

关键工程纪律：
  - 每组参数用【独立 collection + 独立临时目录】，互不串味；
  - 全程【绝不触碰】日常 baseline 的 chroma_db/ 与 idc_ops_kb（96% 资产库）。

用法（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python experiments/exp01_chunk.py
"""
import csv
import functools
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

# 让 print 立刻落盘，后台/管道运行时也能看到实时进度（否则 stdout 被缓冲，
# 要等程序整体结束才一次性刷出，没法看一组组的进度）。
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loaders import load_documents          # noqa: E402
from splitter import split_documents        # noqa: E402
from vector_store import (                   # noqa: E402
    build_vector_store_named,
    load_vector_store_named,
)

QA_PATH = ROOT / "eval" / "qa_set.jsonl"
OUT_DIR = ROOT / "experiments"
FIG_DIR = OUT_DIR / "figs"
CSV_PATH = OUT_DIR / "exp01_results.csv"

# 复用 baseline 的拒答关键词，保证「拒答」判定口径与正式评估一致
sys.path.insert(0, str(ROOT / "eval"))
from baseline import REFUSAL_KEYWORDS, is_refusal  # noqa: E402
from generator import generate                      # noqa: E402

# ---- 实验网格 ----
CHUNK_SIZES = [300, 500, 800, 1200]
OVERLAPS = [50, 100, 200]
TOP_K = 4

# ---- 那 5 道假性拒答题，单独做检索侧/生成侧诊断 ----
FALSE_REJECT_IDS = {5, 8, 9, 16, 20}


def load_qa() -> list[dict]:
    items = []
    with open(QA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def hit_at_k(store, question: str, expected_source: str, k: int = TOP_K) -> int:
    """source 级 hit@k：top-k 切片里有没有来自正确文档的（与 baseline 同口径）。"""
    docs = store.similarity_search(question, k=k)
    sources = [d.metadata["source"] for d in docs]
    return 1 if expected_source in sources else 0


def diagnose_false_reject(store, item: dict) -> dict:
    """对一道假性拒答题：在【当前切片粒度的库】上真的生成一次答案，看还拒不拒答。

    直接调 generator.generate(question, docs)，docs 来自【实验库】的检索结果，
    这样「检索」和「生成」都发生在当前这组切片粒度上，诊断才闭环。
    （不用 rag_chain.answer_question，因为它内部写死走主库 retriever。）
    """
    docs = store.similarity_search(item["question"], k=TOP_K)
    answer = generate(item["question"], docs)
    return {
        "id": item["id"],
        "refused": is_refusal(answer),
        "answer": answer,
    }


def run_one(docs, chunk_size: int, overlap: int, qa: list[dict], workdir: Path) -> dict:
    """跑一组参数：在独立临时库上建索引 → 算全量 hit@4 → 诊断 5 道假性拒答题。"""
    # overlap 必须 < chunk_size，否则比例失衡，标记后跳过（不污染结论）
    if overlap >= chunk_size:
        return {"chunk_size": chunk_size, "overlap": overlap, "skipped": True}

    tag = f"cs{chunk_size}_ov{overlap}"
    persist_dir = workdir / tag
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap)

    t0 = time.time()
    build_vector_store_named(chunks, collection_name=tag, persist_directory=str(persist_dir))
    store = load_vector_store_named(tag, persist_directory=str(persist_dir))
    build_s = time.time() - t0

    # (1) 全量 hit@4（只检索，便宜）
    answerable = [q for q in qa if q["answerable"]]
    hits = sum(hit_at_k(store, q["question"], q["expected_source"]) for q in answerable)
    hit_rate = hits / len(answerable)

    # (2) 5 道假性拒答题：真生成，看拒答翻转（贵，但这是 B 方案的诊断信号）
    fr_items = [q for q in qa if q["id"] in FALSE_REJECT_IDS]
    fr_results = [diagnose_false_reject(store, q) for q in fr_items]
    still_refused = [r["id"] for r in fr_results if r["refused"]]

    print(
        f"  {tag:<14} chunks={len(chunks):>4}  hit@4={hit_rate:.1%}  "
        f"仍拒答={sorted(still_refused) or '无'}  build={build_s:.0f}s"
    )

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "skipped": False,
        "n_chunks": len(chunks),
        "hit_rate": hit_rate,
        "build_s": round(build_s, 1),
        "still_refused": still_refused,
        "fr_results": fr_results,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    qa = load_qa()

    print("[1/3] 加载语料（只加载一次，所有组共用）...")
    docs = load_documents()
    print(f"      共 {len(docs)} 个文档\n")

    print("[2/3] 网格实验：每组独立临时库，绝不碰主库 chroma_db/ ...")
    results = []
    # 手动管理临时目录：Windows 上 Chroma 攥着 sqlite 句柄，TemporaryDirectory
    # 自动清理会抛 PermissionError(WinError 32) 打断脚本；改用 ignore_errors 软删，
    # 删不掉就留在系统 temp（不污染仓库），绝不让收尾报错盖过实验结果。
    workdir = Path(tempfile.mkdtemp(prefix="exp01_"))
    try:
        for cs in CHUNK_SIZES:
            for ov in OVERLAPS:
                results.append(run_one(docs, cs, ov, qa, workdir))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n[3/3] 汇总写 CSV + 画图 ...")
    write_csv(results)
    make_plots(results)
    print_diagnosis(results)


def write_csv(results: list[dict]) -> None:
    rows = [r for r in results if not r.get("skipped")]
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_size", "overlap", "n_chunks", "hit@4", "build_s", "still_refused"])
        for r in rows:
            w.writerow([
                r["chunk_size"], r["overlap"], r["n_chunks"],
                f"{r['hit_rate']:.4f}", r["build_s"],
                "|".join(map(str, sorted(r["still_refused"]))),
            ])
    print(f"  CSV → {CSV_PATH}")


def make_plots(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = [r for r in results if not r.get("skipped")]

    # 图1：hit@4 调参曲线（每条线一个 overlap）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ov in OVERLAPS:
        pts = sorted([r for r in rows if r["overlap"] == ov], key=lambda r: r["chunk_size"])
        if pts:
            ax.plot([p["chunk_size"] for p in pts], [p["hit_rate"] * 100 for p in pts],
                    marker="o", label=f"overlap={ov}")
    ax.set_xlabel("chunk_size"); ax.set_ylabel("hit@4 (%)")
    ax.set_title("exp01: chunk_size × overlap 对 hit@4 的影响")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp01_hit.png", dpi=130); plt.close(fig)

    # 图2：5 道假性拒答题的「拒答翻转」热力图（绿=已答出, 红=仍拒答）
    grid_rows = [(r["chunk_size"], r["overlap"], r) for r in rows]
    ids = sorted(FALSE_REJECT_IDS)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(grid_rows) + 1.5))
    import numpy as np
    mat = np.zeros((len(grid_rows), len(ids)))
    ylabels = []
    for i, (cs, ov, r) in enumerate(grid_rows):
        ylabels.append(f"cs{cs}/ov{ov}")
        refused = set(r["still_refused"])
        for j, qid in enumerate(ids):
            mat[i, j] = 0 if qid in refused else 1  # 1=答出(好), 0=仍拒答(坏)
    ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids))); ax.set_xticklabels([f"id{q}" for q in ids])
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title("假性拒答题：绿=已答出  红=仍拒答")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp01_falsereject.png", dpi=130); plt.close(fig)
    print(f"  图 → {FIG_DIR / 'exp01_hit.png'} , {FIG_DIR / 'exp01_falsereject.png'}")


def print_diagnosis(results: list[dict]) -> None:
    rows = [r for r in results if not r.get("skipped")]
    best = max(rows, key=lambda r: r["hit_rate"])
    print("\n" + "=" * 56)
    print("【调参结论】")
    print(f"  最优：chunk_size={best['chunk_size']}, overlap={best['overlap']}, "
          f"hit@4={best['hit_rate']:.1%}（baseline 96%）")

    print("\n【假性拒答根因诊断】（B 方案的核心产出）")
    # 诚实判据：不再「只要任意一组答出就算检索侧已解决」（那会把偶发翻转误判为搞定）。
    # 改按「在 N 组里答出了几组」分三档，区分度更高，也不替人下草率结论：
    #   - 多数组能答出  → 偏检索侧，调切片大概率能救（Day9/10）
    #   - 仅偶发答出   → 抖动，证据不足，存疑（需 chunk 级标注才能定论）
    #   - 几乎全程拒答 → 偏生成侧，上下文给了仍拒 → prompt/阈值太严（Day13）
    n_groups = len(rows)
    print(f"  （共 {n_groups} 组，统计每题「答出」的组数占比）")
    for qid in sorted(FALSE_REJECT_IDS):
        answered = sum(1 for r in rows if qid not in r["still_refused"])
        ratio = answered / n_groups
        if ratio >= 0.6:
            verdict = "偏检索侧(调切片能救→Day9/10)"
        elif ratio <= 0.2:
            verdict = "偏生成侧(上下文够仍拒→Day13 prompt/阈值)"
        else:
            verdict = "存疑/抖动(证据不足，需 chunk 级标注定论)"
        print(f"    id {qid:>2}: 答出 {answered}/{n_groups} 组 ({ratio:.0%}) → {verdict}")
    print("=" * 56)
    print("注意：本诊断用「拒答是否翻转」做【间接】信号。要在数据上铁证分开")
    print("检索侧/生成侧，需给评估集补 chunk 级黄金标注(gold_span)——见 W2 待办。")


if __name__ == "__main__":
    main()
