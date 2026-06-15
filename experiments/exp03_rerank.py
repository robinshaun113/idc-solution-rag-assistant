"""Day10 Reranker 实验：向量粗排 top-20 → CrossEncoder 精排 top-4。

目的：
  1. 对比 Reranker vs 纯向量 direct top-4 的 hit@4。
  2. 看假性拒答题是否因更精准的上下文而翻转。

实验设置：
  - 切片：cs=300, ov=50（Day8 最优）。
  - 粗排：向量检索 top-20。
  - 精排：BAAI/bge-reranker-base CrossEncoder 对 20 个候选重排序，取 top-4。
  - 对照组：纯向量 direct top-4（即跳过粗排→精排，直接拿向量前 4）。

用法（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python experiments/exp03_rerank.py
"""

import csv
import functools
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loaders import load_documents                                   # noqa: E402
from splitter import split_documents                                 # noqa: E402
from vector_store import build_vector_store_named                    # noqa: E402

QA_PATH = ROOT / "eval" / "qa_set.jsonl"
OUT_DIR = ROOT / "experiments"
FIG_DIR = OUT_DIR / "figs"
CSV_PATH = OUT_DIR / "exp03_results.csv"

sys.path.insert(0, str(ROOT / "eval"))
from baseline import is_refusal                                      # noqa: E402
from generator import generate                                       # noqa: E402

# ---- 实验参数 ----
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
COARSE_K = 20       # 粗排取多少候选
FINAL_K = 4         # 精排后保留几个
FALSE_REJECT_IDS = {5, 8, 9, 16, 20}
RERANKER_MODEL = "BAAI/bge-reranker-base"


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


def rerank(query: str, candidates: list, model: CrossEncoder) -> list:
    """对粗排候选精排，返回按分数降序的切片列表。

    ═══════════════════════════════════════════════════════════════
    这就是 Reranker 的核心逻辑——你看完对照 exp02 回答：
      exp02 的"混合"靠 RRF 融合两个排名，exp03 的"精排"靠什么？
    ═══════════════════════════════════════════════════════════════
    """
    if len(candidates) <= FINAL_K:
        return candidates

    # CrossEncoder 逐个读 (query, doc_content)，给每对一个相关性分
    pairs = [[query, d.page_content] for d in candidates]
    scores = model.predict(pairs)

    # 按分数降序，取 top-k
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:FINAL_K]]


def evaluate_one(query: str, vs, model: CrossEncoder,
                 expected_source: str | None = None,
                 answerable: bool = True) -> dict:
    """跑一道题：纯向量 top-4 + Reranker top-4，双轨对比。"""
    # 粗排：向量取 COARSE_K 个
    coarse = vs.similarity_search(query, k=COARSE_K)

    # 精排：Reranker 读到 top-k
    reranked = rerank(query, coarse, model)

    # 基线：纯向量 direct top-4（不经过 rerank）
    direct = vs.similarity_search(query, k=FINAL_K)

    result = {"query": query}

    if answerable and expected_source:
        result["direct_hit"] = hit_at_k_from_docs(direct, expected_source)
        result["rerank_hit"] = hit_at_k_from_docs(reranked, expected_source)

    # 假性拒答题：生成答案，看拒答是否翻转
    if not answerable:
        direct_answer = generate(query, direct)
        rerank_answer = generate(query, reranked)
        result["direct_refused"] = is_refusal(direct_answer)
        result["rerank_refused"] = is_refusal(rerank_answer)
        result["direct_answer"] = direct_answer
        result["rerank_answer"] = rerank_answer

    return result


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    qa = load_qa()

    # ① 加载 + 切片 + 建库
    print("[1/4] 加载语料 + 切片（cs=300, ov=50）...")
    docs = load_documents()
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"      共 {len(docs)} 文档 → {len(chunks)} 个切片")

    workdir = Path(tempfile.mkdtemp(prefix="exp03_"))
    try:
        t0 = time.time()
        vs = build_vector_store_named(
            chunks, collection_name="exp03",
            persist_directory=str(workdir / "vs"),
        )
        print(f"      向量库就绪，{time.time() - t0:.0f}s")

        # ② 加载 Reranker 模型
        print(f"\n[2/4] 加载 Reranker 模型 {RERANKER_MODEL} ...")
        model = CrossEncoder(RERANKER_MODEL)
        print("      模型就绪\n")

        # ③ 跑评估
        print("[3/4] 评估：可回答题对比 hit@4，无答案题对比拒答...")
        results = []
        t0 = time.time()
        for item in qa:
            r = evaluate_one(
                item["question"], vs, model,
                expected_source=item.get("expected_source"),
                answerable=item["answerable"],
            )
            r["id"] = item["id"]
            r["type"] = item["type"]
            results.append(r)

            if item["answerable"]:
                direct_hit = r.get("direct_hit", "?")
                rerank_hit = r.get("rerank_hit", "?")
                print(f"  [{item['id']:>2}] {item['type']:<13} "
                      f"direct={direct_hit}  rerank={rerank_hit}")
            else:
                dr = "是" if r.get("direct_refused") else "否"
                rr = "是" if r.get("rerank_refused") else "否"
                print(f"  [{item['id']:>2}] {item['type']:<13} "
                      f"direct拒答={dr}  rerank拒答={rr}")
        elapsed = time.time() - t0

        # ④ 汇总
        print(f"\n[4/4] 汇总（总耗时 {elapsed:.0f}s）...")
        answerable = [r for r in results if r["type"] != "unanswerable"]
        unanswerable = [r for r in results if r["type"] == "unanswerable"]

        direct_hits = sum(r.get("direct_hit", 0) for r in answerable)
        rerank_hits = sum(r.get("rerank_hit", 0) for r in answerable)
        n_ans = len(answerable)

        direct_refusal = sum(r.get("direct_refused", False) for r in unanswerable)
        rerank_refusal = sum(r.get("rerank_refused", False) for r in unanswerable)
        n_unans = len(unanswerable)

        print(f"\n  直接 top-4     hit@4 = {direct_hits}/{n_ans} ({direct_hits/n_ans*100:.1f}%)")
        print(f"  Reranker top-4  hit@4 = {rerank_hits}/{n_ans} ({rerank_hits/n_ans*100:.1f}%)")
        print(f"  直接 top-4     拒答率 = {direct_refusal}/{n_unans}")
        print(f"  Reranker top-4 拒答率 = {rerank_refusal}/{n_unans}")

        # 假性拒答翻转检测
        fr_results = [r for r in results if r["type"] == "unanswerable"
                      or r["id"] in FALSE_REJECT_IDS]
        if fr_results:
            print(f"\n  假性拒答题诊断:")
            for r in fr_results:
                dr = r.get("direct_refused")
                rr = r.get("rerank_refused")
                if dr is not None and rr is not None:
                    if dr and not rr:
                        print(f"    ✅ id {r['id']}: direct 拒答 → rerank 答出（翻转！）")
                    elif dr and rr:
                        print(f"    ⚠️  id {r['id']}: direct 和 rerank 均拒答")

        write_csv(results, direct_hits, rerank_hits, n_ans, direct_refusal, rerank_refusal, n_unans)
        print(f"\n  CSV → {CSV_PATH}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def write_csv(results, direct_hits, rerank_hits, n_ans,
              direct_refusal, rerank_refusal, n_unans):
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "direct_hit", "rerank_hit",
                     "direct_refused", "rerank_refused"])
        for r in results:
            w.writerow([
                r["id"], r["type"],
                r.get("direct_hit", ""), r.get("rerank_hit", ""),
                r.get("direct_refused", ""), r.get("rerank_refused", ""),
            ])
        w.writerow([])
        w.writerow(["汇总", "", "", "", "", ""])
        w.writerow(["direct hit@4", f"{direct_hits}/{n_ans}", f"{direct_hits/n_ans*100:.1f}%"])
        w.writerow(["rerank hit@4", f"{rerank_hits}/{n_ans}", f"{rerank_hits/n_ans*100:.1f}%"])
        w.writerow(["direct 拒答率", f"{direct_refusal}/{n_unans}"])
        w.writerow(["rerank 拒答率", f"{rerank_refusal}/{n_unans}"])


if __name__ == "__main__":
    main()
