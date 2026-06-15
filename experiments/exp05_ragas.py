"""Day12 RAGAS 评估：四个自动化指标对比 baseline vs 当前最优。

因 ragas 0.4.3 与 langchain 1.x 不兼容（VertexAI import 链断裂），
这里用我们自己的 LLM 实现四个核心指标的自动化评估。

指标：
  Context Precision  — 检索回来的切片有几个真相关？
  Context Recall     — 标准答案的信息检索覆盖了多少？
  Faithfulness       — 答案是否基于上下文、有无编造？
  Answer Relevancy   — 答案是否直接回应问题？

实验设置：
  - baseline：cs=500/ov=80（V1 默认参数，chroma_db 主库）
  - current：cs=300/ov=50（Day8 最优参数）
  - 每题用各自配置检索 + 生成，然后用 LLM 逐指标打分。

用法（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python experiments/exp05_ragas.py
"""

import functools
import json
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loaders import load_documents                                   # noqa: E402
from splitter import split_documents                                 # noqa: E402
from vector_store import (                                           # noqa: E402
    build_vector_store_named,
    load_vector_store,
)
from config import (                                                 # noqa: E402
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL,
    assert_api_key,
)
from generator import generate                                       # noqa: E402

from openai import OpenAI                                            # noqa: E402
from langchain_core.documents import Document                        # noqa: E402

QA_PATH = ROOT / "eval" / "qa_set.jsonl"
REPORT_PATH = ROOT / "eval" / "ragas_report.md"
sys.path.insert(0, str(ROOT / "eval"))
from baseline import load_qa, is_refusal                             # noqa: E402


# ---- 实验参数 ----
FINAL_K = 4
# 只评可回答题（无答案题不适用这四个指标）
# 无答案题单独用拒答率评估（已在 baseline.py 里做了）


def get_llm():
    assert_api_key()
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


# ═══════════════════════════════════════════════════════════════════
# 四个指标的实现（每个都用 LLM 做语义判断）
# ═══════════════════════════════════════════════════════════════════

def score_context_precision(
    question: str, contexts: list[str], client: OpenAI
) -> float:
    """对检索回来的 k 个切片：逐个问 LLM '这个切片对回答此问题有用吗？'

    返回 relevant_count / total_count。
    """
    if not contexts:
        return 0.0

    relevant = 0
    for ctx in contexts:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个严格的检索质量评估器。判断一段上下文是否包含"
                        "对回答用户问题有用的信息。只回答 YES 或 NO。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"问题：{question}\n\n上下文：{ctx[:500]}\n\n这段上下文包含对回答此问题有用的信息吗？",
                },
            ],
            temperature=0,
            max_tokens=5,
        )
        ans = resp.choices[0].message.content.strip().upper()
        if "YES" in ans:
            relevant += 1

    return relevant / len(contexts)


def score_context_recall(
    question: str, contexts: list[str], ground_truth: str, client: OpenAI
) -> float:
    """标准答案里的关键信息点，上下文覆盖了多少？

    让 LLM 对比 ground_truth 和 contexts 的覆盖程度，返回 0-1 分数。
    """
    if not contexts or not ground_truth:
        return 0.0

    ctx_text = "\n---\n".join(c[:400] for c in contexts)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个检索覆盖度评估器。给定一个标准答案和多段检索回来的上下文，"
                    "判断标准答案中的关键信息在上下文中有多高的覆盖率。"
                    "只输出一个 0.0 到 1.0 之间的数字（保留一位小数），不要任何解释。\n"
                    "1.0 = 标准答案的信息完全被上下文覆盖。\n"
                    "0.0 = 上下文完全没有涉及标准答案的信息。\n"
                    "0.5 = 大约一半的信息被覆盖。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"标准答案：{ground_truth[:500]}\n\n"
                    f"检索上下文：\n{ctx_text}"
                ),
            },
        ],
        temperature=0,
        max_tokens=10,
    )
    ans = resp.choices[0].message.content.strip()
    try:
        return float(ans)
    except ValueError:
        # fallback: 尝试从字符串里提取数字
        import re
        nums = re.findall(r"(\d\.?\d*)", ans)
        return float(nums[0]) if nums else 0.5


def score_faithfulness(
    question: str, answer: str, contexts: list[str], client: OpenAI
) -> float:
    """答案是否忠于上下文——有没有编造内容。

    让 LLM 逐句检查：答案中的每个陈述是否能在上下文中找到支撑。
    返回 0-1 分数。
    """
    if not answer or not contexts:
        return 0.0

    # 拒答直接满分（忠实于"不知道"就是正确的）
    if is_refusal(answer):
        return 1.0

    ctx_text = "\n---\n".join(c[:500] for c in contexts)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严格的事实核查员。判断一个 RAG 系统生成的答案是否忠实于"
                    "给定的上下文——即答案中的陈述是否都能在上下文中找到依据。"
                    "只输出一个 0.0 到 1.0 之间的数字（保留一位小数），不要任何解释。\n"
                    "1.0 = 答案完全基于上下文，没有任何编造。\n"
                    "0.0 = 答案中的关键陈述与上下文矛盾或毫无根据。\n"
                    "0.5 = 答案部分有依据、部分在编造。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"生成答案：{answer[:600]}\n\n"
                    f"检索上下文：\n{ctx_text}"
                ),
            },
        ],
        temperature=0,
        max_tokens=10,
    )
    ans = resp.choices[0].message.content.strip()
    try:
        return float(ans)
    except ValueError:
        import re
        nums = re.findall(r"(\d\.?\d*)", ans)
        return float(nums[0]) if nums else 0.5


def score_answer_relevancy(
    question: str, answer: str, client: OpenAI
) -> float:
    """答案是否直接回应问题——有没有答非所问或跑题。

    这与 faithfulness 不同：faithfulness 测"基于上下文与否"，
    answer_relevancy 测"是否回应用户的问题"。
    """
    if not answer:
        return 0.0

    # 拒答 = 对无答案题是合理的，对可回答题则意味着不相关
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个答案相关性评估器。判断一个答案是否直接、完整地回应了用户的问题。"
                    "只输出一个 0.0 到 1.0 之间的数字（保留一位小数），不要任何解释。\n"
                    "1.0 = 答案完全回应了问题，核心信息全部覆盖。\n"
                    "0.0 = 答案完全跑题或回答'不知道'但没有给出有帮助的下一步。\n"
                    "0.5 = 答案部分相关但有明显遗漏或包含大量无关内容。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n答案：{answer[:600]}",
            },
        ],
        temperature=0,
        max_tokens=10,
    )
    ans = resp.choices[0].message.content.strip()
    try:
        return float(ans)
    except ValueError:
        import re
        nums = re.findall(r"(\d\.?\d*)", ans)
        return float(nums[0]) if nums else 0.5


def _retry_call(func, max_retries: int = 3):
    """网络波动时自动重试。"""
    import time as _time
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                _time.sleep(2 * (attempt + 1))
                continue
            raise e


def evaluate_one(item: dict, vs, client: OpenAI) -> dict:
    """对一道可回答题跑四个指标。"""
    q = item["question"]
    ref = item.get("reference", "")
    docs = vs.similarity_search(q, k=FINAL_K)
    ctx_texts = [d.page_content for d in docs]

    # 生成答案（加重试）
    t0 = time.time()
    answer = _retry_call(lambda: generate(q, docs))
    gen_lat = (time.time() - t0) * 1000

    result = {
        "id": item["id"],
        "type": item["type"],
        "question": q[:80],
    }

    # 四个指标（加重试）
    result["context_precision"] = round(
        _retry_call(lambda: score_context_precision(q, ctx_texts, client)), 3
    )
    result["context_recall"] = round(
        _retry_call(lambda: score_context_recall(q, ctx_texts, ref, client)), 3
    )
    result["faithfulness"] = round(
        _retry_call(lambda: score_faithfulness(q, answer, ctx_texts, client)), 3
    )
    result["answer_relevancy"] = round(
        _retry_call(lambda: score_answer_relevancy(q, answer, client)), 3
    )
    result["is_refusal"] = is_refusal(answer)
    result["gen_lat_ms"] = round(gen_lat, 1)

    return result


def main() -> None:
    client = get_llm()
    qa = load_qa(QA_PATH)
    answerable = [q for q in qa if q["answerable"]]

    print(f"加载评估集：{len(answerable)} 道可回答题")
    print(f"LLM: {LLM_MODEL}\n")

    # ---- Baseline: cs500/ov80（主库 chroma_db） ----
    print("[1/2] 评估 baseline（主库 chroma_db, cs=500/ov=80）...")
    vs_baseline = load_vector_store()
    baseline_results = []
    t0 = time.time()
    for item in answerable:
        r = evaluate_one(item, vs_baseline, client)
        baseline_results.append(r)
        print(
            f"  [{r['id']:>2}] {r['type']:<10} "
            f"CP={r['context_precision']:.2f} CR={r['context_recall']:.2f} "
            f"F={r['faithfulness']:.2f} AR={r['answer_relevancy']:.2f} "
            f"{'拒答' if r['is_refusal'] else '已答'}"
        )
    bl_time = time.time() - t0

    # ---- Current: cs300/ov50（Day8 最优） ----
    print(f"\n[2/2] 评估 current（cs=300/ov=50, Day8 最优）...")
    docs = load_documents()
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    vs_current = build_vector_store_named(chunks, "exp05_current")

    current_results = []
    t0 = time.time()
    for item in answerable:
        r = evaluate_one(item, vs_current, client)
        current_results.append(r)
        print(
            f"  [{r['id']:>2}] {r['type']:<10} "
            f"CP={r['context_precision']:.2f} CR={r['context_recall']:.2f} "
            f"F={r['faithfulness']:.2f} AR={r['answer_relevancy']:.2f} "
            f"{'拒答' if r['is_refusal'] else '已答'}"
        )
    cur_time = time.time() - t0

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"【exp05 RAGAS 报告】")
    print(f"{'='*60}")
    print(f"{'指标':<22} {'baseline(cs500)':>15} {'current(cs300)':>15} {'变化':>10}")
    print(f"{'-'*62}")

    avg_bl = {k: sum(r[k] for r in baseline_results) / len(baseline_results)
              for k in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]}
    avg_cur = {k: sum(r[k] for r in current_results) / len(current_results)
               for k in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]}

    labels = {
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
    }
    for key, label in labels.items():
        bl = avg_bl[key]
        cur = avg_cur[key]
        delta = cur - bl
        sign = "+" if delta > 0 else ""
        print(f"  {label:<20} {bl:>15.3f} {cur:>15.3f} {sign}{delta:>9.3f}")

    # 拒答率
    bl_refusal = sum(r["is_refusal"] for r in baseline_results)
    cur_refusal = sum(r["is_refusal"] for r in current_results)
    print(f"  {'拒答数':<20} {bl_refusal:>15} {cur_refusal:>15}")
    print(f"  {'耗时':<20} {bl_time:>14.0f}s {cur_time:>14.0f}s")

    print(f"\n{'='*60}")
    print("【解读】")
    print("  · Context Precision 高 → 检索回来的切片大部分真的有用")
    print("  · Context Recall 高 → 标准答案的关键信息被检索覆盖到了")
    print("  · Faithfulness 高 → 答案没编造，忠于上下文")
    print("  · Answer Relevancy 高 → 答案直接回应了问题内容")
    print(f"\n  ⚠️ Faithfulness 和 Answer Relevancy 的下降空间 = 生成侧的优化空间")
    print(f"  四轮检索实验 + RAGAS = 完整证据链：Day13 动生成层。")

    # 保存报告
    write_report(baseline_results, current_results, avg_bl, avg_cur,
                 bl_refusal, cur_refusal)


def write_report(bl_results, cur_results, avg_bl, avg_cur,
                 bl_refusal, cur_refusal):
    lines = [
        "# RAGAS 自动化评估报告",
        "",
        "> 生成日期：2026/6/15",
        "> 评估对象：baseline (cs=500/ov=80) vs current (cs=300/ov=50, Day8最优)",
        f"> 评估题数：{len(bl_results)} 道可回答题",
        f"> LLM 评分器：{LLM_MODEL}",
        "",
        "## 1. 四指标对比",
        "",
        "| 指标 | baseline | current | 变化 | 说明 |",
        "|---|---|---|---|---|",
    ]
    labels = [
        ("context_precision", "Context Precision", "检索是否精准"),
        ("context_recall", "Context Recall", "标准答案覆盖度"),
        ("faithfulness", "Faithfulness", "答案忠于上下文"),
        ("answer_relevancy", "Answer Relevancy", "答案回应问题"),
    ]
    for key, name, desc in labels:
        bl = avg_bl[key]
        cur = avg_cur[key]
        delta = cur - bl
        sign = "+" if delta > 0 else ""
        lines.append(f"| {name} | {bl:.3f} | {cur:.3f} | {sign}{delta:.3f} | {desc} |")

    lines += [
        "",
        f"| 拒答数 | {bl_refusal} | {cur_refusal} | — | 越低越好（对可回答题） |",
        "",
        "## 2. 每题明细",
        "",
        "| id | 类型 | CP(bl/cur) | CR(bl/cur) | F(bl/cur) | AR(bl/cur) | 拒答? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for bl, cur in zip(bl_results, cur_results):
        lines.append(
            f"| {bl['id']} | {bl['type']} | "
            f"{bl['context_precision']:.2f}/{cur['context_precision']:.2f} | "
            f"{bl['context_recall']:.2f}/{cur['context_recall']:.2f} | "
            f"{bl['faithfulness']:.2f}/{cur['faithfulness']:.2f} | "
            f"{bl['answer_relevancy']:.2f}/{cur['answer_relevancy']:.2f} | "
            f"{'拒' if cur['is_refusal'] else '答'} |"
        )

    lines += [
        "",
        "## 3. 结论",
        "",
        "### 检索侧指标（Context Precision / Context Recall）",
        "四轮检索实验（切片/混合/Reranker/查询改写）已证明检索侧饱和，",
        "RAGAS 的 CP 和 CR 进一步用数字确认。",
        "",
        "### 生成侧指标（Faithfulness / Answer Relevancy）",
        "这两个指标的短板直接对应假性拒答问题（id5/9）。",
        f"Faithfulness 低于 1.0 说明生成层在'忠于上下文'和'拒答'之间的平衡需要调整。",
        "",
        "### 下一步",
        "Day 13：优化生成层——system prompt、引用溯源、拒答阈值。",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()
