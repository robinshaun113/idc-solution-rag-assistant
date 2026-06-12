"""Baseline 评估脚本：把 eval/qa_set.jsonl 灌进 RAG，自动算 hit@4 和拒答率。

用法（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python eval/baseline.py

产出：
- 终端打印每题结果 + 汇总指标
- eval/baseline_report.md：完整结果表 + 指标，供人工 1~5 分打分
"""
import json
import sys
from pathlib import Path

# 让脚本能 import 到 src/ 里的模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retriever import retrieve  # noqa: E402
from rag_chain import answer_question  # noqa: E402

QA_PATH = ROOT / "eval" / "qa_set.jsonl"
REPORT_PATH = ROOT / "eval" / "baseline_report.md"
ANSWERS_PATH = ROOT / "eval" / "answers_full.md"

# 判定"拒答"的关键词：答案里出现任一即视为成功拒答
REFUSAL_KEYWORDS = ["未覆盖", "不知道", "无法回答", "暂未", "建议咨询"]


def load_qa(path: Path) -> list[dict]:
    """逐行读取 JSONL 评估集。"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def is_refusal(answer: str) -> bool:
    """答案是否触发了拒答。"""
    return any(kw in answer for kw in REFUSAL_KEYWORDS)


# ============================================================
# TODO(尹应善填核心逻辑): 评判单道题
#
# 输入：一道题的 dict（含 question / expected_source / answerable 等字段）
# 要做的判断（回忆你想通的那张图）：
#   - 如果是可回答题(answerable=true):
#       1. 用 retrieve(question) 拿回 4 个片段
#       2. 看这 4 个片段的 metadata["source"] 里有没有 expected_source
#       3. 命中 → hit=1，否则 hit=0
#   - 如果是无答案题(answerable=false):
#       不算 hit@4，改调 answer_question(question) 看答案 is_refusal() 是否拒答
#
# 返回一个 dict，建议字段：{"hit": 0/1 或 None, "refused": True/False 或 None, "answer": 生成的答案文本}
# （可回答题的 refused 填 None，无答案题的 hit 填 None）
# ============================================================
def evaluate_one(item: dict, with_answers: bool = False) -> dict:
    if item["answerable"]:
        docs = retrieve(item["question"])
        sources = [d.metadata["source"] for d in docs]
        hit = 1 if item["expected_source"] in sources else 0
        # ----------------------------------------------------------
        # with_answers=True 时本题也生成答案（供人工打分对比 reference）；
        # False 时保持 None，日常回归只跑检索，省钱省时。
        # ----------------------------------------------------------
        answer = answer_question(item["question"]).answer if with_answers else None
        return {"hit": hit, "refused": None, "answer": answer}
    else:
        result = answer_question(item["question"])
        refused = is_refusal(result.answer)
        return {"hit": None, "refused": refused, "answer": result.answer}


def main(with_answers: bool = False) -> None:
    qa = load_qa(QA_PATH)
    print(f"加载评估集：共 {len(qa)} 题")
    if with_answers:
        print("（--with-answers 已开启：可回答题将额外生成答案，较慢且消耗 API）\n")
    else:
        print("（仅检索模式：如需人工打分用的答案文本，请加 --with-answers）\n")

    results = []
    for item in qa:
        r = evaluate_one(item, with_answers=with_answers)
        results.append({**item, **r})
        # 简易进度打印
        tag = f"hit={r.get('hit')}" if item["answerable"] else f"refused={r.get('refused')}"
        print(f"  [{item['id']:>2}] {item['type']:<13} {tag}")

    # ---- 汇总指标（管道活，已写好）----
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    hit_rate = sum(r["hit"] for r in answerable) / len(answerable) if answerable else 0
    refusal_rate = sum(r["refused"] for r in unanswerable) / len(unanswerable) if unanswerable else 0

    print("\n" + "=" * 50)
    print(f"hit@4（{len(answerable)} 道可回答题）: {hit_rate:.1%}")
    print(f"拒答率（{len(unanswerable)} 道无答案题）: {refusal_rate:.1%}")
    print("=" * 50)

    write_report(results, hit_rate, refusal_rate)
    print(f"\n报告已写入 {REPORT_PATH}")
    if with_answers:
        write_answers_full(results)
        print(f"完整答案明细已写入 {ANSWERS_PATH}")


def write_answers_full(results: list[dict]) -> None:
    """把每题的 reference 与完整生成答案并排导出，供人工逐题打分（不截断）。"""
    lines = ["# Baseline 完整答案明细（打分用）", ""]
    for r in results:
        if not r["answerable"]:
            continue
        lines += [
            f"## id {r['id']}　[{r['type']}]　hit={r.get('hit')}",
            f"**问题**：{r['question']}",
            "",
            f"**标准答案(reference)**：{r.get('reference', '')}",
            "",
            "**生成答案**：",
            "",
            (r.get("answer") or "(无)").strip(),
            "",
            "---",
            "",
        ]
    ANSWERS_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_report(results: list[dict], hit_rate: float, refusal_rate: float) -> None:
    """生成 markdown 报告，含人工打分空列。"""
    lines = [
        "# RAG V1 Baseline 评估报告",
        "",
        f"- hit@4（可回答题检索命中率）: **{hit_rate:.1%}**",
        f"- 拒答率（无答案题）: **{refusal_rate:.1%}**",
        "- 人工分：请在下表 `人工1-5` 列手动填写（5=准确完整有出处，1=错误或答非所问）",
        "",
        "| id | 类型 | 问题 | hit | 拒答 | 标准答案(reference) | 人工1-5 | 生成答案（节选） |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ans = (r.get("answer") or "").replace("\n", " ")[:60]
        ref_ans = (r.get("reference") or "").replace("\n", " ")
        hit = "" if r.get("hit") is None else r["hit"]
        ref = "" if r.get("refused") is None else ("是" if r["refused"] else "否")
        lines.append(
            f"| {r['id']} | {r['type']} | {r['question'][:20]} | {hit} | {ref} | {ref_ans} |  | {ans} |"
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(with_answers="--with-answers" in sys.argv)
