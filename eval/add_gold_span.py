"""为评估集补 gold_span。手动标注 + 自动匹配混合。

gold_span = 答案原文在知识库中的唯一文本段。评估时检查检索回的
           top-k 切片中是否包含此段（chunk 级命中判定）。

使用已生成的 v2 文件，直接追加手工标注的 gold_span。
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "eval" / "qa_set.jsonl"
OUT_PATH = ROOT / "eval" / "qa_set_v2_goldspan.jsonl"

# 手工标注的 gold_span（从原文文档中直接截取）
MANUAL_SPANS = {
    1:  "冷通道或机柜进风区域的温度范围为 18℃ ~27℃",
    3:  "计算机机房最小使用面积不宜小于 20 m",
    6:  "数据中心划分为 A 、 B 、 C 三级。A 级为容错系统，B 级为冗余系统，C 级为基本需求",
    7:  "A 级数据中心应由双重电源供电，并应设置备用电源。后备柴油发电机组的性能等级不应低于 G3 级",
    8:  "单台机柜发热量大于 4kW 的主机房宜采用活动地板下送风、行间制冷前送风等方式，并宜采取冷热通道隔离措施",
    10: "所在建筑已采用自动灭火系统或场地面积≥140m² 时应安装自动灭火系统",
    12: "Scale-Out 指 GPU 服务器/超节点之间互联,Scale-Up 指服务器或超节点内部 GPU 间互联",
    13: "L1 运行辅助、L2 部分运行自动化、L3 有条件运行自动化、L4 高度运行自动化、L5 完全运行自动化",
    15: "用于搬运设备的通道净宽不应小于 1.5m；面对面布置的机柜正面之间距离不宜小于 1.2m",
    16: "A 级:计算机系统运行中断后,会对国家安全、社会秩序、公共利益造成严重损害的",
    17: "A 级计算机机房楼板荷重 10 kN/m²，C 级 6 kN/m²",
    18: "节能机房的能效比应不大于 1.8。机房能效比一般应不大于 2.4",
    19: "A 级开机时夏季温度 24±1℃,冬季 20±1℃",
    22: "铅酸蓄电池寿命通常 3~6 年，锂离子电池寿命 10~15 年",
    23: "中国要求 2025 年新建大型数据中心 PUE 低于 1.3",
    25: "C 级开机温度 15~28℃，A 级夏季 24±1℃/冬季 20±1℃，C 级温度范围更宽",
}


def main():
    # 加载已自动匹配的 v2 文件（如果有的话）
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            qa = [json.loads(l.strip()) for l in f if l.strip()]
        print(f"加载已有 v2 文件: {len(qa)} 题")
    else:
        with open(QA_PATH, encoding="utf-8") as f:
            qa = [json.loads(l.strip()) for l in f if l.strip()]
        print(f"从原始文件加载: {len(qa)} 题")

    # 手工标注覆盖所有未匹配的
    updated = 0
    for item in qa:
        fid = item["id"]
        if (not item.get("gold_span")) and fid in MANUAL_SPANS:
            item["gold_span"] = MANUAL_SPANS[fid]
            updated += 1
            print(f"  ✏️  id {fid}: 手工标注")

    # 终检
    missing = [item["id"] for item in qa if item["answerable"] and not item.get("gold_span")]
    covered = sum(1 for item in qa if item["answerable"] and item.get("gold_span"))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  gold_span 覆盖: {covered}/{sum(1 for q in qa if q['answerable'])}")
    if missing:
        print(f"  仍缺失 id: {missing}")
    print(f"  输出 → {OUT_PATH}")


if __name__ == "__main__":
    main()
