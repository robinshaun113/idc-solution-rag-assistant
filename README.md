# IDC 运维智能知识库助手 · idc-solution-rag-assistant

> 面向数据中心一线运维场景的企业级 RAG 智能问答系统
> An enterprise-grade RAG assistant for IDC operations knowledge.

![status](https://img.shields.io/badge/status-WIP-yellow) ![python](https://img.shields.io/badge/python-3.11-blue) ![stage](https://img.shields.io/badge/stage-V1-lightgrey)

---

## 📖 项目背景（Why）

在中国移动 IDC 业务实习期间观察到，一线运维新人遇到告警时，常常要在数百份分散的 PDF 手册、Wiki、国家标准里翻找处置规程，检索成本高、夜班尤其低效。知识不是不存在，而是"查得慢"。

本项目把这些非结构化文档变成**可对话、可溯源**的知识接口：用自然语言提问，秒级拿到带来源出处的答案；查不到时明确拒答，不编造。

> 详细需求与场景见 [`docs/requirement.md`](docs/requirement.md)。

## 🏗️ 系统架构（How）

详见 [`docs/arch_v1.md`](docs/arch_v1.md)。V1 为「数据接入 → 离线索引（切片/向量化/Chroma）→ 向量检索 → 带拒答的生成」基础链路；后续迭代加入混合检索、Reranker、查询改写、FastAPI 流式服务与 Docker 部署。

## 📊 关键指标（Results）

> 评估闭环：30 题人工 golden set + RAGAS。数字随迭代更新。

| 指标 | Baseline | 当前 |
|---|---|---|
| hit@4 | TODO | TODO |
| 人工评分（1–5）| TODO | TODO |
| answer_relevancy | TODO | TODO |

## 🎬 Demo

> TODO：V1 跑通后录制 CLI 截图 / GIF。

## 🚀 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
cp .env.example .env   # 然后填入你的 API Key

# 4. 构建索引（把 data/raw 下的语料切片入库）
python scripts/build_index.py

# 5. 提问
python main.py "机房温度过高怎么处理"
```

## 📂 目录结构

```
.
├── data/raw/        # 原始语料（PDF / TXT / 国标）
├── docs/            # 需求文档 + 架构图
├── src/             # 核心模块（loader/splitter/embeddings/vector_store/retriever/generator/rag_chain）
├── scripts/         # build_index.py 等离线脚本
├── eval/            # 评估集 + 评估脚本
├── experiments/     # 优化实验脚本与结果
└── notes/           # 学习笔记 / troubleshooting
```

## 🗺️ 路线图

- [x] V1：基础 RAG 链路（W1）
- [ ] V2：混合检索 + Reranker + 查询改写 + RAGAS（W2）
- [ ] V3：FastAPI 流式 + Streamlit + Docker + 可观测（W3）

---

*本项目是个人 AI 解决方案工程师能力建设的一部分，以 IDC 业务理解为支点。*
