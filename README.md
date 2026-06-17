# 🏭 IDC 运维智能知识库助手

> 面向数据中心一线运维场景的 RAG 智能问答系统。
> 把散落的 PDF 手册变成可对话、可溯源的知识接口——用自然语言提问，秒级拿到带来源出处的答案。

![python](https://img.shields.io/badge/python-3.12-blue) ![fastapi](https://img.shields.io/badge/FastAPI-0.49-green) ![chroma](https://img.shields.io/badge/Chroma-1.5-orange) ![status](https://img.shields.io/badge/W3-engineering--complete-brightgreen)

---

## 📖 为什么做这个项目

在中国移动 IDC 实习时，作为新人遇到不懂的问题只有两条路：**翻 PDF 手册，或者打断带教导师**。

PDF 散落在不同文件夹里，搜索靠 `Ctrl+F`——运气不好翻十几分钟也找不到。问导师倒是快，但问多了自己也不好意思。知识不是不存在，而是"查得慢"。

**这个项目把非结构化文档变成可对话的知识接口。** 用户用自然语言问"机房温度标准是多少"，系统自动从所有手册里检索最相关的片段，交给 LLM 生成带来源出处的答案。查不到就诚实说不知道，不编造。

> 详细需求：**[docs/requirement.md](docs/requirement.md)**

---

## 🏗️ 系统架构

```
用户问题
  │
  ├─→ embedding（百炼 text-embedding-v2）
  │     │
  │     └─→ 向量检索（Chroma，cosine 相似度）
  │           │
  │           ├─→ 稀疏检索（BM25 + jieba 中文分词）
  │           │
  │           └─→ 混合结果 top-20
  │                 │
  │                 └─→ Reranker（BGE-reranker-base）精排 top-4
  │                       │
  │                       └─→ LLM 生成（qwen-max，三层分级回答策略）
  │                             │
  │                             └─→ 用户拿到答案 + 引用来源
```

**向量库**：Chroma 持久化到 `./chroma_db/`，语料覆盖 GB50174、GBT2887、中国移动/华为白皮书。

> 架构设计文档：**[docs/arch_v1.md](docs/arch_v1.md)**

---

## 📊 关键指标

> 30 题人工 golden set + 自实现 RAGAS 四项指标。完整评估报告见 **[eval/ragas_report.md](eval/ragas_report.md)**。

| 指标 | Baseline (V1) | 优化后 (V2) | 提升 |
|---|---|---|---|
| **hit@4** | 96% | **100%** | +4% |
| Context Precision | — | 0.36 | — |
| Context Recall | — | 0.76 | — |
| Faithfulness | — | 1.0 | — |
| Answer Relevancy | — | 0.71 | — |
| 人工评分（1–5） | 3.76 | — | — |
| 拒答率 | 100%（5/5 无答案题） | — | — |

### 为什么 Faithfulness=1.0 不是满分

拒答题不计入 Faithfulness——这是 RAGAS 的盲区。真正反映问题的是 **Answer Relevancy 0.71**，它抓到了 30% 的假性拒答（V1 prompt 的"部分沾边也拒答"规则太激进）。Day 13 三层分级回答策略修复后翻转。

---

## 🐛 最深的坑：id5 追击记

**现象**：问"IDC 机房洁净度标准是多少 Pa"，连续 5 轮实验回答"未覆盖"。

**排查过程**：
1. Reranker 重排 → 没用
2. Multi-Query 改写 → 没用
3. HyDE 先写假答案再检索 → 没用
4. 怀疑是生成层 prompt 太激进 → 改了，还是没用

**根因**：三道锁叠加——

| 层 | 问题 | 修复 |
|---|---|---|
| **向量稀释** | 答案 chunk 含 4 个主题，embedding 方向模糊，排名 **52/871** | — |
| **中文分词** | BM25 把"机房温度过高"当整个词，完全失配 | jieba 分词 |
| **Prompt 拒答** | 旧规则"部分沾边也拒答"阻止 LLM 回答 | 三层分级策略 |

三道锁全解，id5 才翻转："10Pa（GB50174 7.4.4 条）"。

> 完整踩坑记录：**[notes/troubleshooting.md](notes/troubleshooting.md)** | 优化对比：**[docs/optimization_summary.md](docs/optimization_summary.md)**

---

## 🚀 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
cp .env.example .env   # 编辑填入 DASHSCOPE_API_KEY（百炼 sk- 开头）

# 4. 构建索引
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/build_index.py

# 5. CLI 提问
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python main.py "机房温度过高怎么处理"
```

### API 模式

```bash
# 启动服务
uvicorn app.main:app --reload --port 8000

# 问答
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"机房温度过高怎么处理"}'

# 流式输出（SSE）
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"机房温度标准是多少"}'

# 健康检查
curl http://localhost:8000/health
```

### 前端界面

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 streamlit run web/app.py
```

### Docker（代码已就绪，本地 BIOS 虚拟化未开，可用 GitHub Codespaces）

```bash
docker compose up -d
```

---

## 📂 目录结构

```
.
├── app/                 # FastAPI 服务 + 日志中间件
│   ├── main.py          # /chat /chat/stream /upload /health
│   └── middleware.py    # loguru 日志中间件（request_id + 耗时 + token）
├── web/                 # Streamlit 前端
│   └── app.py
├── src/                 # RAG 核心模块
│   ├── loaders.py       # PDF/TXT 文档加载
│   ├── splitter.py      # 文本切片
│   ├── embeddings.py    # 百炼 text-embedding-v2 封装
│   ├── vector_store.py  # Chroma 持久化
│   ├── retriever.py     # 向量检索
│   ├── hybrid_retriever.py  # BM25+向量混合检索（RRF 融合）
│   ├── reranker.py      # BGE-reranker-base 重排
│   ├── generator.py     # LLM 生成（三层分级回答 + 流式）
│   ├── rag_chain.py     # 检索→生成编排
│   └── config.py        # .env 配置中心
├── scripts/             # build_index.py 等离线脚本
├── eval/                # 30 题评估集 + baseline/ragas 脚本
├── experiments/         # exp01-04 切片/混合/Reranker/查询改写
├── data/raw/            # 原始语料（GB50174 · GBT2887 · 白皮书）
├── notes/               # troubleshooting.md + 学习笔记
├── logs/                # app.log（loguru 按天滚动）
├── Dockerfile           # 多阶段构建（python:3.11-slim）
└── docker-compose.yml   # app + Chroma 卷挂载
```

---

## 🗺️ 路线图

- [x] V1：基础 RAG 链路（W1 · Day 1–7）
- [x] V2：BM25 + Reranker + 查询改写 + RAGAS + 生成层优化（W2 · Day 8–14）
- [x] V3：FastAPI 流式 + Streamlit + Docker + loguru 可观测性（W3 · Day 15–19）
- [ ] 博客 #1 + Demo 视频（Day 20）
- [ ] 模拟面试 #2：工程化与系统设计（Day 21）

---

*本项目是 AI 解决方案工程师 / FDE 能力建设的一部分，以 IDC 业务理解为支点，三个项目覆盖 RAG 全栈 → Agent 编排 → 多模态落地。*

*"一个能力，服务多种客户"是传统 SWE。"一个客户，使用多种能力"是 FDE。*
