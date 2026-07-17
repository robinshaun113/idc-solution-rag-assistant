# 🏭 IDC 运维智能知识库助手

> 面向数据中心一线运维场景的 RAG 智能问答系统。
> 把散落的 PDF 手册变成可对话、可溯源的知识接口——用自然语言提问，秒级拿到带来源出处的答案。

![python](https://img.shields.io/badge/python-3.12-blue) ![api](https://img.shields.io/badge/API-FastAPI-green) ![chroma](https://img.shields.io/badge/Chroma-1.5-orange) ![status](https://img.shields.io/badge/V4-evidence--ready-brightgreen)

---

## 📖 为什么做这个项目

在中国移动 IDC 实习时，作为新人遇到不懂的问题只有两条路：**翻 PDF 手册，或者打断带教导师**。

PDF 散落在不同文件夹里，搜索靠 `Ctrl+F`——运气不好翻十几分钟也找不到。问导师倒是快，但问多了自己也不好意思。知识不是不存在，而是"查得慢"。

**这个项目把非结构化文档变成可对话的知识接口。** 用户用自然语言询问
“机房温度标准是多少”，系统从手册中检索相关片段并生成带出处的回答；证据不足时
返回未覆盖，而不是补充上下文之外的信息。

> 详细需求：**[docs/requirement.md](docs/requirement.md)**

---

## 🏗️ 系统架构

```
用户问题
  │
  ├─→ embedding（百炼 text-embedding-v2）
  │     │
  │     └─→ 向量检索（Chroma，cosine top-4）
  │           │
  │           └─→ LLM 生成（qwen-max，三层分级回答策略）
  │                 │
  │                 └─→ 答案 + evidence_id / 文件 / 页码证据
```

**向量库**：Chroma 持久化到 `./chroma_db/`，语料覆盖 GB50174、GBT2887、中国移动/华为白皮书。

> **线上链路说明**：BM25、Reranker、Multi-Query 和 HyDE 均作为可复现实验保留，
> 但在当前小规模知识库上没有带来可验证收益，因此默认链路主动保持纯向量检索。
> 复杂度是否上线由评测决定，而不是为了堆叠技术名词。

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

### 如何理解 Faithfulness=1.0

拒答题不计入 Faithfulness，因此这个指标不能单独代表回答质量。结合
**Answer Relevancy 0.71** 复查后，我发现约 30% 的问题属于假性拒答，并据此将
生成策略调整为“证据充分时回答／部分相关时说明范围／没有依据时拒答”三级处理。

---

## 🔎 典型问题定位：机房正压值

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

完成上述修正后，系统能够稳定返回“10Pa（GB 50174 7.4.4 条）”。

> 完整问题记录：**[notes/troubleshooting.md](notes/troubleshooting.md)** | 优化对比：**[docs/optimization_summary.md](docs/optimization_summary.md)**

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

### Docker

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
│   ├── evidence.py      # 稳定 evidence_id + 页码级证据契约
│   ├── generator.py     # LLM 生成（三层分级回答 + 流式）
│   ├── rag_chain.py     # 检索→生成编排
│   └── config.py        # .env 配置中心
├── scripts/             # build_index.py 等离线脚本
├── eval/                # 30 题评估集 + baseline/ragas 脚本
├── tests/               # 证据 ID 与 API 证据契约测试
├── experiments/         # exp01-04 切片/混合/Reranker/查询改写
├── data/raw/            # 原始语料（GB50174 · GBT2887 · 白皮书）
├── notes/               # troubleshooting.md + 学习笔记
├── logs/                # app.log（loguru 按天滚动）
├── Dockerfile           # 多阶段构建（python:3.11-slim）
└── docker-compose.yml   # app + Chroma 卷挂载
```

---

## 当前状态

- [x] 基础 RAG、生成策略与 30 题评测集
- [x] BM25、Reranker、Multi-Query、HyDE 对照实验
- [x] FastAPI 流式接口、Streamlit、Docker Compose 与请求日志
- [x] 稳定 evidence_id、页码级证据、进程缓存与上传安全校验
- [ ] 扩充公开语料和困难样本，重新评估检索策略

## 可信边界

- `/chat` 返回稳定 `evidence_id`、文件名、页码、chunk 序号和原文预览。
- 回答 Prompt 只允许引用本次上下文出现的 evidence_id。
- 当前语料规模较小，检索实验结论不外推为大规模生产指标。
- `/upload` 只安全落盘；索引更新仍是显式离线步骤，避免请求线程执行长任务。
