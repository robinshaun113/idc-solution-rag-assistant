"""Streamlit 前端 — IDC 解决方案 RAG 助手对话界面。

启动（项目根目录、已激活 .venv）：
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 streamlit run web/app.py

功能：
  - 对话窗口（支持流式逐字展示）
  - 显示引用来源
  - 文件上传（追加知识库）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from retriever import retrieve
from generator import generate, generate_stream
from config import assert_api_key

st.set_page_config(page_title="IDC RAG 助手", page_icon="🏭", layout="wide")
st.title("🏭 IDC 解决方案 RAG 助手")
st.caption("面向数据中心行业的 AI 知识库问答系统")

# ── 侧边栏：文件上传 ──
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded = st.file_uploader("上传新文档（PDF/TXT）", type=["pdf", "txt"])
    if uploaded:
        dest = ROOT / "data" / "raw" / uploaded.name
        dest.write_bytes(uploaded.read())
        st.success(f"已保存：{uploaded.name}（需重建索引生效）")
        st.info("运行 `python scripts/build_index.py` 重建索引")

    st.divider()
    st.caption("知识库：GB50174 · GBT2887 · 中国移动白皮书 · 华为白皮书")

# ── 聊天区域 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# 历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for s in msg["sources"]:
                    st.caption(f"• {s['source']}")

# 输入框
if question := st.chat_input("输入你的 IDC 运维问题…"):
    # 显示用户问题
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 检索 + 流式生成
    with st.chat_message("assistant"):
        try:
            docs = retrieve(question)
            # 流式输出
            placeholder = st.empty()
            full_answer = ""
            for token in generate_stream(question, docs):
                full_answer += token
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)

            # 来源
            sources = [
                {"source": d.metadata.get("source", "unknown")}
                for d in docs
            ]
            with st.expander("📎 参考来源"):
                for i, d in enumerate(docs, 1):
                    preview = d.page_content[:100].replace("\n", " ")
                    st.caption(f"{i}. [{d.metadata['source']}] {preview}...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
            })
        except Exception as e:
            st.error(f"出错了：{e}")
