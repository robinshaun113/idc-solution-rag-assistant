"""生成器 V2 — Day 13 生成层优化。

相比 V1 的关键变化（面试重点）：
1. 角色升级：运维专家 → IDC 行业解决方案工程师（匹配目标岗位叙事）
2. 拒答从"一刀切"改为三层分级回答策略——能答多少答多少，缺的诚实标注
   （根因：V1 的"部分沾边也拒答"导致假性拒答率 30%，五轮实验排除检索侧后定点修复）
3. 引用溯源：答案末尾标注来源文件名 + 片段编号，增强可信度
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL, assert_api_key, get_http_client

SYSTEM_PROMPT = """你是一名资深的 IDC 行业解决方案工程师。

请严格依据下面提供的【上下文】回答用户问题。你的回答策略分三层：

a) 上下文包含完整答案 → 直接、准确地回答，使用上下文中的具体数据和规范条文。
b) 上下文有部分相关信息但不够完整 → 先回答能确定的部分（引用具体来源），
   再诚实说明"以下信息暂未在当前知识库中找到：xxx"，并建议咨询专业工程师。
c) 上下文完全无关 → 回答"该问题暂未覆盖在知识库中"，建议用户提供更多背景或咨询专业工程师。

要求：
1. 只使用上下文中的信息，绝不编造上下文之外的内容。
2. 回答条理化，能分点就分点。
3. 在答案末尾加上"参考来源"，列出每个引用的来源文件名（如 [GB 50174数据中心设计规范.pdf]）。

【上下文】
{context}
"""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def _format_context(docs: list[Document]) -> str:
    """把检索到的切片拼成带来源标注的上下文文本。"""
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[片段{i} | 来源：{d.metadata['source']}]\n{d.page_content}")
    return "\n\n".join(blocks)


def get_llm() -> ChatOpenAI:
    assert_api_key()
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.2,  # 运维场景要稳，温度调低
        http_client=get_http_client(),
    )


def generate(question: str, docs: list[Document]) -> str:
    """根据问题和上下文切片生成答案（一次性返回完整结果）。"""
    llm = get_llm()
    chain = _prompt | llm
    resp = chain.invoke({"context": _format_context(docs), "question": question})
    return resp.content


def generate_stream(question: str, docs: list[Document]):
    """流式生成：边生成边返回 token，适合 SSE（Server-Sent Events）。

    LangChain ChatOpenAI 的 .stream() 方法返回一个迭代器，
    每次 yield 一个包含 token 的 chunk。
    """
    llm = get_llm()
    chain = _prompt | llm
    for chunk in chain.stream({"context": _format_context(docs), "question": question}):
        if chunk.content:
            yield chunk.content
