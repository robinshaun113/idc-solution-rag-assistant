"""生成器：把检索到的上下文 + 问题交给 LLM，生成带出处的答案。

关键设计（面试常问）：
- system prompt 设定"资深 IDC 运维专家"角色；
- 明确要求"只依据上下文回答，无相关信息就直说不知道"——这是抑制幻觉的拒答机制；
- 要求在答案末尾列出引用来源。
"""
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL, assert_api_key

SYSTEM_PROMPT = """你是一名资深的 IDC（互联网数据中心）运维专家。
请严格依据下面提供的【上下文】回答用户问题，要求：
1. 只使用上下文中的信息，不要编造、不要补充上下文之外的内容；
2. 如果上下文中没有足以回答问题的信息（包括只有部分沾边、不足以支撑准确回答的情况），直接回答"该问题还未覆盖在知识库中"，不要编造答案，可以建议用户咨询带班工程师寻求解决；
3. 回答要条理化，能分点就分点；
4. 在答案最后用"参考来源："列出你用到的文件名。

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
    )


def generate(question: str, docs: list[Document]) -> str:
    """根据问题和上下文切片生成答案。"""
    llm = get_llm()
    chain = _prompt | llm
    resp = chain.invoke({"context": _format_context(docs), "question": question})
    return resp.content
