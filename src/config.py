"""集中读取 .env 配置，供其他模块复用。

百炼（DashScope）提供 OpenAI 兼容接口，所以 embedding 和 LLM 都用
langchain-openai 的类，只需把 base_url 指向百炼的兼容端点。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env
# override=True：让 .env 文件值覆盖已存在的同名环境变量，
# 避免 shell 里残留的旧值（如旧 API Key）盖过文件，造成"改了文件却不生效"
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env", override=True)

# 清除 Windows 系统代理（开发机常设 127.0.0.1:7890/7897 指向 Clash/V2Ray，
# 代理客户端没开时所有 HTTP 请求都会 Connection Refused）
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
             "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
    os.environ.pop(_key, None)

# 百炼 OpenAI 兼容端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
CHROMA_PERSIST_DIR = str(_ROOT / "chroma_db")


def assert_api_key() -> None:
    """运行真正调用 API 前检查 Key 是否已填，给出友好提示。"""
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your_dashscope_key_here":
        raise RuntimeError(
            "未检测到有效的 DASHSCOPE_API_KEY。\n"
            "请打开项目根目录的 .env 文件，把 DASHSCOPE_API_KEY 改成你的真实 Key（sk- 开头）。\n"
            "百炼控制台：https://bailian.console.aliyun.com"
        )


# ── 共享 httpx 客户端（禁用系统代理）──
# 问题背景：Windows 上 httpx 会自动读取注册表里的系统代理设置
# （HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings\ProxyServer），
# 代理客户端（Clash/V2Ray 在 127.0.0.1:7890）没开时所有 LLM API 调用都会
# Connection Refused。解决方案：创建一个 proxy=False 的 httpx 客户端，
# 传给 OpenAI / ChatOpenAI，让它直连百炼。
# 注意：_http_client 用函数懒加载，不在模块顶层 import httpx（给 streamlit
# 等其他入口留路，它们未必需要 httpx 就可以 import config）。
_http_client = None

import httpx                                                  # noqa: E402

def get_http_client() -> httpx.Client:
    """返回一个不走系统代理的 httpx 同步客户端（单例）。"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(proxy=None)
    return _http_client
