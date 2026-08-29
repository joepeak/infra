"""infra.llm：通用 LLM 客户端（Provider 抽象）。

模块：
- infra.llm.client：LLMClient 抽象基类 + OpenAIClient + LLMFactory
- infra.llm.models：LLMRequest / LLMResponse pydantic
- infra.llm.retry：LLMRetryConfig（LLM 专用重试配置）
- infra.llm.usage：进程内 Token 用量聚合（ContextVar + Lock + defaultdict）

用法：
    from infra.llm import init_llm_client, get_llm_client, LLMRequest
    from pathlib import Path

    # 启动时（由 infra.bootstrap 调）
    await init_llm_client(config)

    # 业务使用
    client = get_llm_client()
    if client is None:
        raise RuntimeError("LLM 未配置")
    response = await client.ainvoke(
        LLMRequest(messages=[{"role": "user", "content": "hello"}])
    )
    print(response.content)
"""
from infra.llm.client import (
    LLMClient,
    OpenAIClient,
    LLMFactory,
    init_llm_client,
    get_llm_client,
)
from infra.llm.models import LLMRequest, LLMResponse
from infra.llm.retry import LLMRetryConfig
from infra.llm.usage import (
    record_usage,
    get_usage_summary,
    reset_usage_summary,
    log_usage_summary,
    set_llm_purpose,
    reset_llm_purpose,
)


def _reset_for_test():
    """测试用——重置全局 LLM 单例。"""
    import infra.llm.client as _client_mod
    _client_mod._llm_client = None
    reset_usage_summary()


__all__ = [
    # client
    "LLMClient", "OpenAIClient", "LLMFactory",
    "init_llm_client", "get_llm_client",
    # models
    "LLMRequest", "LLMResponse",
    # retry
    "LLMRetryConfig",
    # usage
    "record_usage", "get_usage_summary", "reset_usage_summary",
    "log_usage_summary", "set_llm_purpose", "reset_llm_purpose",
]
