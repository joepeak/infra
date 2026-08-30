"""infra.llm.client：LLM 客户端抽象基类 + Provider 实现。

设计：
- LLMClient 抽象基类——子类实现具体 provider
- 集成 infra.utils.retry 做重试
- 集成 infra.llm.usage 做 token 聚合
- 自动记录 token 用量
"""
import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional

from infra.utils.retry import RetryConfig
from infra.llm.models import LLMRequest, LLMResponse
from infra.llm.usage import record_usage
from infra.llm.retry import LLMRetryConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """LLM 客户端抽象基类——子类实现 ainvoke/invoke。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: float = 120.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = model
        self.timeout = timeout
        if retry_config is None:
            retry_config = LLMRetryConfig.default().retry_config.copy(max_retries=max_retries)
        self.retry_config = retry_config
        self._initialize_provider()
        logger.info(
            f"Initializing LLMClient | provider={self.provider}, "
            f"model={self.default_model}, max_retries={max_retries}"
        )

    @abstractmethod
    def _initialize_provider(self) -> None:
        """子类初始化 provider 客户端（如 AsyncOpenAI）——必须实现。"""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider 名（'openai' / 'anthropic' / 'deepseek' / 'local'）。"""

    @abstractmethod
    async def ainvoke(self, request: LLMRequest) -> LLMResponse:
        """异步调用——子类必须实现。"""

    @abstractmethod
    def invoke(self, request: LLMRequest) -> LLMResponse:
        """同步调用——子类必须实现。"""

    def _record_usage_from_response(self, response: LLMResponse) -> None:
        """记录 token 用量——子类 ainvoke/invoke 后调用。"""
        if response.usage:
            record_usage(response.usage)


# ===========================================================================
# OpenAI Provider（第一期实现）
# ===========================================================================

class OpenAIClient(LLMClient):
    """OpenAI / DeepSeek 等 OpenAI 兼容 API 客户端。"""

    def _initialize_provider(self) -> None:
        from openai import OpenAI, AsyncOpenAI
        kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._sync_client = OpenAI(**kwargs)  # type: ignore[arg-type]
        self._async_client = AsyncOpenAI(**kwargs)  # type: ignore[arg-type]

    @property
    def provider(self) -> str:
        return "openai"

    async def ainvoke(self, request: LLMRequest) -> LLMResponse:
        from infra.utils.retry import retry_async
        # request.model 显式给出时优先（partial 里不重复传 model，避免 **kwargs 冲突）
        kwargs = request.to_openai_kwargs()
        if "model" in kwargs:
            create_func = partial(
                self._async_client.chat.completions.create,
                **kwargs,
            )
        else:
            model = request.model or self.default_model
            create_func = partial(
                self._async_client.chat.completions.create,
                model=model,
                **kwargs,
            )
        response = await retry_async(create_func, config=self.retry_config)
        result = LLMResponse.from_openai_response(response)
        self._record_usage_from_response(result)
        return result

    def invoke(self, request: LLMRequest) -> LLMResponse:
        from infra.utils.retry import retry_sync
        kwargs = request.to_openai_kwargs()
        if "model" in kwargs:
            create_func = partial(
                self._sync_client.chat.completions.create,
                **kwargs,
            )
        else:
            model = request.model or self.default_model
            create_func = partial(
                self._sync_client.chat.completions.create,
                model=model,
                **kwargs,
            )
        response = retry_sync(create_func, config=self.retry_config)
        result = LLMResponse.from_openai_response(response)
        self._record_usage_from_response(result)
        return result


# ===========================================================================
# Factory
# ===========================================================================

class LLMFactory:
    """LLM 工厂——按 config['provider'] 选 provider 客户端。"""

    @staticmethod
    def create(config: dict) -> LLMClient:
        provider = config.get("provider", "openai").lower()
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        model = config.get("default_model", "gpt-4o")
        max_retries = config.get("max_retries", 3)
        timeout = config.get("timeout", 120.0)

        if provider == "openai":
            # narrow api_key: Any | None → str（None 时抛错——空 api_key 是真业务 bug）
            if not api_key:
                raise ValueError(
                    "LLM config 缺 api_key——请在 config dict 配 api_key 或环境变量 LLM_API_KEY"
                )
            return OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_retries=max_retries,
                timeout=timeout,
            )
        # 未来扩展点：anthropic / deepseek / local
        # elif provider == "anthropic":
        #     return AnthropicClient(...)
        raise ValueError(
            f"unsupported provider: {provider}（已实现：openai；待扩展：anthropic/deepseek/local）"
        )


# ===========================================================================
# 全局单例 + init_llm_client / get_llm_client
# ===========================================================================

import os

_llm_client: Optional[LLMClient] = None


def _pick(yaml_val: Any, env_key: str) -> Any:
    """优先级：yaml 有效值（非空且非占位符）→ 环境变量。

    占位符集合防止 'YOUR_API_KEY_HERE' 这样的字面值被误用。
    """
    placeholder = {"", "YOUR_API_KEY_HERE", None}
    if yaml_val and str(yaml_val) not in placeholder:
        return yaml_val
    return os.getenv(env_key)


async def init_llm_client(config: dict) -> Optional[LLMClient]:
    """初始化全局 LLM 客户端（按配置创建）。

    配置优先级（高→低）：
      1. yaml llm.* 节（且非占位符/非空）
      2. 环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（.env 由 bootstrap 加载）

    若两边都未配置 api_key——返 None（不抛错，让上层检查）。
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    llm_config = config.get("llm", {}) or {}

    api_key = _pick(llm_config.get("api_key"), "LLM_API_KEY")
    if not api_key:
        logger.error(
            "LLM API key 未配置（yaml llm.api_key 与环境变量 LLM_API_KEY 均为空），LLM client 不可用。"
        )
        return None

    base_url = _pick(llm_config.get("base_url"), "LLM_BASE_URL")
    model = _pick(llm_config.get("default_model"), "LLM_MODEL")

    _llm_client = LLMFactory.create({
        "provider": llm_config.get("provider", "openai"),
        "api_key": str(api_key),
        "base_url": str(base_url) if base_url else None,
        "default_model": str(model) if model else "gpt-4o",
        "max_retries": llm_config.get("max_retries", 5),
        "timeout": llm_config.get("timeout", 120.0),
    })
    logger.info(
        f"LLM global client initialized. provider={_llm_client.provider}, "
        f"model={_llm_client.default_model}"
    )
    return _llm_client


def get_llm_client() -> Optional[LLMClient]:
    """获取全局 LLM 客户端实例。"""
    if _llm_client is None:
        logger.warning("LLM client accessed before initialization or was not configured.")
    return _llm_client
