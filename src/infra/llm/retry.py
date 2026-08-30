"""infra.llm.retry：LLM 专用重试配置。

设计：
- 继承 infra.utils.retry.RetryConfig——通用 retry 机制
- 默认 deny_exceptions=认证类错误（401/403/404）——绝不再试
- 默认 allow_exceptions=限流/超时/网络类——可重试
"""
from infra.utils.retry import RetryConfig
from typing import Tuple  # noqa: F811


def _build_default_deny_exceptions() -> Tuple[type, ...]:
    """认证/权限类异常——绝不再试。"""
    deny = []
    try:
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError
        deny.extend([AuthenticationError, BadRequestError, PermissionDeniedError])
    except ImportError:
        pass
    return tuple(deny)


def _build_default_allow_exceptions() -> Tuple[type, ...]:
    """限流/超时/网络类异常——可重试。"""
    allow: list = [ConnectionError, TimeoutError, OSError]
    try:
        from openai import APITimeoutError, RateLimitError, APIError
        allow.extend([APITimeoutError, RateLimitError, APIError])
    except ImportError:
        pass
    return tuple(allow)


class LLMRetryConfig:
    """LLM 专用重试配置——包装通用 RetryConfig。"""

    def __init__(self, retry_config: RetryConfig):
        self.retry_config = retry_config

    def should_retry(self, exc: Exception) -> bool:
        """公共方法——判断异常是否应被重试（暴露给测试/业务使用）。"""
        # 委托给 RetryExecutor 默认行为——但保持轻量级直接判断
        from infra.utils.retry import RetryConfig as _RC
        # 用临时 RetryConfig 风格判断（deny > allow > 其他 False）
        if self.retry_config.deny_exceptions and isinstance(exc, self.retry_config.deny_exceptions):
            return False
        if self.retry_config.allow_exceptions and isinstance(exc, self.retry_config.allow_exceptions):
            return True
        return False

    @classmethod
    def default(cls) -> "LLMRetryConfig":
        """默认配置——认证不重试/限流超时可重试。"""
        rc = RetryConfig.default().copy(
            deny_exceptions=_build_default_deny_exceptions(),
            allow_exceptions=_build_default_allow_exceptions(),
        )
        return cls(rc)

    @classmethod
    def aggressive(cls) -> "LLMRetryConfig":
        """激进——更多重试——适合网络抖动。"""
        rc = (
            RetryConfig.aggressive()
            .copy(
                deny_exceptions=_build_default_deny_exceptions(),
                allow_exceptions=_build_default_allow_exceptions(),
            )
        )
        return cls(rc)

    @classmethod
    def conservative(cls) -> "LLMRetryConfig":
        """保守——少重试——适合严格限流服务。"""
        rc = (
            RetryConfig.conservative()
            .copy(
                deny_exceptions=_build_default_deny_exceptions(),
                allow_exceptions=_build_default_allow_exceptions(),
            )
        )
        return cls(rc)
