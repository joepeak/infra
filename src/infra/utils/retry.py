"""infra.utils.retry：通用重试机制（20260829 升级）。

升级要点（采纳 dramacraft 设计）：
- @dataclass 替代老式 __init__ 类——更 Pythonic
- deny_exceptions 黑名单（deny > allow）——精准控制（不重试 KeyError/ValueError 等程序错误）
- 失败 raise 原始异常（不 return None）——调用方 try/except 清晰
- .copy(**override) 单点配置覆盖
- 去 requests 依赖——通用（不绑死 HTTP）
- 默认 exceptions 改用通用类（ConnectionError, TimeoutError, OSError）

用法：
    from infra.utils.retry import retry_async, RetryConfig

    cfg = RetryConfig.default()
    result = await retry_async(http_get, url, config=cfg)

    @retry_async_deco(config=RetryConfig.aggressive())
    async def fetch(): ...
"""
import asyncio
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (Callable, TypeVar, Any, Optional, Tuple, Type, Dict)

from infra.logger import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


# 默认重试异常（通用——非 HTTP 专属）
_DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError, TimeoutError, OSError,
)


@dataclass
class RetryConfig:
    """重试配置——纯数据对象，可自由拷贝修改。"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    allow_exceptions: Tuple[Type[Exception], ...] = _DEFAULT_RETRY_EXCEPTIONS
    deny_exceptions: Tuple[Type[Exception], ...] = ()

    def copy(self, **override) -> "RetryConfig":
        """拷贝并局部覆盖参数——方便单次调用修改配置。"""
        data = {k: v for k, v in self.__dict__.items()}
        data.update(override)
        return RetryConfig(**data)

    @classmethod
    def default(cls) -> "RetryConfig":
        return cls()

    @classmethod
    def aggressive(cls) -> "RetryConfig":
        """激进：更多重试，等待更短，适合网络抖动场景。"""
        return cls(max_retries=10, base_delay=0.5, max_delay=10.0, exponential_base=1.5)

    @classmethod
    def conservative(cls) -> "RetryConfig":
        """保守：少重试，长等待，适合限流严格服务。"""
        return cls(max_retries=2, base_delay=5.0, max_delay=60.0, exponential_base=2.0)


class RetryExecutor:
    """重试执行器（同步/异步统一支持）。"""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config: RetryConfig = config or RetryConfig.default()

    def _calc_delay(self, attempt: int) -> float:
        """计算退避延迟——带上限 + jitter。"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        if self.config.jitter:
            # ±50% jitter
            delay *= 0.5 + random.random()
        return delay

    def _should_retry(self, exc: Exception) -> bool:
        """
        判断是否应重试。
        规则：
        1. 命中 deny_exceptions 黑名单 → False（绝不再试）
        2. 在 allow_exceptions 白名单 → True
        3. 其它所有异常 → False
        """
        # 黑名单优先
        if self.config.deny_exceptions and isinstance(exc, self.config.deny_exceptions):
            return False
        if self.config.allow_exceptions and isinstance(exc, self.config.allow_exceptions):
            return True
        return False

    def should_retry(self, exc: Exception) -> bool:
        """公开方法——判断异常是否应被重试（暴露给业务使用）。"""
        return self._should_retry(exc)

    def execute_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """同步执行带重试——失败 raise 原始异常。"""
        last_exc: Optional[Exception] = None
        cfg = self.config
        for attempt in range(cfg.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if not self._should_retry(e) or attempt >= cfg.max_retries:
                    break
                delay = self._calc_delay(attempt)
                func_name = getattr(func, "__name__", str(func))
                logger.warning(
                    f"[Retry] {func_name} attempt={attempt+1}/{cfg.max_retries+1} "
                    f"error={type(e).__name__}, sleep={delay:.2f}s"
                )
                time.sleep(delay)

        # 全部失败——raise 原始异常
        assert last_exc is not None
        raise last_exc

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """异步执行带重试——失败 raise 原始异常。

        支持同步函数（在线程池中运行，避免阻塞事件循环）。
        """
        last_exc: Optional[Exception] = None
        cfg = self.config
        for attempt in range(cfg.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # 同步函数在线程池中运行，避免阻塞事件循环
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                # 防御：partial(async_func, ...) 在 iscoroutinefunction 里会判 False，
                # 走线程池分支调一次，拿到的是 coroutine 对象不是真值。补一下 await。
                import inspect
                if inspect.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                last_exc = e
                if not self._should_retry(e) or attempt >= cfg.max_retries:
                    break
                delay = self._calc_delay(attempt)
                func_name = getattr(func, "__name__", str(func))
                logger.warning(
                    f"[Retry] {func_name} attempt={attempt+1}/{cfg.max_retries+1} "
                    f"error={type(e).__name__}, sleep={delay:.2f}s"
                )
                await asyncio.sleep(delay)

        # 全部失败——raise 原始异常
        assert last_exc is not None
        raise last_exc


# ==================== 函数式 API ====================

def retry_sync(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """同步函数重试（函数式调用）——失败 raise 原始异常。"""
    return RetryExecutor(config).execute_sync(func, *args, **kwargs)


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """异步函数重试（函数式调用）——失败 raise 原始异常。"""
    return await RetryExecutor(config).execute_async(func, *args, **kwargs)


# ==================== 装饰器 ====================

def retry_sync_deco(config: Optional[RetryConfig] = None):
    """同步函数重试装饰器。"""
    cfg = config or RetryConfig.default()
    exe = RetryExecutor(cfg)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return exe.execute_sync(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_async_deco(config: Optional[RetryConfig] = None):
    """异步函数重试装饰器。"""
    cfg = config or RetryConfig.default()
    exe = RetryExecutor(cfg)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await exe.execute_async(func, *args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "RetryConfig",
    "RetryExecutor",
    "retry_sync",
    "retry_async",
    "retry_sync_deco",
    "retry_async_deco",
]
