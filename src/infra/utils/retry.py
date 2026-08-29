# macro_monitor/utils/retry.py
"""
统一的重试机制，支持同步和异步函数
"""
import asyncio
import time
import random
from typing import Callable, TypeVar, Any, Optional, Tuple, Type, Union, Dict
from functools import wraps
import requests

from infra.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RetryConfig:
    """重试配置类"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2,
        jitter: bool = True,
        retry_exceptions: Tuple[Type[Exception], ...] = (
            requests.RequestException,
            requests.ConnectionError,
            requests.Timeout,
            ConnectionError,
            TimeoutError,
            ValueError,
        )
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'max_retries': self.max_retries,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'exponential_base': self.exponential_base,
            'jitter': self.jitter,
        }
    
    @classmethod
    def default(cls) -> 'RetryConfig':
        """默认配置"""
        return cls()
    
    @classmethod
    def aggressive(cls) -> 'RetryConfig':
        """激进的重试配置（更多重试，更短延迟）"""
        return cls(
            max_retries=10,
            base_delay=0.5,
            max_delay=10.0,
            exponential_base=1.5
        )
    
    @classmethod
    def conservative(cls) -> 'RetryConfig':
        """保守的重试配置（更少重试，更长延迟）"""
        return cls(
            max_retries=3,
            base_delay=5.0,
            max_delay=60.0,
            exponential_base=2
        )


class RetryExecutor:
    """
    统一的重试执行器，支持同步和异步函数
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig.default()
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算退避延迟"""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # 添加 ±50% 的随机抖动
            delay = delay * (0.5 + random.random())
        
        return delay
    
    def _should_retry(self, exception: Exception) -> bool:
        """检查是否应该重试"""
        return isinstance(exception, self.config.retry_exceptions)
    
    def execute_sync(self, func: Callable[..., T], *args, **kwargs) -> Optional[T]:
        """
        同步执行带重试的函数
        
        Example:
            result = executor.execute_sync(requests.get, url, timeout=30)
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e) or attempt == self.config.max_retries:
                    logger.error(
                        f"{func.__name__} 执行失败，不再重试: {e}"
                    )
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"{func.__name__} 失败 (尝试 {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                    f"等待 {delay:.2f} 秒后重试..."
                )
                time.sleep(delay)
        
        return None
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """
        异步执行带重试的函数
        
        Example:
            result = await executor.execute_async(async_fetch, url)
            result = await executor.execute_async(requests.get, url)  # 自动处理同步函数
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    # 同步函数在线程池中运行，避免阻塞事件循环
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    )
                    
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e) or attempt == self.config.max_retries:
                    logger.error(
                        f"{func.__name__} 执行失败，不再重试: {e}"
                    )
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"{func.__name__} 失败 (尝试 {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                    f"等待 {delay:.2f} 秒后重试..."
                )
                await asyncio.sleep(delay)
        
        return None


# ============ 装饰器（向后兼容） ============

def retry_sync_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (
        requests.RequestException,
        requests.ConnectionError,
        requests.Timeout,
        ConnectionError,
        TimeoutError,
    )
):
    """
    同步函数重试装饰器（向后兼容）
    
    Example:
        @retry_sync_decorator(max_retries=3)
        def fetch_data():
            return requests.get(url)
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_exceptions=retry_exceptions
    )
    executor = RetryExecutor(config)
    
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            return executor.execute_sync(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_async_decorator(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2,
    jitter: bool = True,
    retry_exceptions: Tuple[Type[Exception], ...] = (
        requests.RequestException,
        requests.ConnectionError,
        requests.Timeout,
        ConnectionError,
        TimeoutError,
    )
):
    """
    异步函数重试装饰器
    
    Example:
        @retry_async_decorator(max_retries=3)
        async def fetch_data():
            return await async_request(url)
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_exceptions=retry_exceptions
    )
    executor = RetryExecutor(config)
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Optional[Any]:
            return await executor.execute_async(func, *args, **kwargs)
        return wrapper
    return decorator


# ============ 函数式调用 ============

def retry_sync(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Optional[T]:
    """
    同步函数重试（函数式调用）
    
    Example:
        result = retry_sync(requests.get, url, timeout=30, config=RetryConfig.default())
    """
    executor = RetryExecutor(config)
    return executor.execute_sync(func, *args, **kwargs)


async def retry_async(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Optional[Any]:
    """
    异步函数重试（函数式调用）
    
    Example:
        result = await retry_async(async_fetch, url, config=RetryConfig.aggressive())
    """
    executor = RetryExecutor(config)
    return await executor.execute_async(func, *args, **kwargs)