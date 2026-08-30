# src/macro_monitor/core/redis/decorators.py
"""
Redis 装饰器
"""

import functools
import asyncio
from typing import Optional, Callable, Any

from .client import get_redis


def cache(key: Optional[str] = None, ttl: Optional[int] = None) -> Callable:
    """
    缓存装饰器：将异步函数返回值缓存到 Redis

    使用示例:
        @cache(ttl=3600)
        async def get_data():
            return {"key": "value"}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            redis = get_redis()
            if not redis:
                # Redis 未初始化，直接执行原函数
                return await func(*args, **kwargs)

            cache_key = key or f"cache:{func.__module__}:{func.__name__}"

            # 缓存获取/设置的逻辑
            cached = await redis.get(cache_key)
            if cached is not None:
                try:
                    import json
                    return json.loads(cached)
                except (json.JSONDecodeError, TypeError):
                    return cached

            # 执行原函数
            result = await func(*args, **kwargs)

            # 存入缓存
            await redis.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator