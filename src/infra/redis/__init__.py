# src/macro_monitor/core/redis/__init__.py
"""
Redis 模块 - 统一客户端
"""

from .client import RedisClient, init_redis, get_redis, close_redis
from .decorators import cache

__all__ = [
    'RedisClient',
    'init_redis',
    'get_redis',
    'close_redis',
    'cache',
]
