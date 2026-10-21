# src/macro_monitor/core/redis/client.py
"""
Redis 统一客户端
提供连接管理、基础操作、分布式锁
"""

import json
import asyncio
import time
import uuid
from typing import AsyncIterator, Optional, Any, Dict, List, Set, Callable
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from infra.logger import get_logger
from infra.exceptions import RedisError

logger = get_logger(__name__)


class RedisClient:
    """Redis 统一客户端（异步）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client: Optional[aioredis.Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._lock_tokens: Dict[str, str] = {}
        self._init_connection()
    
    def _init_connection(self) -> None:
        """初始化连接"""
        try:
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 6379)
            db = self.config.get('db', 0)
            password = self.config.get('password') or None
            
            # 获取连接池配置（支持两种格式）
            pool_config = self.config.get('connection_pool', {})
            max_connections = pool_config.get('max_connections', 
                               self.config.get('max_connections', 20))
            socket_timeout = pool_config.get('socket_timeout', 
                           self.config.get('socket_timeout', 5))
            socket_connect_timeout = pool_config.get('socket_connect_timeout',
                                   self.config.get('socket_connect_timeout', 5))
            
            self._pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            logger.info(f"Redis 连接成功: {host}:{port}")
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            raise RedisError(f"Redis 连接失败: {e}")
    
    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RedisError("Redis 客户端未初始化")
        return self._client

    @property
    def client(self) -> aioredis.Redis:
        """底层 redis.asyncio 客户端（公开访问器）。

        用于 pipeline / scan_iter / 批量命令等本类未封装的能力。
        优先用本类提供的方法（keys / scan_iter / pipeline 等）；
        确实需要原生能力时再用这个属性——不要再去碰 ``_get_client()``。
        """
        return self._get_client()

    # ========== Key 扫描 ==========

    async def keys(self, pattern: str = "*") -> List[str]:
        """按模式列出所有匹配的 key（返回 str）。

        ⚠️ 生产环境慎用：``KEYS`` 会阻塞 Redis 单线程。
        大 keyspace 请改用 :meth:`scan_iter`（增量扫描，不阻塞）。
        """
        raw = await self._get_client().keys(pattern)
        return [self._decode_key(k) for k in raw]

    async def scan_iter(
        self, pattern: str = "*", count: int = 100
    ) -> AsyncIterator[str]:
        """增量扫描匹配的 key（生产推荐——不阻塞 Redis）。

        用法::

            async for key in redis.scan_iter("llm:article_body:*"):
                ...

        Args:
            pattern: 匹配模式（glob 语法）
            count: 每批扫描的 hint 条数（越大吞吐越高，单次阻塞越久）
        """
        async for k in self._get_client().scan_iter(match=pattern, count=count):
            yield self._decode_key(k)

    @staticmethod
    def _decode_key(key: Any) -> str:  # noqa: ANN401
        """key 统一成 str。

        连接池开了 ``decode_responses=True``，正常返回就是 str；
        这里保留 bytes 兜底（自定义连接池 / mock 场景）。
        """
        if isinstance(key, bytes):
            return key.decode("utf-8", errors="replace")
        return str(key)

    # ========== 批量操作 ==========

    def pipeline(self, transaction: bool = True) -> Any:  # noqa: ANN401
        """创建 pipeline（批量命令，减少 RTT）。

        用法::

            async with redis.pipeline() as pipe:
                pipe.set("k1", "v1")
                pipe.get("k1")
                await pipe.execute()

        底层返回 ``redis.asyncio.client.Pipeline``。
        """
        return self._get_client().pipeline(transaction=transaction)

    async def mget(self, *keys: str) -> List[Any]:  # noqa: ANN401
        """批量取多个 key 的值（一次 RTT）。

        不存在的 key 对应位置为 None。
        """
        if not keys:
            return []
        return list(await self._get_client().mget(*keys))

    async def mset(self, mapping: Dict[str, Any]) -> bool:  # noqa: ANN401
        """批量写多个 key（一次 RTT）。

        Args:
            mapping: {key: value}；value 为 dict/list 时自动 JSON 序列化
        """
        if not mapping:
            return True
        payload: Dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(v, (dict, list)):
                payload[k] = json.dumps(v, ensure_ascii=False)
            elif not isinstance(v, str):
                payload[k] = str(v)
            else:
                payload[k] = v
        return bool(await self._get_client().mset(payload))

    async def delete_pattern(self, pattern: str, batch: int = 500) -> int:
        """按模式批量删除 key（用 SCAN 增量扫描，不阻塞 Redis）。

        Args:
            pattern: 匹配模式（glob 语法）
            batch: 每批删除的条数

        Returns:
            实际删除的 key 数量
        """
        deleted = 0
        batch_keys: List[str] = []
        async for key in self.scan_iter(pattern):
            batch_keys.append(key)
            if len(batch_keys) >= batch:
                deleted += await self.delete(*batch_keys)
                batch_keys = []
        if batch_keys:
            deleted += await self.delete(*batch_keys)
        return deleted

    # ========== 基础操作 ==========
    
    async def get(self, key: str) -> Any:  # noqa: ANN401  # bytes | str | None
        """获取键值"""
        return await self._get_client().get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置键值，支持 TTL"""
        client = self._get_client()
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif not isinstance(value, str):
            value = str(value)
        
        if ttl:
            return bool(await client.set(key, value, ex=ttl))
        return bool(await client.set(key, value))
    
    async def set_nx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        原子 SET NX（不存在才设置），可选 TTL。
        
        底层单命令 ``SET key value NX EX ttl``，共享 ``set`` 的序列化逻辑，
        但额外保证加锁原子性——避免 ``setnx`` + ``expire`` 两步非原子导致的死锁。
        """
        client = self._get_client()
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif not isinstance(value, str):
            value = str(value)
        if ttl:
            return bool(await client.set(key, value, nx=True, ex=ttl))
        return bool(await client.set(key, value, nx=True))
    
    async def delete(self, *keys: str) -> int:
        """删除键"""
        if not keys:
            return 0
        return await self._get_client().delete(*keys)
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return await self._get_client().exists(key) > 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        return await self._get_client().expire(key, ttl)
    
    async def ttl(self, key: str) -> int:
        """获取剩余生存时间"""
        return await self._get_client().ttl(key)
    
    # ========== 计数器 ==========
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        return await self._get_client().incrby(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        return await self._get_client().decrby(key, amount)
    
    # ========== Hash 操作 ==========
    
    async def hset(self, key: str, field: str, value: Any) -> int:
        """设置 Hash 字段"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif not isinstance(value, str):
            value = str(value)
        return await self._get_client().hset(key, field, value)
    
    async def hget(self, key: str, field: str) -> Any:  # noqa: ANN401  # bytes | str | None
        """获取 Hash 字段"""
        return await self._get_client().hget(key, field)
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """获取整个 Hash（redis-py 返 dict[bytes|str, bytes|str]——decode_responses=True 实际是 str）"""
        raw: Any = await self._get_client().hgetall(key)
        if not raw:
            return {}
        return {str(k): str(v) for k, v in raw.items()}
    
    async def hdel(self, key: str, *fields: str) -> int:
        """删除 Hash 字段"""
        return await self._get_client().hdel(key, *fields)
    
    # ========== Set 操作（去重用） ==========
    
    async def sadd(self, key: str, *members: str) -> int:
        """添加成员到 Set"""
        return await self._get_client().sadd(key, *members)
    
    async def sismember(self, key: str, member: str) -> bool:
        """检查是否为 Set 成员（redis-py 5+ 返 Literal[0, 1]——mypy 期望 bool）"""
        result = await self._get_client().sismember(key, member)
        return bool(result)
    
    async def smembers(self, key: str) -> Set[str]:
        """获取 Set 所有成员"""
        raw: Any = await self._get_client().smembers(key)
        return {str(m) for m in raw} if raw else set()
    
    async def srem(self, key: str, *members: str) -> int:
        """从 Set 移除成员"""
        return await self._get_client().srem(key, *members)
    
    # ========== ZSET 操作（有序集合，用于限流/排行榜） ==========
    
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """
        添加元素到有序集合
        
        Args:
            key: Redis key
            mapping: {member: score} 字典
        
        Returns:
            添加的元素数量
        """
        result = await self._get_client().zadd(key, mapping)
        return int(result) if result is not None else 0
    
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """
        按分数范围移除元素
        
        Args:
            key: Redis key
            min_score: 最小分数（包含）
            max_score: 最大分数（包含）
        
        Returns:
            移除的元素数量
        """
        return await self._get_client().zremrangebyscore(key, min_score, max_score)
    
    async def zcard(self, key: str) -> int:
        """
        获取有序集合的元素数量
        
        Args:
            key: Redis key
        
        Returns:
            元素数量
        """
        return await self._get_client().zcard(key)
    
    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list:
        """
        按索引范围获取有序集合的元素
        
        Args:
            key: Redis key
            start: 起始索引
            stop: 结束索引
            withscores: 是否返回分数
        
        Returns:
            元素列表，如果 withscores=True，返回 [(member, score), ...]
        """
        return await self._get_client().zrange(key, start, stop, withscores=withscores)
    
    async def zrangebyscore(
        self, 
        key: str, 
        min_score: float, 
        max_score: float, 
        withscores: bool = False,
        offset: Optional[int] = None,
        count: Optional[int] = None
    ) -> list:
        """
        按分数范围获取有序集合的元素
        
        Args:
            key: Redis key
            min_score: 最小分数（包含）
            max_score: 最大分数（包含）
            withscores: 是否返回分数
            offset: 偏移量（分页）
            count: 返回数量（分页）
        
        Returns:
            元素列表
        """
        client = self._get_client()
        if offset is not None and count is not None:
            return await client.zrangebyscore(
                key, min_score, max_score, 
                withscores=withscores, 
                start=offset, 
                num=count
            )
        return await client.zrangebyscore(key, min_score, max_score, withscores=withscores)
    
    async def zrem(self, key: str, *members: str) -> int:
        """
        移除有序集合中的指定元素
        
        Args:
            key: Redis key
            members: 要移除的元素
        
        Returns:
            移除的元素数量
        """
        if not members:
            return 0
        return await self._get_client().zrem(key, *members)
    
    async def zscore(self, key: str, member: str) -> Optional[float]:
        """
        获取元素的分数
        
        Args:
            key: Redis key
            member: 元素
        
        Returns:
            分数，不存在返回 None
        """
        return await self._get_client().zscore(key, member)
    
    async def zcount(self, key: str, min_score: float, max_score: float) -> int:
        """
        获取分数范围内的元素数量
        
        Args:
            key: Redis key
            min_score: 最小分数（包含）
            max_score: 最大分数（包含）
        
        Returns:
            元素数量
        """
        return await self._get_client().zcount(key, min_score, max_score)
    
    async def zpopmin(self, key: str, count: int = 1) -> list:
        """
        弹出分数最小的元素
        
        Args:
            key: Redis key
            count: 弹出数量
        
        Returns:
            [(member, score), ...]
        """
        return await self._get_client().zpopmin(key, count)
    
    async def zpopmax(self, key: str, count: int = 1) -> list:
        """
        弹出分数最大的元素
        
        Args:
            key: Redis key
            count: 弹出数量
        
        Returns:
            [(member, score), ...]
        """
        return await self._get_client().zpopmax(key, count)
    
    # ========== 分布式锁 ==========
    
    _UNLOCK_LUA = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    else
        return 0
    end
    """

    async def lock(self, key: str, ttl: int = 30) -> bool:
        """
        获取分布式锁（原子安全版）
        
        Args:
            key: 锁的键名
            ttl: 锁的过期时间（秒）
        
        Returns:
            是否获取成功
        """
        lock_key = f"lock:{key}"
        token = uuid.uuid4().hex
        acquired = await self.set_nx(lock_key, token, ttl=ttl)
        if acquired:
            self._lock_tokens[key] = token
        return acquired
    
    async def unlock(self, key: str, token: Optional[str] = None) -> bool:
        """
        释放分布式锁（持有者校验版）
        
        Args:
            key: 锁的键名
            token: 持有者 token；不传时使用 lock() 记录的 token
        
        Returns:
            是否释放成功
        """
        lock_key = f"lock:{key}"
        check = token if token is not None else self._lock_tokens.pop(key, None)
        if check is None:
            return False
        client = self._get_client()
        result = await client.eval(self._UNLOCK_LUA, 1, lock_key, check)
        return bool(result)
    
    @asynccontextmanager
    async def lock_context(
        self, key: str, ttl: int = 30, wait: bool = False, timeout: int = 10
    ) -> "AsyncIterator[bool]":
        """
        分布式锁上下文管理器
        
        Args:
            key: 锁的键名
            ttl: 锁的过期时间（秒）
            wait: 是否等待获取锁
            timeout: 等待超时时间（秒）
        
        Yields:
            bool: 是否获取到锁
        """
        acquired = await self.lock(key, ttl)
        if not acquired and wait:
            start = time.time()
            while not acquired and (time.time() - start) < timeout:
                await asyncio.sleep(0.5)
                acquired = await self.lock(key, ttl)
        
        try:
            yield acquired
        finally:
            if acquired:
                await self.unlock(key)
    
    # ========== 高级缓存 ==========
    
    async def get_or_set(self, key: str, func: Callable, ttl: Optional[int] = None) -> Any:
        """
        获取缓存，如果不存在则通过函数设置
        
        Args:
            key: 缓存键名
            func: 获取数据的函数（可以是同步或异步）
            ttl: 缓存过期时间（秒）
        
        Returns:
            缓存的数据
        """
        cached = await self.get(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                return cached
        
        result = func()
        if asyncio.iscoroutinefunction(func):
            result = await result
        await self.set(key, result, ttl)
        return result
    
    # ========== 健康检查 ==========
    
    async def ping(self) -> bool:
        """
        健康检查
        
        Returns:
            Redis 是否可用
        """
        try:
            return await self._get_client().ping()
        except Exception:
            return False
    
    async def close(self) -> None:
        """关闭连接"""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis 连接已关闭")


# 全局客户端实例
_redis_client: Optional[RedisClient] = None


async def init_redis(config: Dict[str, Any]) -> RedisClient:
    """初始化全局 Redis 客户端"""
    global _redis_client
    _redis_client = RedisClient(config.get('redis', {}))
    await _redis_client.ping()
    logger.info("Redis 全局客户端初始化完成")
    return _redis_client


def get_redis() -> Optional[RedisClient]:
    """获取全局 Redis 客户端"""
    global _redis_client
    if _redis_client is None:
        logger.warning("Redis 客户端未初始化")
    return _redis_client

async def close_redis() -> None:
    """关闭 Redis 连接"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis 全局客户端已关闭")