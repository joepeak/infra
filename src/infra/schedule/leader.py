# infra/schedule/leader.py
"""
Leader Election 模块
使用 Redis 分布式锁确保只有一个 worker 启动 scheduler
"""

import asyncio
import uuid
from typing import AsyncIterator, Optional
from contextlib import asynccontextmanager

import redis.asyncio as redis

from infra.logger import get_logger

logger = get_logger(__name__)


class LeaderElection:
    """基于 Redis 的 Leader Election"""

    def __init__(self, redis_config: dict) -> None:
        self.redis_config = redis_config
        self._redis: Optional[redis.Redis] = None
        self._lock_key = "scheduler:leader:lock"
        self._lock_value = str(uuid.uuid4())
        self._lock_ttl = 30  # 锁超时（秒）
        self._heartbeat_interval = 10  # 心跳间隔（秒）
        self._is_leader = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            host = self.redis_config.get('host', 'localhost')
            port = self.redis_config.get('port', 6379)
            password = self.redis_config.get('password', '')
            db = self.redis_config.get('db', 1)  # 使用独立 db，避免与 MQ 冲突
            url = f"redis://:{password}@{host}:{port}/{db}" if password else f"redis://{host}:{port}/{db}"
            self._redis = await redis.from_url(url, decode_responses=True)
        return self._redis

    async def try_acquire(self) -> bool:
        """尝试获取 leader 锁"""
        redis_client = await self._get_redis()
        # redis.set 在 redis-py 5+ 返 bool | str | bytes | None
        # 我们只关心 truthy——是否成功拿到锁
        acquired = bool(await redis_client.set(
            self._lock_key,
            self._lock_value,
            nx=True,
            ex=self._lock_ttl
        ))
        if acquired:
            self._is_leader = True
            logger.info("✅ 成为 Leader，将启动 Scheduler")
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        else:
            logger.debug("未能成为 Leader，等待其他实例执行")
        return acquired

    async def _heartbeat_loop(self) -> None:
        """心跳循环：定期续期锁"""
        while self._is_leader:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                redis_client = await self._get_redis()
                # Lua 脚本：只有持有锁的实例才能续期
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("expire", KEYS[1], ARGV[2])
                else
                    return 0
                end
                """
                result = await redis_client.eval(lua_script, 1, self._lock_key, self._lock_value, self._lock_ttl)
                if result:
                    logger.debug(f"Leader 锁续期成功，TTL={self._lock_ttl}s")
                else:
                    logger.warning("Leader 锁续期失败，可能已被其他实例抢占")
                    self._is_leader = False
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳续期异常: {e}")

    async def release(self) -> None:
        """释放 leader 锁"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._is_leader:
            redis_client = await self._get_redis()
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            await redis_client.eval(lua_script, 1, self._lock_key, self._lock_value)
            logger.info("Leader 锁已释放")
            self._is_leader = False
        if self._redis:
            await self._redis.close()

    def is_leader(self) -> bool:
        return self._is_leader

    @asynccontextmanager
    async def leader_context(self) -> "AsyncIterator[bool]":
        acquired = await self.try_acquire()
        try:
            yield acquired
        finally:
            if acquired:
                await self.release()