# infra/schedule/leader.py
"""
Leader Election 模块
使用 Redis 分布式锁确保只有一个 worker 启动 scheduler
"""

import asyncio
import inspect
import uuid
from typing import Any, AsyncIterator, Callable, Optional
from contextlib import asynccontextmanager

import redis.asyncio as redis

from infra.logger import get_logger

logger = get_logger(__name__)


class LeaderElection:
    """基于 Redis 的 Leader Election

    **失去 leader 时的通知**：通过 ``on_lost_leader`` 回调通知调用方——
    调用方**必须**在此回调里停掉自己的 leader-only 工作（如 APScheduler），
    否则会出现「脑裂」：锁已被别人抢走，本实例却还在跑定时任务，
    导致同一个任务被多个实例同时执行。
    """

    def __init__(
        self,
        redis_config: dict,
        *,
        on_lost_leader: Optional[Callable[[], Any]] = None,
        heartbeat_interval: int = 10,
        lock_ttl: int = 30,
        max_heartbeat_failures: int = 3,
    ) -> None:
        self.redis_config = redis_config
        self._redis: Optional[redis.Redis] = None
        self._lock_key = "scheduler:leader:lock"
        self._lock_value = str(uuid.uuid4())
        self._lock_ttl = lock_ttl  # 锁超时（秒）
        self._heartbeat_interval = heartbeat_interval  # 心跳间隔（秒）
        #: 连续心跳失败多少次才判定「失去 leader」。
        #: 单次网络抖动不应立刻放弃——锁的 TTL 还有余量，
        #: 过早放弃会导致 leader 频繁切换（抖动放大）。
        self._max_heartbeat_failures = max_heartbeat_failures
        #: 失去 leader 时调用（可以是 async）——调用方应在此停掉 leader-only 工作
        self._on_lost_leader = on_lost_leader
        self._is_leader = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0

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
        """心跳循环：定期续期锁

        两种失败要区分：
          - **续期返回 0**：锁已不属于自己（被别的实例抢走 / TTL 过期）
            → 立即判定失去 leader
          - **续期抛异常**：网络抖动等临时故障
            → 累计失败次数，连续失败超过阈值才判定失去 leader
              （单次抖动就放弃会造成 leader 频繁切换，抖动被放大）
        """
        renew_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """
        while self._is_leader:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                redis_client = await self._get_redis()
                result = await redis_client.eval(
                    renew_script, 1, self._lock_key, self._lock_value, self._lock_ttl
                )
                if result:
                    self._consecutive_failures = 0
                    logger.debug(f"Leader 锁续期成功，TTL={self._lock_ttl}s")
                else:
                    # 锁已不属于自己——确定失去 leader
                    logger.warning("Leader 锁续期失败，锁已被其他实例抢占")
                    await self._mark_lost_leader("lock_lost")
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    f"心跳续期异常（连续第 {self._consecutive_failures} 次）: {e}"
                )
                if self._consecutive_failures >= self._max_heartbeat_failures:
                    # 连续失败到阈值——认为已经失去 leader
                    logger.error(
                        f"心跳连续失败 {self._consecutive_failures} 次，判定失去 leader"
                    )
                    await self._mark_lost_leader("heartbeat_failed")
                    break

    async def _mark_lost_leader(self, reason: str) -> None:
        """标记为失去 leader，并通知调用方。

        调用 ``on_lost_leader`` 回调（支持 async 回调）——
        调用方应在此停掉 leader-only 工作（APScheduler 等），否则会脑裂。
        """
        if not self._is_leader:
            return
        self._is_leader = False
        logger.warning(f"失去 Leader（原因: {reason}），通知调用方停止 leader-only 工作")
        callback = self._on_lost_leader
        if callback is None:
            return
        try:
            result = callback()
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            # 回调失败不该让心跳协程崩掉——但必须记日志
            logger.error(f"on_lost_leader 回调执行失败: {type(e).__name__}: {e}")

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
            # aclose() 是 redis-py 5+ 的推荐 API（close() 仍可用但已非首选）
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning(f"关闭 Redis 连接失败: {e}")
            finally:
                self._redis = None

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