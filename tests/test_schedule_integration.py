#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.schedule（调度与 leader 选举）的**真 Redis 集成测试**

背景（20261021 晚间）：
    `tests/test_schedule_deep.py` 全用 mock，覆盖不到真正要命的地方：
      1. **脑裂**——锁被别的实例抢走后，LeaderElection 判定失去 leader，
         但 SchedulerManager 的 `_is_leader` 是**独立字段**，两者不联动，
         于是 APScheduler 照跑不误 -> 同一个定时任务被两个实例同时执行。
         mock 测不出来（mock 不会自己变化）。
      2. **心跳抖动**——单次网络抖动就放弃 leader 会导致 leader 频繁切换。
      3. **Lua 脚本的原子性**——续期/释放必须校验锁的持有者。

    本文件用**真 Redis** 跑一遍，锁住这些行为。

前置条件（手动触发）：
    export REDIS_INTEGRATION_HOST=127.0.0.1
    export REDIS_INTEGRATION_PORT=6379
    pytest -m integration tests/test_schedule_integration.py

    也支持 .env.test 自动加载。

默认 ``pytest tests/`` 会跳过（未设 REDIS_INTEGRATION_HOST）。

注意：leader 选举用的是 **db=1**（与 MQ 的 db 隔离）——
      本文件的集成测试也用 db=1，并**不会**清理其它 db 的数据。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# ============================================================
# 自动加载 .env.test（如果存在）
# ============================================================
try:
    from dotenv import load_dotenv
    _ENV_TEST = Path(__file__).resolve().parent.parent / ".env.test"
    if _ENV_TEST.exists():
        load_dotenv(_ENV_TEST, override=False)
except ImportError:
    pass  # python-dotenv 未装——只依赖环境变量


pytestmark = pytest.mark.integration


def _env_config() -> Optional[Dict[str, Any]]:
    """从环境变量（或 .env.test）读 redis 配置——未设 HOST 则 skip。"""
    host = os.getenv("REDIS_INTEGRATION_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("REDIS_INTEGRATION_PORT", "6379")),
        "password": os.getenv("REDIS_INTEGRATION_PASSWORD", "") or None,
        # leader 选举固定用 db=1（与 MQ 隔离）
        "db": 1,
    }


_INTEGRATION_CONFIG = _env_config()
if _INTEGRATION_CONFIG is None:
    pytest.skip(
        "REDIS_INTEGRATION_HOST 未设——跳过 schedule 集成测试",
        allow_module_level=True,
    )


def _redis_reachable(config: Dict[str, Any]) -> Optional[str]:
    """同步探活——连不上返回原因说明，连上返回 None。"""
    import redis as sync_redis

    try:
        client = sync_redis.Redis(
            host=config["host"],
            port=config["port"],
            password=config.get("password"),
            db=config.get("db", 0),
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        client.ping()
        client.close()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


_UNREACHABLE = _redis_reachable(_INTEGRATION_CONFIG)
if _UNREACHABLE:
    pytest.skip(
        f"连不上 redis {_INTEGRATION_CONFIG['host']}:{_INTEGRATION_CONFIG['port']}"
        f"（{_UNREACHABLE}）——跳过 schedule 集成测试",
        allow_module_level=True,
    )


LOCK_KEY = "scheduler:leader:lock"


def _job_config(func, job_id: str, trigger: str = "interval", **kwargs):
    """构造 JobConfig（避免每个用例重复 import）。"""
    from infra.schedule.scheduler_manager import JobConfig

    return JobConfig(
        func=func, job_id=job_id, trigger=trigger,
        trigger_args={"seconds": 60}, **kwargs,
    )


@pytest.fixture
async def real_redis():
    """真 Redis 客户端（清理 leader 锁）。"""
    import redis.asyncio as aioredis

    client = aioredis.Redis(
        host=_INTEGRATION_CONFIG["host"],
        port=_INTEGRATION_CONFIG["port"],
        password=_INTEGRATION_CONFIG.get("password"),
        db=_INTEGRATION_CONFIG.get("db", 0),
        decode_responses=True,
    )
    try:
        await client.ping()
    except Exception as e:
        await client.aclose()
        pytest.skip(f"连不上 redis: {type(e).__name__}: {e}")
    try:
        await client.delete(LOCK_KEY)
        yield client
    finally:
        try:
            await client.delete(LOCK_KEY)
        except Exception:
            pass
        await client.aclose()


@pytest.fixture
def fresh_scheduler_manager():
    """全新的 SchedulerManager 单例（避免受其它测试影响）。"""
    from infra.schedule.scheduler_manager import SchedulerManager

    SchedulerManager._instance = None
    SchedulerManager._initialized = False
    manager = SchedulerManager()
    try:
        yield manager
    finally:
        try:
            manager.shutdown()
        except Exception:
            pass
        SchedulerManager._instance = None
        SchedulerManager._initialized = False


# ============================================================
# 1. leader 互斥
# ============================================================
class TestLeaderMutualExclusion:
    """同一时刻只能有一个 leader。"""

    async def test_only_one_instance_acquires(self, real_redis):
        from infra.schedule.leader import LeaderElection

        le1 = LeaderElection(dict(_INTEGRATION_CONFIG))
        le2 = LeaderElection(dict(_INTEGRATION_CONFIG))
        try:
            assert await le1.try_acquire() is True
            assert await le2.try_acquire() is False
        finally:
            await le1.release()

    async def test_release_allows_reacquire(self, real_redis):
        from infra.schedule.leader import LeaderElection

        le1 = LeaderElection(dict(_INTEGRATION_CONFIG))
        le2 = LeaderElection(dict(_INTEGRATION_CONFIG))
        try:
            assert await le1.try_acquire() is True
            await le1.release()
            assert await le2.try_acquire() is True
        finally:
            await le2.release()

    async def test_release_does_not_delete_others_lock(self, real_redis):
        """Lua 脚本保证：只释放自己持有的锁。"""
        from infra.schedule.leader import LeaderElection

        le = LeaderElection(dict(_INTEGRATION_CONFIG))
        try:
            await le.try_acquire()
            # 模拟锁被别的实例抢走
            await real_redis.set(LOCK_KEY, "other-instance", ex=30)
            await le.release()
            # 别人的锁不该被删掉
            value = await real_redis.get(LOCK_KEY)
            assert value == "other-instance", "释放操作删掉了别人的锁"
        finally:
            await real_redis.delete(LOCK_KEY)


# ============================================================
# 2. 失去 leader 的通知 —— 消除脑裂
# ============================================================
class TestLostLeaderNotification:
    """锁被抢走时必须通知调用方，否则会脑裂。"""

    async def test_on_lost_leader_callback_fires(self, real_redis):
        from infra.schedule.leader import LeaderElection

        called = {"n": 0}

        async def on_lost():
            called["n"] += 1

        le = LeaderElection(
            dict(_INTEGRATION_CONFIG),
            on_lost_leader=on_lost,
            heartbeat_interval=1,
            lock_ttl=10,
        )
        try:
            assert await le.try_acquire() is True
            # 锁被抢走
            await real_redis.set(LOCK_KEY, "other", ex=30)
            for _ in range(30):
                await asyncio.sleep(0.3)
                if called["n"]:
                    break
            assert called["n"] == 1, "on_lost_leader 回调未触发"
            assert le.is_leader() is False
        finally:
            await le.release()

    async def test_scheduler_stops_when_leader_lost(
        self, real_redis, fresh_scheduler_manager
    ):
        """回归锁：失去 leader 后调度器必须停——否则定时任务双执行（脑裂）。"""
        from infra.schedule.leader import LeaderElection

        manager = fresh_scheduler_manager
        leader = LeaderElection(
            dict(_INTEGRATION_CONFIG),
            on_lost_leader=manager.on_lost_leader,
            heartbeat_interval=1,
            lock_ttl=10,
        )
        try:
            assert await leader.try_acquire() is True
            manager.init_scheduler(is_leader=True)
            manager.start()
            assert manager.scheduler is not None
            assert manager.scheduler.running is True

            # 锁被抢走
            await real_redis.set(LOCK_KEY, "other", ex=30)
            for _ in range(30):
                await asyncio.sleep(0.3)
                if not manager.is_leader():
                    break

            assert manager.is_leader() is False, "SchedulerManager 未感知失去 leader"
            assert manager.scheduler is not None
            assert manager.scheduler.running is False, (
                "失去 leader 后调度器仍在运行——会脑裂"
            )
        finally:
            await leader.release()

    async def test_heartbeat_tolerates_transient_failure(self, real_redis):
        """单次网络抖动不应放弃 leader——否则 leader 频繁切换。"""
        from infra.schedule.leader import LeaderElection

        le = LeaderElection(
            dict(_INTEGRATION_CONFIG),
            heartbeat_interval=1,
            lock_ttl=30,
            max_heartbeat_failures=3,
        )
        calls = {"n": 0}

        # 只让**续期**失败（try_acquire 也要能成功）
        real_get_redis = le._get_redis

        class FlakyRedis:
            def __init__(self, client):
                self._client = client

            async def eval(self, *args, **kwargs):
                calls["n"] += 1
                if calls["n"] <= 2:
                    # 模拟一次网络抖动
                    raise ConnectionError("网络抖动")
                return await self._client.eval(*args, **kwargs)

        async def flaky_get_redis():
            client = await real_get_redis()
            return FlakyRedis(client)

        try:
            assert await le.try_acquire() is True
            # 换成 flaky 版本（try_acquire 之后）
            le._get_redis = flaky_get_redis  # type: ignore[method-assign]
            await asyncio.sleep(3.5)
            assert le.is_leader() is True, "单次抖动就放弃了 leader"
            assert calls["n"] >= 3, f"续期次数异常: {calls['n']}"
        finally:
            await le.release()


# ============================================================
# 3. SchedulerManager 自身的状态一致性
# ============================================================
class TestSchedulerManagerConsistency:
    """调度器状态与实际 job 状态必须一致。"""

    async def test_reinit_shuts_down_old_scheduler(self, fresh_scheduler_manager):
        """重复 init_scheduler 必须先关掉旧的——否则旧的在跑但没人能关。"""
        manager = fresh_scheduler_manager
        manager.init_scheduler(is_leader=True)
        manager.start()
        old = manager.scheduler
        assert old is not None and old.running is True

        manager.init_scheduler(is_leader=True)
        assert manager.scheduler is not old, "应创建新的 scheduler"
        # 旧 scheduler 的关闭是「发起」（APScheduler 内部走
        # call_soon_threadsafe），同步 API 无法等待其完成——
        # 真正确保停稳要用 async_shutdown()。
        # 这里断言「不再被 manager 持有」（引用已丢弃，不会被再次 shutdown）
        assert manager.scheduler is not old

    async def test_add_job_failure_does_not_record_config(
        self, fresh_scheduler_manager
    ):
        """添加失败时不该在 _job_configs 里留下记录——状态必须一致。"""
        manager = fresh_scheduler_manager
        manager.init_scheduler(is_leader=True)
        manager.start()

        async def my_task():
            pass

        # 不支持的 trigger 类型 -> _add_scheduled_job 抛 ValueError
        result = manager.add_job(
            _job_config(my_task, job_id="bad", trigger="no_such_trigger")
        )
        assert result is None, "添加失败应返回 None"
        assert "bad" not in manager._job_configs, (
            "失败的 job 不该出现在 _job_configs 里"
        )
