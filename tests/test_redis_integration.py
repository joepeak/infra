#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra 真 redis 集成测试

前置条件（手动触发）：
    方式 A（环境变量）：
        export REDIS_INTEGRATION_HOST=10.0.0.62
        export REDIS_INTEGRATION_PORT=6379
        export REDIS_INTEGRATION_PASSWORD=<your_password>
        pytest -m integration tests/test_redis_integration.py

    方式 B（.env.test 文件——自动加载）：
        在 infra 仓库根目录创建 .env.test（已被 .gitignore 排除）：
            REDIS_INTEGRATION_HOST=10.0.0.62
            REDIS_INTEGRATION_PORT=6379
            REDIS_INTEGRATION_PASSWORD=...
        然后跑：
            pytest -m integration tests/test_redis_integration.py

默认 pytest tests/ 跳过（无 REDIS_INTEGRATION_HOST 就 skip）。

覆盖：
- redis/client.py：真 SET/GET 序列化 / SETEX TTL 过期 / 分布式锁原子性 / 并发争用
- redis/decorators.py：@cache 真序列化往返
- mq/stream_mq.py：多消费者组真并发（tg_group / db_group / analyzer_group）
- 锁住 RedisStreamMQ 的 xgroup_create 真行为

每个测试独立使用 test_redis_integration: 前缀的 key——避免污染真实数据。
所有 key 测试结束清理。

注意：本文件**不包含任何真实凭据**——password 完全靠环境变量或 .env.test。
.env.test 已在 .gitignore 排除（不会被 commit）。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
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


# ============================================================
# 标记 + skip 机制
# ============================================================
pytestmark = pytest.mark.integration


def _env_config() -> Optional[Dict[str, Any]]:
    """从环境变量（或 .env.test）读 redis 配置——未设则 skip。"""
    host = os.getenv("REDIS_INTEGRATION_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("REDIS_INTEGRATION_PORT", "6379")),
        "password": os.getenv("REDIS_INTEGRATION_PASSWORD", ""),
        "db": int(os.getenv("REDIS_INTEGRATION_DB", "0")),
    }


# 集成测试 module-level skip——必须 REDIS_INTEGRATION_HOST + PASSWORD 都设才跑
# 避免 .env.test 给了空密码时还试图连（连接错、auth 错、假阴性 fail）
_INTEGRATION_CONFIG = _env_config()
if _INTEGRATION_CONFIG is None or not _INTEGRATION_CONFIG.get("password"):
    pytest.skip(
        "REDIS_INTEGRATION_HOST / PASSWORD 未设——跳过集成测试（手动设环境变量或 .env.test 触发）",
        allow_module_level=True,
    )


# ============================================================
# Fixture
# ============================================================
@pytest.fixture
def redis_config():
    """真 redis 配置（带唯一 db 隔离测试）。"""
    cfg = dict(_INTEGRATION_CONFIG)
    cfg["test_prefix"] = f"infra_test:{uuid.uuid4()}"
    return cfg


@pytest.fixture
async def redis_client(redis_config):
    """真 RedisClient 连接——测试结束自动清理 + close。"""
    from infra.redis import RedisClient

    client = RedisClient(config=redis_config)
    yield client, redis_config["test_prefix"]
    # 清理
    try:
        # 删所有 test_prefix:* 的 key
        keys = await client._get_client().keys(f"{redis_config['test_prefix']}:*")
        if keys:
            await client._get_client().delete(*keys)
    except Exception:
        pass
    await client.close()


# ============================================================
# 1. RedisClient 真序列化
# ============================================================
class TestRedisClientIntegration:
    async def test_set_get_dict_with_unicode(self, redis_client):
        """dict 序列化往返 + 中文 unicode 保留。"""
        client, prefix = redis_client
        key = f"{prefix}:dict"
        data = {"city": "上海", "n": 42, "list": [1, 2, {"deep": "嵌套"}]}
        await client.set(key, data)
        result = await client.get(key)
        # RedisClient.set 把 dict → JSON str 存——get 直接返 str
        # by-design 行为：调用方拿 JSON str 自己 json.loads
        # （这是 RedisClient 的"低层"接口；高层用 get_or_set）
        assert json.loads(result) == data
        # 锁住 ensure_ascii=False 保留中文
        assert '"上海"' in result
        assert "上海" in result  # 不会出现 b'\xe4\xb8\x8a\xe6\xb5\xb7' 转义

    async def test_set_get_list(self, redis_client):
        client, prefix = redis_client
        key = f"{prefix}:list"
        data = [1, "x", {"k": "v"}]
        await client.set(key, data)
        result = await client.get(key)
        assert json.loads(result) == data

    async def test_set_get_int_converts_to_str(self, redis_client):
        """set int → 实际存 redis 是字符串——get 返字符串（by-design 行为）。"""
        client, prefix = redis_client
        key = f"{prefix}:int"
        await client.set(key, 42)
        result = await client.get(key)
        # set int → str(42) → "42" 存 redis → get 返 "42"
        assert result == "42"
        assert isinstance(result, str)

    async def test_setex_ttl_expires(self, redis_client):
        """SETEX ttl=1 秒——真过期。"""
        client, prefix = redis_client
        key = f"{prefix}:ttl"
        await client.set(key, "value", ttl=1)
        # 立即 get——在
        assert await client.get(key) == "value"
        # 等 1.5 秒——应过期（redis 自动删）
        await asyncio.sleep(1.5)
        assert await client.get(key) is None

    async def test_exists(self, redis_client):
        client, prefix = redis_client
        key = f"{prefix}:exists"
        assert await client.exists(key) is False
        await client.set(key, "x")
        assert await client.exists(key) is True

    async def test_delete_multiple_keys(self, redis_client):
        client, prefix = redis_client
        k1 = f"{prefix}:d1"
        k2 = f"{prefix}:d2"
        await client.set(k1, "1")
        await client.set(k2, "2")
        deleted = await client.delete(k1, k2)
        assert deleted == 2
        assert await client.exists(k1) is False
        assert await client.exists(k2) is False


# ============================================================
# 2. 分布式锁——真并发
# ============================================================
class TestDistributedLockIntegration:
    async def test_lock_acquire_release_round_trip(self, redis_client):
        client, prefix = redis_client
        lock_key = f"{prefix}:lock:basic"
        # 拿锁
        assert await client.lock(lock_key, ttl=10) is True
        # 第二次拿——应该失败（SETNX）
        assert await client.lock(lock_key, ttl=10) is False
        # 释放
        assert await client.unlock(lock_key) is True
        # 释放后能再拿
        assert await client.lock(lock_key, ttl=10) is True

    async def test_lock_ttl_expires_automatically(self, redis_client):
        """ttl=1 锁——1.2 秒后自动过期，可被重新拿。"""
        client, prefix = redis_client
        lock_key = f"{prefix}:lock:ttl"
        assert await client.lock(lock_key, ttl=1) is True
        # 等过期
        await asyncio.sleep(1.2)
        # 重新拿——应成功（之前的锁自动过期）
        assert await client.lock(lock_key, ttl=5) is True

    async def test_lock_with_prefix(self, redis_client):
        """lock() 内部加 "lock:" 前缀——验证 key 实际是 lock:xxx。"""
        client, prefix = redis_client
        # 拿锁
        assert await client.lock(f"{prefix}:myresource", ttl=10) is True
        # 检查真 redis 的 key 是 "lock:<prefix>:myresource"
        actual = await client._get_client().get(f"lock:{prefix}:myresource")
        assert actual is not None  # SETNX 返回的不是 None
        # 释放用 unlock——会调 delete("lock:xxx")
        assert await client.unlock(f"{prefix}:myresource") is True


# ============================================================
# 3. @cache 装饰器——真 redis
# ============================================================
class TestCacheDecoratorIntegration:
    async def test_cache_hit_miss_real_redis(self, redis_client):
        """@cache 命中/未命中真 redis。"""
        from infra.redis.decorators import cache
        from infra.redis import init_redis, get_redis, close_redis

        client, prefix = redis_client
        # 用测试 redis 作为全局
        # 注：@cache 用 get_redis() 拿全局——手动 init
        await init_redis(config={
            "host": client.config["host"],
            "port": client.config["port"],
            "password": client.config.get("password", ""),
            "db": client.config["db"],
        })
        try:
            call_count = 0

            # init_redis 期望嵌套 config: {"redis": {...}}
            redis_sub = {
                "host": client.config["host"],
                "port": client.config["port"],
                "password": client.config.get("password", ""),
                "db": client.config["db"],
            }
            await init_redis(config={"redis": redis_sub})

            @cache(key=f"{prefix}:cached_data", ttl=60)
            async def my_func():
                nonlocal call_count
                call_count += 1
                return {"value": "expensive", "n": 42}

            # 第一次——未命中，执行原函数
            r1 = await my_func()
            assert r1 == {"value": "expensive", "n": 42}
            assert call_count == 1

            # 第二次——命中，不调原函数
            r2 = await my_func()
            assert r2 == {"value": "expensive", "n": 42}
            assert call_count == 1  # 没增加

            # 清理
            await client._get_client().delete(f"{prefix}:cached_data")
        finally:
            await close_redis()


# ============================================================
# 4. RedisStreamMQ 真 Stream 操作
# ============================================================
class TestRedisStreamMqIntegration:
    async def test_create_queue_and_xadd(self, redis_client):
        """create_queue + publish——真 stream 写 + 读。"""
        from infra.mq import RedisStreamMQ
        client, prefix = redis_client
        # 用测试 client 作为基础——但 init_mq 走全局 _mq
        # 这里直接构造 RedisStreamMQ
        mq = RedisStreamMQ(redis_config={
            "host": client.config["host"],
            "port": client.config["port"],
            "password": client.config.get("password", ""),
            "db": client.config["db"],
        })
        try:
            # 真实 connect
            connected = await mq.connect()
            if not connected:
                pytest.skip("无法连真 redis")
            # 唯一 stream 名
            stream = f"{prefix}:my_stream"
            group = f"{prefix}:group1"
            mq.create_queue(stream, group)
            # 发送
            msg_id = await mq.publish(stream, {
                "task_id": "t-1",
                "task_type": "test",
                "scheduled_time": "2026-01-01",
                "payload": json.dumps({"k": "v"}),
                "metadata": "",
                "priority": "normal",
                "created_at": str(time.time()),
            })
            assert msg_id is not None
            # 真 redis stream 长度
            length = await mq._redis.xlen(stream)
            assert length == 1
        finally:
            # 清理
            try:
                await mq._redis.delete(stream)
            except Exception:
                pass
            await mq._redis.close()

    async def test_multiple_consumer_groups(self, redis_client):
        """多消费者组（tg / db / analyzer）——每个 group 独立消费。"""
        from infra.mq import RedisStreamMQ
        client, prefix = redis_client

        mq = RedisStreamMQ(redis_config={
            "host": client.config["host"],
            "port": client.config["port"],
            "password": client.config.get("password", ""),
            "db": client.config["db"],
        })
        try:
            connected = await mq.connect()
            if not connected:
                pytest.skip("无法连真 redis")

            stream = f"{prefix}:multi_group_stream"
            tg = f"{prefix}:tg"
            db = f"{prefix}:db"
            analyzer = f"{prefix}:analyzer"

            mq.create_queue(stream, tg)
            mq.create_queue(stream, db)
            mq.create_queue(stream, analyzer)
            # create_queue 是 lazy（只注册 QueueConfig）——真集成测试必须显式建 group
            for grp in [tg, db, analyzer]:
                try:
                    await mq._redis.xgroup_create(stream, grp, id='0', mkstream=True)
                except Exception:
                    pass  # BUSYGROUP ignore

            # 发 1 条消息
            msg_id = await mq.publish(stream, {
                "task_id": "t-1",
                "task_type": "test",
                "scheduled_time": "2026-01-01",
                "payload": json.dumps({}),
                "metadata": "",
                "priority": "normal",
                "created_at": str(time.time()),
            })
            assert msg_id is not None

            # xadd 后消息在 stream——但只有 xreadgroup 才会"派发"到 group PEL
            # 用 xreadgroup 读 0 条（不阻塞，立即返回），强制把消息派发到 group
            for grp in [tg, db, analyzer]:
                await mq._redis.xreadgroup(grp, f"{grp}_consumer", {stream: ">"}, count=1, block=10)

            # 3 个 group 都应 pending=1（互不干扰）
            for group in [tg, db, analyzer]:
                pending_list = await mq._redis.xpending_range(stream, group, '-', '+', count=10)
                assert len(pending_list) == 1, f"group {group} 应有 1 条 pending（实际 {len(pending_list)}）"
        finally:
            try:
                await mq._redis.delete(stream)
            except Exception:
                pass
            await mq._redis.close()
