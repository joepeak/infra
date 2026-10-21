#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.mq（Redis Stream 消息队列）的**真 Redis 集成测试**

背景（20261021）：
    `tests/test_mq_stream_mq.py` 全部用 mock，覆盖不到真正要命的地方：
      1. **XPENDINGRANGE 的字段名**——redis-py 返回 ``time_since_delivered``，
         旧代码读 ``idle`` 恒拿到 0，导致「超时判断」永远不成立，
         重试与死信两个分支从不执行（mock 测不出来，因为 mock 返回什么
         由测试自己写）。
      2. **XAUTOCLAIM 的返回值顺序**——``[next_cursor, messages, deleted]``，
         cursor 在**第一个**；解包顺序错了会导致「认领到消息但拿到的是游标」。
      3. **认领后必须真正处理**——认领只是转移所有权，worker 用 ``>``
         只读新消息，认领后没人处理就会永远空转。
      4. 真实服务端对 ``XADD`` / ``XACK`` / ``XDEL`` 的行为。

    本文件用**真 Redis** 跑一遍完整链路，锁住这些 mock 测不到的行为。

前置条件（手动触发）：
    export REDIS_INTEGRATION_HOST=127.0.0.1
    export REDIS_INTEGRATION_PORT=6379
    pytest -m integration tests/test_mq_integration.py

    也支持 .env.test 自动加载（同 test_redis_lists_integration.py）。

默认 ``pytest tests/`` 会跳过（未设 REDIS_INTEGRATION_HOST）。

所有 stream 用 ``infra_test_mq:<uuid>:`` 前缀隔离，测试结束清理。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "db": int(os.getenv("REDIS_INTEGRATION_DB", "0")),
    }


_INTEGRATION_CONFIG = _env_config()
if _INTEGRATION_CONFIG is None:
    pytest.skip(
        "REDIS_INTEGRATION_HOST 未设——跳过 MQ 集成测试",
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
        f"（{_UNREACHABLE}）——跳过 MQ 集成测试",
        allow_module_level=True,
    )


@pytest.fixture
async def real_redis():
    """真 Redis 客户端（用于直接检查 stream / PEL / 死信状态）。"""
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
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def mq():
    """RedisStreamMQ 实例——测试结束清理用到的所有 key。"""
    from infra.mq.stream_mq import RedisStreamMQ

    prefix = f"infra_test_mq:{uuid.uuid4().hex[:8]}"
    instance = RedisStreamMQ(dict(_INTEGRATION_CONFIG))
    try:
        assert await instance.connect(), "连接 Redis 失败"
        instance._test_prefix = prefix  # type: ignore[attr-defined]
        yield instance
    finally:
        try:
            await instance.stop()
        except Exception:
            pass
        # 清理所有用到的 key
        try:
            import redis.asyncio as aioredis
            cleanup = aioredis.Redis(
                host=_INTEGRATION_CONFIG["host"],
                port=_INTEGRATION_CONFIG["port"],
                password=_INTEGRATION_CONFIG.get("password"),
                db=_INTEGRATION_CONFIG.get("db", 0),
                decode_responses=True,
            )
            keys = []
            async for key in cleanup.scan_iter(match=f"{prefix}*"):
                keys.append(key)
            if keys:
                await cleanup.delete(*keys)
            await cleanup.aclose()
        except Exception:
            pass


def _len(seq) -> int:
    return len(seq)


def _stream_name(mq_instance, suffix: str = "q") -> str:
    """生成带测试前缀的 stream 名。"""
    return f"{mq_instance._test_prefix}:{suffix}"


async def _wait_for(
    predicate,
    timeout: float = 8.0,
    interval: float = 0.2,
) -> bool:
    """轮询等待条件成立——返回是否在超时前成立。

    ``predicate`` 可以是同步 callable 或返回 awaitable 的 callable——
    两种都支持（内部统一 await 结果）。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            result = await result
        if result:
            return True
        await asyncio.sleep(interval)
    return False


# ============================================================
# 1. 基础发布 / 消费
# ============================================================
class TestBasicFlow:
    """消息能发布、能被消费、能被 ACK。"""

    async def test_publish_and_consume(self, mq, real_redis):
        stream = _stream_name(mq, "basic")
        mq.create_queue(stream, concurrency=1, block_ms=100)
        received: List[Any] = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        msg_id = await mq.publish(
            stream, {"task_id": "t1", "task_type": "demo", "payload": {"x": 1}}
        )
        assert msg_id is not None

        ok = await _wait_for(lambda: _len(received) == 1)
        assert ok, "消息未被消费"
        assert received[0].task_id == "t1"
        assert received[0].payload == {"x": 1}

        # 消费成功后应被 ACK——PEL 为空
        pending = await real_redis.xpending_range(
            stream, f"{stream}_group", "-", "+", count=10
        )
        assert len(pending) == 0, f"消息未被 ACK，PEL: {pending}"

    async def test_publish_task_message_object(self, mq, real_redis):
        """TaskMessage 对象也能发布并被正确反序列化。"""
        from infra.messages import TaskMessage

        stream = _stream_name(mq, "taskmsg")
        mq.create_queue(stream, concurrency=1, block_ms=100)
        received: List[Any] = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        task = TaskMessage(
            task_id="tid-1",
            task_type="demo",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v"},
            metadata={"m": 1},
        )
        assert await mq.publish(stream, task) is not None
        assert await _wait_for(lambda: len(received) == 1)
        assert received[0].task_id == "tid-1"
        assert received[0].payload == {"k": "v"}
        assert received[0].metadata == {"m": 1}


# ============================================================
# 2. 失败重试 —— 核心回归
# ============================================================
class TestRetryAndDeadLetter:
    """handler 失败后应重试，超限后进死信队列。

    这是本轮修复的核心：**旧实现里重试与死信两个分支从不执行**
    （读错了 XPENDINGRANGE 的字段名），消息永久卡在 PEL。
    """

    async def test_failed_message_is_retried(self, mq, real_redis):
        stream = _stream_name(mq, "retry")
        mq.create_queue(
            stream,
            concurrency=1,
            block_ms=100,
            claim_interval=1,
            claim_min_idle_ms=300,
            max_retries=3,
        )
        calls: List[int] = []

        async def handler(msg):
            calls.append(1)
            raise RuntimeError("故意失败")

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(stream, {"task_id": "t", "task_type": "demo", "payload": {}})

        # 应至少被调用 2 次（首次 + 至少 1 次重试）
        ok = await _wait_for(lambda: len(calls) >= 2, timeout=8.0)
        assert ok, f"失败消息未被重试，handler 只调用了 {len(calls)} 次"

    async def test_failed_message_moves_to_dead_letter(self, mq, real_redis):
        """重试次数超限后，消息应进死信队列且不再占用 PEL。"""
        stream = _stream_name(mq, "dead")
        mq.create_queue(
            stream,
            concurrency=1,
            block_ms=100,
            claim_interval=1,
            claim_min_idle_ms=300,
            max_retries=2,
        )
        calls: List[int] = []

        async def handler(msg):
            calls.append(1)
            raise RuntimeError("故意失败")

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(stream, {"task_id": "t", "task_type": "demo", "payload": {}})

        # 等到进死信
        ok = await _wait_for(
            lambda: _dead_has_msg(real_redis, stream), timeout=12.0
        )
        assert ok, "重试超限后消息未进死信队列"

        # 死信记录应带原因
        entries = await real_redis.xrange(f"{stream}:dead_letter", "-", "+")
        assert entries, "死信队列为空"
        record = entries[0][1]
        assert record.get("reason") == "max_retries_exceeded"
        assert int(record.get("delivery_count", 0)) >= 2

        # 不应继续占用 PEL
        pending = await real_redis.xpending_range(
            stream, f"{stream}_group", "-", "+", count=10
        )
        assert len(pending) == 0, f"消息仍卡在 PEL: {pending}"

    async def test_dead_letter_can_be_retried(self, mq, real_redis):
        """死信消息可以通过 retry_dead_letter 恢复到原 stream。"""
        stream = _stream_name(mq, "redl")
        mq.create_queue(
            stream,
            concurrency=1,
            block_ms=100,
            claim_interval=1,
            claim_min_idle_ms=300,
            max_retries=1,
        )
        should_fail = {"v": True}

        async def handler(msg):
            if should_fail["v"]:
                raise RuntimeError("先失败")

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(stream, {"task_id": "t", "task_type": "demo", "payload": {}})
        assert await _wait_for(
            lambda: _dead_has_msg(real_redis, stream), timeout=12.0
        ), "消息未进死信"

        # 从死信恢复
        entries = await real_redis.xrange(f"{stream}:dead_letter", "-", "+")
        dead_id = entries[0][0]
        assert await mq.retry_dead_letter(stream, dead_id) is True

        # 死信里应已被移除
        assert not await _dead_has_msg(real_redis, stream)


async def _dead_has_msg(real_redis, stream: str) -> bool:
    """死信队列里是否有消息——返回 bool 供 _wait_for 判定。"""
    return int(await real_redis.xlen(f"{stream}:dead_letter")) > 0


# ============================================================
# 3. 无处理器 —— 不应静默丢弃
# ============================================================
class TestNoHandler:
    """没有对应 handler 的消息应进死信，而不是静默消失。"""

    async def test_message_without_handler_goes_to_dead_letter(self, mq, real_redis):
        stream = _stream_name(mq, "nohandler")
        mq.create_queue(stream, concurrency=1, block_ms=100)
        # 故意不注册 handler
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "t", "task_type": "没人处理", "payload": {}}
        )

        ok = await _wait_for(
            lambda: _dead_has_msg(real_redis, stream), timeout=8.0
        )
        assert ok, "无 handler 的消息被静默丢弃，未进死信队列"

        entries = await real_redis.xrange(f"{stream}:dead_letter", "-", "+")
        assert entries[0][1].get("reason") == "no_handler"


# ============================================================
# 4. publish 的序列化错误
# ============================================================
class TestPublishSerialization:
    """payload 不可序列化时应抛异常，而不是静默丢消息。"""

    async def test_unserializable_payload_raises(self, mq):
        stream = _stream_name(mq, "ser")
        mq.create_queue(stream)

        class NotJson:
            pass

        # 断言错误信息包含明确提示——而不只是类型。
        # 裸的 json.dumps 也会抛 TypeError，但信息是
        # "Object of type NotJson is not JSON serializable"，
        # 无法区分是「MQ 序列化失败」还是别处的 JSON 错误。
        with pytest.raises(TypeError) as exc_info:
            await mq.publish(
                stream,
                {"task_id": "t", "task_type": "x", "payload": {"obj": NotJson()}},
            )
        assert "无法 JSON 序列化" in str(exc_info.value)

    async def test_publish_to_unknown_queue_raises(self, mq):
        with pytest.raises(ValueError):
            await mq.publish("不存在的队列", {"task_type": "x"})


# ============================================================
# 5. 死信数据损坏
# ============================================================
class TestCorruptDeadLetter:
    """死信数据损坏时不应抛异常打断调用方。"""

    async def test_corrupt_dead_letter_does_not_raise(self, mq, real_redis):
        stream = _stream_name(mq, "corrupt")
        mq.create_queue(stream)

        bad_id = await real_redis.xadd(
            f"{stream}:dead_letter", {"original_data": "这不是合法 JSON{{{"}
        )
        result = await mq.retry_dead_letter(stream, bad_id)
        assert result is False


# ============================================================
# 6. 时效性（TTL）
# ============================================================
class TestMessageTTL:
    """过期消息应被丢弃并 ACK，不占用 PEL。"""

    async def test_expired_message_is_dropped(self, mq, real_redis):
        stream = _stream_name(mq, "ttl")
        mq.create_queue(stream, concurrency=1, block_ms=100, ttl_seconds=1)
        received: List[Any] = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        # 直接往 stream 里塞一条 created_at 为过去时间的消息
        import time as _time
        await real_redis.xadd(
            stream,
            {
                "task_id": "old",
                "task_type": "demo",
                "scheduled_time": "2026-01-01T00:00:00",
                "payload": "{}",
                "metadata": "",
                "priority": "normal",
                "created_at": str(_time.time() - 3600),
            },
        )

        ok = await _wait_for(
            lambda: _expired_count(mq) >= 1, timeout=8.0
        )
        assert ok, "过期消息未被识别"
        assert len(received) == 0, "过期消息不应被交给 handler"

        pending = await real_redis.xpending_range(
            stream, f"{stream}_group", "-", "+", count=10
        )
        assert len(pending) == 0, "过期消息仍占用 PEL"


def _expired_count(mq_instance) -> int:
    return int(mq_instance._stats.get("total_expired", 0))


# ============================================================
# 7. 停止时的清理
# ============================================================
class TestStop:
    """stop() 应真正结束所有任务，并释放连接。"""

    async def test_stop_clears_tasks(self, mq):
        stream = _stream_name(mq, "stop")
        mq.create_queue(stream, concurrency=2, block_ms=100)

        async def handler(msg):
            pass

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        assert mq._consumer_tasks.get(stream), "消费者任务未启动"
        assert mq._claim_tasks.get(stream), "认领任务未启动"

        await mq.stop()

        assert not mq._consumer_tasks, "消费者任务未清理"
        assert not mq._claim_tasks, "认领任务未清理"
        assert mq._redis is None, "连接未释放（应置空以便下次重连）"
