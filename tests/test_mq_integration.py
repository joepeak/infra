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


# ============================================================
# 8. 消费者清理 —— 避免僵尸 consumer 累积
# ============================================================
class TestConsumerCleanup:
    """stop() 应删除本次注册过的 consumer。

    否则每次 start() 都会在消费者组里留下僵尸 consumer——
    长期运行 / 频繁重启会无限累积（实测每轮 +3）。
    """

    async def test_stop_removes_consumers(self, mq, real_redis):
        stream = _stream_name(mq, "cleanup")
        mq.create_queue(stream, concurrency=2, block_ms=100)

        async def handler(msg):
            pass

        mq.register(stream, "demo", handler)
        await mq.start([stream])
        await asyncio.sleep(0.3)

        groups = await real_redis.xinfo_groups(stream)
        assert groups[0]["consumers"] > 0, "消费者未注册"

        await mq.stop()

        groups_after = await real_redis.xinfo_groups(stream)
        assert groups_after[0]["consumers"] == 0, (
            f"stop() 后仍残留 {groups_after[0]['consumers']} 个僵尸 consumer"
        )

    async def test_repeated_start_stop_does_not_accumulate(self, mq, real_redis):
        """反复 start/stop 不应累积 consumer。"""
        stream = _stream_name(mq, "accum")
        mq.create_queue(stream, concurrency=2, block_ms=100)

        async def handler(msg):
            pass

        mq.register(stream, "demo", handler)

        counts = []
        for _ in range(3):
            await mq.start([stream])
            await asyncio.sleep(0.2)
            await mq.stop()
            groups = await real_redis.xinfo_groups(stream)
            counts.append(groups[0]["consumers"])

        assert all(c == 0 for c in counts), f"consumer 数在累积: {counts}"


# ============================================================
# 9. 并发限流
# ============================================================
class TestMaxConcurrency:
    """max_concurrency 应真正限制「同时执行的 handler 数量」。

    旧实现里 concurrency 只决定 worker 数，而每个 worker 一次会读
    batch_size 条并行处理——实际并发度 = concurrency × batch_size
    （实测 concurrency=2 + batch_size=10 跑出 17 个并发 handler）。
    """

    async def test_max_concurrency_limits_parallel_handlers(self, mq):
        stream = _stream_name(mq, "conc")
        limit = 3
        mq.create_queue(
            stream,
            concurrency=2,
            batch_size=10,
            block_ms=100,
            max_concurrency=limit,
        )
        active = {"now": 0, "max": 0}

        async def handler(msg):
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.2)
            active["now"] -= 1

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        for i in range(20):
            await mq.publish(
                stream, {"task_id": f"t{i}", "task_type": "demo", "payload": {}}
            )
        await asyncio.sleep(2)

        assert active["max"] <= limit, (
            f"并发峰值 {active['max']} 超过上限 {limit}"
        )

    async def test_no_limit_when_not_configured(self, mq):
        """没配 max_concurrency 时不额外限制（行为与旧版一致）。"""
        stream = _stream_name(mq, "nolimit")
        mq.create_queue(stream, concurrency=2, batch_size=5, block_ms=100)

        async def handler(msg):
            pass

        mq.register(stream, "demo", handler)
        await mq.start([stream])
        assert mq._queues[stream].max_concurrency is None

    async def test_semaphore_released_on_handler_error(self, mq):
        """handler 抛异常时也必须释放并发额度——否则额度耗尽队列卡死。"""
        stream = _stream_name(mq, "semaerr")
        mq.create_queue(
            stream, concurrency=1, batch_size=5, block_ms=100, max_concurrency=1
        )
        calls = {"n": 0}

        async def handler(msg):
            calls["n"] += 1
            raise RuntimeError("失败")

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        for i in range(5):
            await mq.publish(
                stream, {"task_id": f"t{i}", "task_type": "demo", "payload": {}}
            )
        # 额度只有 1 个——如果异常路径不释放，第 2 条就会永久阻塞
        await asyncio.sleep(4)
        assert calls["n"] >= 2, f"并发额度泄漏，handler 只调用了 {calls['n']} 次"


# ============================================================
# 10. 幂等：handler 能拿到 msg_id
# ============================================================
class TestMessageIdForIdempotency:
    """handler 应能拿到 MQ 消息 ID——「至少一次」投递下做幂等需要它。"""

    async def test_handler_receives_msg_id(self, mq):
        stream = _stream_name(mq, "msgid")
        mq.create_queue(stream, concurrency=1, block_ms=100)
        received = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        published_id = await mq.publish(
            stream, {"task_id": "t", "task_type": "demo", "payload": {}}
        )
        assert await _wait_for(lambda: len(received) == 1)

        assert received[0].msg_id == published_id
        assert received[0].msg_id, "msg_id 不应为空"


# ============================================================
# 11. create_queue 幂等
# ============================================================
class TestCreateQueueIdempotent:
    """重复 create_queue 不应清空已注册的 handler。"""

    async def test_recreate_keeps_handlers(self, mq):
        stream = _stream_name(mq, "recreate")

        async def handler(msg):
            pass

        mq.create_queue(stream)
        mq.register(stream, "demo", handler)
        assert len(mq._handlers[stream]) == 1

        mq.create_queue(stream)
        assert len(mq._handlers[stream]) == 1, "重复 create_queue 清空了 handler"


# ============================================================
# 12. 死信队列管理
# ============================================================
class TestDeadLetterManagement:
    """死信队列应可查询与清理。"""

    async def test_get_and_clear_dead_letters(self, mq, real_redis):
        stream = _stream_name(mq, "dlmgmt")
        mq.create_queue(stream, concurrency=1, block_ms=100)
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "t", "task_type": "无处理器", "payload": {"x": 1}}
        )
        assert await _wait_for(
            lambda: _dead_has_msg(real_redis, stream), timeout=8.0
        )

        dead = await mq.get_dead_letters(stream)
        assert len(dead) == 1
        assert dead[0].get("reason") == "no_handler"
        assert dead[0].get("original_id")

        cleared = await mq.clear_dead_letters(stream)
        assert cleared == 1
        assert not await _dead_has_msg(real_redis, stream)

    async def test_clear_dead_letters_on_empty_queue(self, mq):
        stream = _stream_name(mq, "dlempty")
        mq.create_queue(stream)
        assert await mq.clear_dead_letters(stream) == 0


# ============================================================
# 13. get_stats 不应有副作用
# ============================================================
class TestGetStatsNoSideEffect:
    """统计查询是只读操作——不该触发阻塞式重连。"""

    async def test_get_stats_without_connection_returns_error(self, mq):
        stream = _stream_name(mq, "nostats")
        mq.create_queue(stream)
        mq._redis = None
        result = await mq.get_stats()
        assert "error" in result

    async def test_get_stats_works_when_connected(self, mq):
        stream = _stream_name(mq, "stats")
        mq.create_queue(stream)
        result = await mq.get_stats(stream)
        assert "error" not in result
        assert result["stream_name"] == stream


# ============================================================
# 14. 按优先级消费
# ============================================================
class TestPriorityConsumption:
    """多优先级队列：真正的优先级消费。

    实现方式：**一个逻辑队列 = 多个 Redis Stream**
    （``q:critical`` / ``q:high`` / ``q:normal`` / ``q:low``）。
    消费者用**一次** XREADGROUP 按优先级顺序读这些 stream——Redis 按传入的
    streams 参数顺序分组返回，所以优先级顺序是 **Redis 原生保证的**。

    为什么不用单 stream + 应用层排序：单 stream 里 critical 消息若排在
    第 5 批，前面 4 批的 low 已经处理完了——那不是真正的优先级。
    """

    async def test_priority_order_is_respected(self, mq, real_redis):
        stream = _stream_name(mq, "prio")
        mq.create_queue(
            stream,
            concurrency=1,
            block_ms=200,
            priorities=["critical", "high", "normal", "low"],
        )
        order: List[str] = []

        async def handler(msg):
            order.append(msg.priority)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        # **逆优先级**发布——low 最先，critical 最后
        for p in ("low", "normal", "high", "critical"):
            await mq.publish(
                stream,
                {"task_id": f"t-{p}", "task_type": "demo",
                 "priority": p, "payload": {}},
            )
        await asyncio.sleep(2)

        assert order == ["critical", "high", "normal", "low"], (
            f"消费顺序不是按优先级: {order}"
        )

    async def test_messages_stored_in_separate_streams(self, mq, real_redis):
        """不同优先级的消息应落在不同的 stream 里。"""
        stream = _stream_name(mq, "prio2")
        mq.create_queue(
            stream, concurrency=1, block_ms=100,
            priorities=["critical", "normal"],
        )
        config = mq._queues[stream]
        assert config.priority_streams == [
            f"{stream}:critical", f"{stream}:normal"
        ]

        async def handler(msg):
            pass

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "c", "task_type": "demo",
                     "priority": "critical", "payload": {}},
        )
        await mq.publish(
            stream, {"task_id": "n", "task_type": "demo",
                     "priority": "normal", "payload": {}},
        )
        await asyncio.sleep(1)

        assert await real_redis.xlen(f"{stream}:critical") == 1
        assert await real_redis.xlen(f"{stream}:normal") == 1

    async def test_acked_on_correct_stream(self, mq, real_redis):
        """ACK 必须打到消息实际所在的 stream——否则消息永远留在 PEL。"""
        stream = _stream_name(mq, "prio3")
        mq.create_queue(
            stream, concurrency=1, block_ms=100,
            priorities=["critical", "normal"],
        )
        received = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "c", "task_type": "demo",
                     "priority": "critical", "payload": {}},
        )
        ok = await _wait_for(lambda: len(received) == 1, timeout=8.0)
        assert ok, "消息未被消费"

        # 各 stream 的 PEL 都应为空
        for st in mq._queues[stream].priority_streams:
            pending = await real_redis.xpending_range(
                st, f"{stream}_group", "-", "+", count=10
            )
            assert len(pending) == 0, f"{st} 仍有未 ACK 消息: {pending}"

    async def test_single_priority_queue_unchanged(self, mq, real_redis):
        """不配 priorities 时行为与之前完全一致（单 stream）。"""
        stream = _stream_name(mq, "prio4")
        mq.create_queue(stream, concurrency=1, block_ms=100)
        assert mq._queues[stream].priority_streams == [stream]

        received = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "t", "task_type": "demo", "payload": {}},
        )
        assert await _wait_for(lambda: len(received) == 1, timeout=8.0)

    async def test_unconfigured_priority_falls_back(self, mq, real_redis):
        """消息的优先级未在队列中启用时——回落到最低优先级 stream，不丢消息。"""
        stream = _stream_name(mq, "prio5")
        mq.create_queue(
            stream, concurrency=1, block_ms=100,
            priorities=["critical", "normal"],
        )
        received = []

        async def handler(msg):
            received.append(msg)

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "t", "task_type": "demo",
                     "priority": "low", "payload": {}},
        )
        ok = await _wait_for(lambda: len(received) == 1, timeout=8.0)
        assert ok, "未启用优先级的消息被丢弃了"
        # 回落到最低优先级的 stream
        assert await real_redis.xlen(f"{stream}:normal") == 1

    async def test_dead_letter_in_priority_queue(self, mq, real_redis):
        """多优先级队列下的重试与死信同样有效。"""
        stream = _stream_name(mq, "prio6")
        mq.create_queue(
            stream, concurrency=1, block_ms=100, claim_interval=1,
            claim_min_idle_ms=300, max_retries=2,
            priorities=["critical", "normal"],
        )
        calls = {"n": 0}

        async def handler(msg):
            calls["n"] += 1
            raise RuntimeError("失败")

        mq.register(stream, "demo", handler)
        await mq.start([stream])

        await mq.publish(
            stream, {"task_id": "t", "task_type": "demo",
                     "priority": "critical", "payload": {}},
        )
        ok = await _wait_for(
            lambda: _dead_has_msg(real_redis, stream), timeout=12.0
        )
        assert ok, "多优先级队列的消息未进死信"

        stats = await mq.get_stats(stream)
        assert stats["pending_count"] == 0, "消息仍卡在 PEL"
