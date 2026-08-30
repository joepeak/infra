#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.mq.stream_mq 深度测试

infra 是下游项目的基础设施类库——MQ 是生产关键组件。
完整覆盖 RedisStreamMQ 的所有公开方法 + 边界 + 错误路径。
redis.asyncio 全部用 AsyncMock——不连真 redis。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.mq import (
    MessagePriority,
    QueueConfig,
    QueueName,
    RedisStreamMQ,
    get_mq,
    init_mq,
)
from infra.messages import TaskMessage, create_message


# ============================================================
# 1. 简单数据结构（MessagePriority / QueueName / QueueConfig）
# ============================================================
class TestMessagePriority:
    def test_constants(self):
        assert MessagePriority.CRITICAL == "critical"
        assert MessagePriority.HIGH == "high"
        assert MessagePriority.NORMAL == "normal"
        assert MessagePriority.LOW == "low"

    def test_unique_values(self):
        """4 个优先级互不相同。"""
        values = {
            MessagePriority.CRITICAL,
            MessagePriority.HIGH,
            MessagePriority.NORMAL,
            MessagePriority.LOW,
        }
        assert len(values) == 4


class TestQueueName:
    def test_constants_exist(self):
        """QueueName 是常量类——业务约定。"""
        # 不强求具体值，只验证类存在
        assert QueueName is not None


class TestQueueConfig:
    def test_default_values(self):
        """QueueConfig 默认值锁住——下游配置依赖。"""
        from infra.mq.stream_mq import QueueConfig as Qc
        cfg = Qc(
            stream_name="test:stream",
            group_name="test:group",
        )
        assert cfg.stream_name == "test:stream"
        assert cfg.group_name == "test:group"
        assert cfg.maxlen == 10000
        assert cfg.block_ms == 1000
        assert cfg.batch_size == 10
        assert cfg.max_retries == 3
        assert cfg.concurrency == 5
        assert cfg.claim_interval == 30
        assert cfg.claim_min_idle_ms == 60000
        assert cfg.ttl_seconds == 0

    def test_custom_values(self):
        from infra.mq.stream_mq import QueueConfig as Qc
        cfg = Qc(
            stream_name="x", group_name="y",
            maxlen=100, concurrency=2, ttl_seconds=3600,
        )
        assert cfg.maxlen == 100
        assert cfg.concurrency == 2
        assert cfg.ttl_seconds == 3600


# ============================================================
# 2. RedisStreamMQ 构造
# ============================================================
class TestRedisStreamMqConstruct:
    def test_construct_stores_config(self):
        mq = RedisStreamMQ(redis_config={"host": "redis", "port": 6380, "db": 2})
        assert mq.redis_config == {"host": "redis", "port": 6380, "db": 2}

    def test_construct_initializes_internal_state(self):
        mq = RedisStreamMQ(redis_config={"host": "localhost"})
        # 内部 dict / state 初始化
        assert mq._queues == {}
        assert mq._handlers == {}
        assert mq._routing == {}
        assert mq._consumer_tasks == {}
        assert mq._claim_tasks == {}
        assert mq._running is False
        assert mq._default_queue is None
        assert mq._stats == {
            "total_processed": 0,
            "total_failed": 0,
            "total_expired": 0,
            "last_error": None,
        }

    def test_construct_missing_keys_use_defaults(self):
        """redis_config 缺字段时 _get_redis_url 走默认。"""
        mq = RedisStreamMQ(redis_config={})
        # _get_redis_url 是方法，调用验证
        assert "localhost" in mq._get_redis_url()
        assert "6379" in mq._get_redis_url()

    def test_redis_url_with_password(self):
        mq = RedisStreamMQ(redis_config={"host": "h", "port": 1, "password": "secret"})
        url = mq._get_redis_url()
        assert "secret" in url
        assert ":1/0" in url

    def test_redis_url_no_password(self):
        mq = RedisStreamMQ(redis_config={"host": "h", "port": 1})
        url = mq._get_redis_url()
        # 不应含 :secret@
        assert "@" not in url.split("//")[1]
        assert "h:1" in url


# ============================================================
# 3. create_queue 队列注册
# ============================================================
class TestCreateQueue:
    def test_register_with_defaults(self):
        mq = RedisStreamMQ(redis_config={})
        ret = mq.create_queue("orders")
        # 返回 self 支持链式调用
        assert ret is mq
        # 队列配置注册
        assert "orders" in mq._queues
        cfg = mq._queues["orders"]
        assert cfg.stream_name == "orders"
        assert cfg.group_name == "orders_group"  # 默认 {stream_name}_group
        # handlers dict 初始化
        assert mq._handlers["orders"] == {}

    def test_register_with_custom_group(self):
        mq = RedisStreamMQ(redis_config={})
        mq.create_queue("orders", group_name="my_group")
        assert mq._queues["orders"].group_name == "my_group"

    def test_register_with_kwargs(self):
        mq = RedisStreamMQ(redis_config={})
        mq.create_queue("orders", maxlen=100, concurrency=10, ttl_seconds=600)
        cfg = mq._queues["orders"]
        assert cfg.maxlen == 100
        assert cfg.concurrency == 10
        assert cfg.ttl_seconds == 600

    def test_chain_multiple_queues(self):
        mq = RedisStreamMQ(redis_config={})
        mq.create_queue("a").create_queue("b").create_queue("c")
        assert set(mq._queues.keys()) == {"a", "b", "c"}


# ============================================================
# 4. set_routing 路由规则
# ============================================================
class TestSetRouting:
    def test_routing_stored(self):
        mq = RedisStreamMQ(redis_config={})
        mq.set_routing({"etl": "data_q", "alert": "alert_q"})
        assert mq._routing == {"etl": "data_q", "alert": "alert_q"}

    def test_routing_update_merges(self):
        mq = RedisStreamMQ(redis_config={})
        mq.set_routing({"a": "q1"})
        mq.set_routing({"b": "q2"})
        # set_routing 用 self._routing.update——累加
        assert mq._routing == {"a": "q1", "b": "q2"}

    def test_routing_overrides_existing(self):
        mq = RedisStreamMQ(redis_config={})
        mq.set_routing({"a": "q1", "b": "q2"})
        mq.set_routing({"a": "q3"})
        assert mq._routing["a"] == "q3"


# ============================================================
# 5. register 处理器注册
# ============================================================
class TestRegisterHandler:
    def test_register_existing_queue(self):
        mq = RedisStreamMQ(redis_config={})
        mq.create_queue("orders")
        async def handler(msg):
            return msg
        mq.register("orders", "new_order", handler)
        assert mq._handlers["orders"]["new_order"] is handler

    def test_register_unregistered_queue_raises(self):
        mq = RedisStreamMQ(redis_config={})
        async def handler(msg):
            return msg
        with pytest.raises(ValueError, match="队列不存在"):
            mq.register("nonexistent", "x", handler)


# ============================================================
# 6. set_default_queue 默认队列
# ============================================================
class TestSetDefaultQueue:
    def test_default_queue_stored(self):
        mq = RedisStreamMQ(redis_config={})
        mq.create_queue("default_q")
        mq.set_default_queue("default_q")
        assert mq._default_queue == "default_q"


# ============================================================
# 7. publish 发布消息（mock redis）
# ============================================================
@pytest.fixture
def mq_with_redis():
    """构造 mq + 注入 mock redis（已 connect 状态）。"""
    mq = RedisStreamMQ(redis_config={"host": "h", "port": 1})
    # 关键：connect() 也要 mock 掉，否则会走真 redis
    mq.connect = AsyncMock(return_value=True)
    mq._redis = MagicMock()  # 假装已连
    # xadd 返回 msg_id
    mq._redis.xadd = AsyncMock(return_value=b"1234567890-0")
    return mq


class TestPublish:
    async def test_publish_task_message(self, mq_with_redis):
        mq = mq_with_redis
        mq.create_queue("orders")
        msg = create_message("new_order", {"sku": "ABC"})
        # TaskMessage 应被 to_mq 转换
        result = await mq.publish("orders", msg)
        assert result == b"1234567890-0"
        # 验证 xadd 被调，参数含 data dict
        mq._redis.xadd.assert_called_once()
        call_args = mq._redis.xadd.call_args
        # xadd(stream_name, data, maxlen=...)
        assert call_args.args[0] == "orders"
        data = call_args.args[1]
        assert data["task_type"] == "new_order"
        assert data["task_id"] == msg.task_id
        assert json.loads(data["payload"]) == {"sku": "ABC"}
        # maxlen 来自 QueueConfig
        assert call_args.kwargs["maxlen"] == 10000

    async def test_publish_dict_message(self, mq_with_redis):
        """兼容旧 dict 格式——自动补齐 task_id / time 等字段。"""
        mq = mq_with_redis
        mq.create_queue("orders")
        result = await mq.publish("orders", {
            "task_type": "manual",
            "payload": {"k": "v"},
        })
        assert result == b"1234567890-0"
        data = mq._redis.xadd.call_args.args[1]
        # 自动补齐
        assert "task_id" in data  # uuid 自动生成
        assert "scheduled_time" in data
        assert "created_at" in data

    async def test_publish_unregistered_queue_raises(self, mq_with_redis):
        mq = mq_with_redis
        with pytest.raises(ValueError, match="队列不存在"):
            await mq.publish("nonexistent", {"task_type": "x"})

    async def test_publish_redis_error_returns_none(self, mq_with_redis):
        """xadd 抛异常——返回 None（不抛）。by-design：当前没存 last_error。"""
        mq = mq_with_redis
        mq.create_queue("orders")
        mq._redis.xadd = AsyncMock(side_effect=Exception("redis down"))
        result = await mq.publish("orders", {"task_type": "x"})
        # 不抛错、返 None
        assert result is None

    async def test_publish_redis_not_connected_returns_none(self):
        """connect 失败——publish 返 None。"""
        mq = RedisStreamMQ(redis_config={})
        # 强制 connect 失败
        with patch.object(mq, "connect", new=AsyncMock(return_value=False)):
            mq.create_queue("orders")
            result = await mq.publish("orders", {"task_type": "x"})
        assert result is None


# ============================================================
# 8. publish_by_routing 路由发布
# ============================================================
class TestPublishByRouting:
    async def test_routing_hit(self, mq_with_redis):
        mq = mq_with_redis
        mq.create_queue("etl_q")
        mq.set_routing({"etl": "etl_q"})
        mq._redis.xadd = AsyncMock(return_value=b"msg-1")
        msg = create_message("etl", {"k": "v"})
        result = await mq.publish_by_routing(msg)
        assert result == b"msg-1"
        # 验证 xadd 走到了 etl_q
        assert mq._redis.xadd.call_args.args[0] == "etl_q"

    async def test_routing_miss_falls_back_to_default(self, mq_with_redis):
        """路由表没匹配时——走默认队列。"""
        mq = mq_with_redis
        mq.create_queue("default_q")
        mq.set_default_queue("default_q")
        mq._redis.xadd = AsyncMock(return_value=b"msg-2")
        msg = create_message("unknown_type", {"k": "v"})
        result = await mq.publish_by_routing(msg)
        assert result == b"msg-2"
        assert mq._redis.xadd.call_args.args[0] == "default_q"

    async def test_routing_miss_no_default_raises(self, mq_with_redis):
        """路由未命中 + 无默认队列——抛 ValueError。"""
        mq = mq_with_redis
        msg = create_message("x", {})
        with pytest.raises(ValueError, match="未找到"):
            await mq.publish_by_routing(msg)

    async def test_no_task_type_raises(self, mq_with_redis):
        """消息缺 task_type 字段——抛 ValueError。"""
        mq = mq_with_redis
        # dict 缺 task_type
        with pytest.raises(ValueError, match="缺少 task_type"):
            await mq.publish_by_routing({})


# ============================================================
# 9. publish_simple 默认队列发布
# ============================================================
class TestPublishSimple:
    async def test_publish_to_default_queue(self, mq_with_redis):
        mq = mq_with_redis
        mq.create_queue("default_q")
        mq.set_default_queue("default_q")
        mq._redis.xadd = AsyncMock(return_value=b"msg-3")
        msg = create_message("x", {})
        result = await mq.publish_simple(msg)
        assert result == b"msg-3"
        assert mq._redis.xadd.call_args.args[0] == "default_q"


# ============================================================
# 10. _is_message_expired 时效性
# ============================================================
class TestIsMessageExpired:
    def test_not_expired_when_ttl_zero(self):
        """ttl=0 表示永不过期。"""
        mq = RedisStreamMQ(redis_config={})
        # 任意 created_at，ttl=0 都不过期
        assert mq._is_message_expired("1700000000.0", ttl_seconds=0) is False

    def test_not_expired_within_ttl(self):
        mq = RedisStreamMQ(redis_config={})
        now = "1700000000.0"
        # ttl=60, 消息创建 30 秒前——未过期
        with patch("time.time", return_value=1700000030.0):
            assert mq._is_message_expired(now, ttl_seconds=60) is False

    def test_expired_past_ttl(self):
        mq = RedisStreamMQ(redis_config={})
        now = "1700000000.0"
        # ttl=60, 消息创建 120 秒前——过期
        with patch("time.time", return_value=1700000120.0):
            assert mq._is_message_expired(now, ttl_seconds=60) is True

    def test_invalid_timestamp_not_expired(self):
        """created_at 无法 parse——不抛错，返 False（保守不丢消息）。"""
        mq = RedisStreamMQ(redis_config={})
        # 锁住 by-design 行为
        assert mq._is_message_expired("not a number", ttl_seconds=60) is False


# ============================================================
# 11. connect 连接管理（mock 各种异常）
# ============================================================
class TestConnect:
    async def test_connect_success_first_time(self, monkeypatch):
        """首次 connect 走重连分支，patch redis.from_url。"""
        mq = RedisStreamMQ(redis_config={"host": "h", "port": 1})
        fake_redis = MagicMock()
        fake_redis.ping = AsyncMock(return_value=True)

        # 直接 patch 源码里 `import redis.asyncio as redis` 后 `redis.from_url` 调用的位置
        monkeypatch.setattr("infra.mq.stream_mq.redis.from_url", AsyncMock(return_value=fake_redis))
        # 防 reconnect 的 asyncio.sleep 真实等待——mock 掉
        monkeypatch.setattr("infra.mq.stream_mq.asyncio.sleep", AsyncMock())
        result = await mq.connect()
        assert result is True
        assert mq._redis is fake_redis
        assert mq._reconnect_count == 0

    async def test_connect_already_connected_uses_existing(self):
        """_redis 已存在 + ping 成功——直接返 True，不重建。"""
        mq = RedisStreamMQ(redis_config={})
        existing = MagicMock()
        existing.ping = AsyncMock(return_value=True)
        mq._redis = existing
        result = await mq.connect()
        assert result is True
        assert mq._redis is existing

    async def test_connect_ping_fails_resets_and_reconnects(self, monkeypatch):
        """现有连接 ping 失败——重置 _redis 走重连。"""
        mq = RedisStreamMQ(redis_config={"host": "h", "port": 1})
        bad = MagicMock()
        bad.ping = AsyncMock(side_effect=Exception("conn lost"))
        mq._redis = bad

        good = MagicMock()
        good.ping = AsyncMock(return_value=True)
        monkeypatch.setattr("infra.mq.stream_mq.redis.from_url", AsyncMock(return_value=good))
        monkeypatch.setattr("infra.mq.stream_mq.asyncio.sleep", AsyncMock())
        result = await mq.connect()
        assert result is True
        assert mq._redis is good

    async def test_connect_all_retries_fail(self, monkeypatch):
        """重连 10 次全失败——返 False。"""
        mq = RedisStreamMQ(redis_config={"host": "h", "port": 1})
        monkeypatch.setattr(
            "infra.mq.stream_mq.redis.from_url",
            AsyncMock(side_effect=Exception("always fail")),
        )
        # mock sleep 防真实等待
        monkeypatch.setattr("infra.mq.stream_mq.asyncio.sleep", AsyncMock())
        result = await mq.connect()
        assert result is False
        assert mq._reconnect_count == mq._max_reconnect


# ============================================================
# 12. 全局 singleton get_mq / init_mq
# ============================================================
class TestGlobalSingleton:
    def setup_method(self):
        """每个测试清空全局 mq。"""
        import infra.mq.stream_mq as mq_mod
        mq_mod._mq = None

    async def test_init_mq_creates_singleton(self, monkeypatch):
        """init_mq 是 async——传 config dict 创建全局单例。"""
        # mock 掉 connect 避免真连 redis
        from infra.mq import stream_mq as mq_mod
        monkeypatch.setattr(mq_mod.RedisStreamMQ, "connect", AsyncMock(return_value=True))
        # queues 用 normal_queue（init_mq 内部 hardcode 这个 default）
        mq = await init_mq(
            config={"redis": {"host": "h", "port": 1}},
            routing={"etl": "normal_queue"},
            queues=[{"name": "normal_queue"}],
        )
        assert isinstance(mq, RedisStreamMQ)
        # get_mq 返同一实例
        assert get_mq() is mq

    def test_get_mq_before_init_raises(self):
        """未 init 时 get_mq 抛 RuntimeError（不是返 None）。"""
        import infra.mq.stream_mq as mq_mod
        mq_mod._mq = None
        with pytest.raises(RuntimeError, match="未初始化"):
            get_mq()

    async def test_init_mq_replaces_existing(self, monkeypatch):
        """重复 init_mq 替换前一个。"""
        from infra.mq import stream_mq as mq_mod
        monkeypatch.setattr(mq_mod.RedisStreamMQ, "connect", AsyncMock(return_value=True))
        m1 = await init_mq(
            config={"redis": {"host": "h1"}},
            routing={"a": "normal_queue"},
            queues=[{"name": "normal_queue"}],
        )
        m2 = await init_mq(
            config={"redis": {"host": "h2"}},
            routing={"b": "normal_queue"},
            queues=[{"name": "normal_queue"}],
        )
        # 后者替换
        assert get_mq() is m2
        assert m1 is not m2


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
