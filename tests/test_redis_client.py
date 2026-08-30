#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.redis.client 深度测试

infra 是下游项目的基础设施类库——redis 是生产关键。
完整覆盖 RedisClient 的所有公开方法 + 边界 + 错误路径。
aioredis 全部用 AsyncMock——不连真 redis。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.redis import RedisClient, close_redis, get_redis, init_redis
from infra.exceptions import RedisError


# ============================================================
# Fixture：注入 mock redis client（不连真 redis）
# ============================================================
@pytest.fixture
def client_with_mock_redis():
    """构造 RedisClient + 注入 mock _client（mock init_connection）。"""
    with patch("infra.redis.client.aioredis.Redis") as mock_redis_class:
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        c = RedisClient(config={"host": "mock", "port": 6379})
        yield c, mock_client


# ============================================================
# 1. 构造 + 状态初始化
# ============================================================
class TestRedisClientConstruct:
    def test_default_config(self, monkeypatch):
        """不传 config——走默认值。"""
        with patch("infra.redis.client.aioredis.Redis"):
            c = RedisClient()
        assert c.config == {}

    def test_custom_config(self, monkeypatch):
        with patch("infra.redis.client.aioredis.Redis"):
            c = RedisClient(config={"host": "h", "port": 1, "db": 2})
        assert c.config["host"] == "h"
        assert c.config["port"] == 1

    def test_init_connection_failure_raises_redis_error(self, monkeypatch):
        """init_connection 抛错——包装为 RedisError。"""
        from infra.redis import client as rc_mod
        # mock ConnectionPool 让它构造时抛错
        def fake_pool_fail(*args, **kwargs):
            raise Exception("conn failed")
        monkeypatch.setattr(rc_mod, "ConnectionPool", fake_pool_fail)
        with pytest.raises(RedisError, match="Redis 连接失败"):
            RedisClient(config={"host": "h"})

    def test_get_client_raises_if_uninitialized(self, monkeypatch):
        """_client = None 时 _get_client 抛 RedisError。"""
        with patch("infra.redis.client.aioredis.Redis"):
            c = RedisClient()
        c._client = None
        with pytest.raises(RedisError, match="客户端未初始化"):
            c._get_client()


# ============================================================
# 2. 基础字符串操作
# ============================================================
class TestBasicStringOps:
    async def test_get(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.get = AsyncMock(return_value="value")
        result = await c.get("key")
        assert result == "value"
        mock.get.assert_called_once_with("key")

    async def test_get_missing_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.get = AsyncMock(return_value=None)
        result = await c.get("missing")
        assert result is None

    async def test_set_without_ttl(self, client_with_mock_redis):
        """set 无 TTL——调 set。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        result = await c.set("k", "v")
        assert result is True
        mock.set.assert_called_once_with("k", "v")

    async def test_set_with_ttl_uses_set(self, client_with_mock_redis):
        """set 有 TTL——调 set(key, value, ex=ttl)（redis-py 5.0+ 替代 setex）。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        result = await c.set("k", "v", ttl=60)
        assert result is True
        mock.set.assert_called_once_with("k", "v", ex=60)

    async def test_set_dict_json_encodes(self, client_with_mock_redis):
        """set dict——自动 JSON 序列化。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        await c.set("k", {"city": "上海"})
        # 验证传给 redis 的是 JSON 字符串
        call_args = mock.set.call_args
        # 位置参数
        assert json.loads(call_args.args[1]) == {"city": "上海"}

    async def test_set_list_json_encodes(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        await c.set("k", [1, 2, 3])
        assert json.loads(mock.set.call_args.args[1]) == [1, 2, 3]

    async def test_set_int_str_converts(self, client_with_mock_redis):
        """set int——转 str。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        await c.set("k", 42)
        assert mock.set.call_args.args[1] == "42"

    async def test_set_unicode_preserved(self, client_with_mock_redis):
        """set unicode——ensure_ascii=False 保留中文。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        await c.set("k", "中文测试")
        # 字符串直传——不是 JSON（不走 dumps）
        assert "中文测试" in mock.set.call_args.args[1]

    async def test_delete_keys(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.delete = AsyncMock(return_value=2)
        result = await c.delete("k1", "k2", "k3")
        assert result == 2
        mock.delete.assert_called_once_with("k1", "k2", "k3")

    async def test_delete_no_keys_returns_zero(self, client_with_mock_redis):
        """delete 不传 keys——返 0，不调 redis。"""
        c, mock = client_with_mock_redis
        mock.delete = AsyncMock()
        result = await c.delete()
        assert result == 0
        mock.delete.assert_not_called()

    async def test_exists_true(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.exists = AsyncMock(return_value=1)
        assert await c.exists("k") is True

    async def test_exists_false(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.exists = AsyncMock(return_value=0)
        assert await c.exists("k") is False

    async def test_expire(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.expire = AsyncMock(return_value=True)
        result = await c.expire("k", 60)
        assert result is True
        mock.expire.assert_called_once_with("k", 60)

    async def test_ttl(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.ttl = AsyncMock(return_value=30)
        result = await c.ttl("k")
        assert result == 30


# ============================================================
# 3. 计数器
# ============================================================
class TestCounterOps:
    async def test_incr_default_amount(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.incrby = AsyncMock(return_value=1)
        result = await c.incr("k")
        assert result == 1
        mock.incrby.assert_called_once_with("k", 1)  # 默认 amount=1

    async def test_incr_custom_amount(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.incrby = AsyncMock(return_value=10)
        result = await c.incr("k", amount=5)
        assert result == 10
        mock.incrby.assert_called_once_with("k", 5)

    async def test_decr_default_amount(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.decrby = AsyncMock(return_value=-1)
        result = await c.decr("k")
        assert result == -1
        mock.decrby.assert_called_once_with("k", 1)


# ============================================================
# 4. Hash 操作
# ============================================================
class TestHashOps:
    async def test_hset(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hset = AsyncMock(return_value=1)
        result = await c.hset("h", "f", "v")
        assert result == 1
        mock.hset.assert_called_once_with("h", "f", "v")

    async def test_hget_existing(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hget = AsyncMock(return_value="v")
        result = await c.hget("h", "f")
        assert result == "v"

    async def test_hget_missing(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hget = AsyncMock(return_value=None)
        result = await c.hget("h", "f")
        assert result is None

    async def test_hgetall(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hgetall = AsyncMock(return_value={"a": "1", "b": "2"})
        result = await c.hgetall("h")
        assert result == {"a": "1", "b": "2"}

    async def test_hdel(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hdel = AsyncMock(return_value=2)
        result = await c.hdel("h", "f1", "f2")
        assert result == 2
        mock.hdel.assert_called_once_with("h", "f1", "f2")


# ============================================================
# 5. Set 操作
# ============================================================
class TestSetOps:
    async def test_sadd(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sadd = AsyncMock(return_value=2)
        result = await c.sadd("s", "m1", "m2")
        assert result == 2
        mock.sadd.assert_called_once_with("s", "m1", "m2")

    async def test_sismember_true(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sismember = AsyncMock(return_value=True)
        assert await c.sismember("s", "m") is True

    async def test_sismember_false(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sismember = AsyncMock(return_value=False)
        assert await c.sismember("s", "m") is False

    async def test_smembers(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.smembers = AsyncMock(return_value={"a", "b", "c"})
        result = await c.smembers("s")
        assert result == {"a", "b", "c"}

    async def test_srem(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.srem = AsyncMock(return_value=1)
        result = await c.srem("s", "m1")
        assert result == 1


# ============================================================
# 6. Sorted Set 操作
# ============================================================
class TestSortedSetOps:
    async def test_zadd(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zadd = AsyncMock(return_value=2)
        result = await c.zadd("z", {"a": 1.0, "b": 2.0})
        assert result == 2
        mock.zadd.assert_called_once_with("z", {"a": 1.0, "b": 2.0})

    async def test_zremrangebyscore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zremrangebyscore = AsyncMock(return_value=3)
        result = await c.zremrangebyscore("z", 0, 100)
        assert result == 3
        mock.zremrangebyscore.assert_called_once_with("z", 0, 100)

    async def test_zcard(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zcard = AsyncMock(return_value=5)
        result = await c.zcard("z")
        assert result == 5

    async def test_zrange_default(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrange = AsyncMock(return_value=["a", "b"])
        result = await c.zrange("z", 0, -1)
        assert result == ["a", "b"]
        mock.zrange.assert_called_once_with("z", 0, -1, withscores=False)

    async def test_zrange_with_scores(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrange = AsyncMock(return_value=[("a", 1.0), ("b", 2.0)])
        result = await c.zrange("z", 0, -1, withscores=True)
        assert result == [("a", 1.0), ("b", 2.0)]
        mock.zrange.assert_called_once_with("z", 0, -1, withscores=True)

    async def test_zrangebyscore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrangebyscore = AsyncMock(return_value=["a"])
        result = await c.zrangebyscore("z", 0, 100)
        assert result == ["a"]

    async def test_zrem(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrem = AsyncMock(return_value=1)
        result = await c.zrem("z", "a")
        assert result == 1

    async def test_zscore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zscore = AsyncMock(return_value=1.5)
        result = await c.zscore("z", "a")
        assert result == 1.5

    async def test_zscore_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zscore = AsyncMock(return_value=None)
        result = await c.zscore("z", "missing")
        assert result is None

    async def test_zcount(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zcount = AsyncMock(return_value=10)
        result = await c.zcount("z", 0, 100)
        assert result == 10

    async def test_zpopmin(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zpopmin = AsyncMock(return_value=[("a", 1.0)])
        result = await c.zpopmin("z")
        assert result == [("a", 1.0)]
        mock.zpopmin.assert_called_once_with("z", 1)

    async def test_zpopmax(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zpopmax = AsyncMock(return_value=[("z", 9.0)])
        result = await c.zpopmax("z")
        assert result == [("z", 9.0)]
        mock.zpopmax.assert_called_once_with("z", 1)

    async def test_zpopmin_custom_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zpopmin = AsyncMock(return_value=[])
        await c.zpopmin("z", count=5)
        mock.zpopmin.assert_called_once_with("z", 5)


# ============================================================
# 7. 分布式锁
# ============================================================
class TestDistributedLock:
    async def test_lock_acquire(self, client_with_mock_redis):
        """setnx 成功——加锁成功，expire 被调。"""
        c, mock = client_with_mock_redis
        mock.setnx = AsyncMock(return_value=True)
        mock.expire = AsyncMock(return_value=True)
        result = await c.lock("mykey", ttl=60)
        assert result is True
        mock.setnx.assert_called_once()
        # 验证 lock key 前缀
        lock_key = mock.setnx.call_args.args[0]
        assert lock_key == "lock:mykey"
        mock.expire.assert_called_once_with(lock_key, 60)

    async def test_lock_already_held(self, client_with_mock_redis):
        """setnx 失败——不加锁，不调 expire。"""
        c, mock = client_with_mock_redis
        mock.setnx = AsyncMock(return_value=False)
        mock.expire = AsyncMock()
        result = await c.lock("k", ttl=30)
        assert result is False
        mock.expire.assert_not_called()

    async def test_unlock_success(self, client_with_mock_redis):
        """unlock——内部用 delete('lock:key')，返 delete 结果。"""
        c, mock = client_with_mock_redis
        mock.delete = AsyncMock(return_value=1)
        result = await c.unlock("mykey")
        assert result is True
        # 验证调了 lock:mykey
        mock.delete.assert_called_once_with("lock:mykey")

    async def test_unlock_no_lock(self, client_with_mock_redis):
        """unlock——delete 返 0（无锁），返 False。"""
        c, mock = client_with_mock_redis
        mock.delete = AsyncMock(return_value=0)
        result = await c.unlock("nonexistent")
        assert result is False

    async def test_lock_context_acquire(self, client_with_mock_redis):
        """lock_context 立即获得锁——yield True，exit 时 unlock。"""
        c, mock = client_with_mock_redis
        mock.setnx = AsyncMock(return_value=True)
        mock.expire = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        async with c.lock_context("k", ttl=30) as acquired:
            assert acquired is True
        # exit 后 unlock 被调
        mock.delete.assert_called_once_with("lock:k")

    async def test_lock_context_fail_no_unlock(self, client_with_mock_redis):
        """lock_context 拿不到锁——不调 unlock。"""
        c, mock = client_with_mock_redis
        mock.setnx = AsyncMock(return_value=False)
        mock.delete = AsyncMock()
        async with c.lock_context("k") as acquired:
            assert acquired is False
        # 拿不到锁——不应 unlock（避免误删别人的锁）
        mock.delete.assert_not_called()


# ============================================================
# 8. get_or_set 缓存
# ============================================================
class TestGetOrSet:
    async def test_cache_hit_dict(self, client_with_mock_redis):
        """缓存命中——JSON 反序列化。"""
        c, mock = client_with_mock_redis
        mock.get = AsyncMock(return_value=json.dumps({"k": "v"}))
        result = await c.get_or_set("k", lambda: "should not call")
        assert result == {"k": "v"}

    async def test_cache_hit_string(self, client_with_mock_redis):
        """缓存命中——非 JSON 字符串原样返回。"""
        c, mock = client_with_mock_redis
        mock.get = AsyncMock(return_value="raw_string")
        result = await c.get_or_set("k", lambda: "should not call")
        assert result == "raw_string"

    async def test_get_or_set_cache_miss_sync_func(self, client_with_mock_redis):
        """缓存未命中——调 sync func，结果 set 后返。"""
        c, mock = client_with_mock_redis
        mock.get = AsyncMock(return_value=None)
        # 不传 ttl——set 走 set 分支（不调 setex）
        mock.set = AsyncMock(return_value=True)
        result = await c.get_or_set("k", lambda: {"new": "data"})
        assert result == {"new": "data"}
        mock.set.assert_called_once()

    async def test_get_or_set_cache_miss_async_func(self, client_with_mock_redis):
        """缓存未命中——调 async func（coroutine function）。"""
        c, mock = client_with_mock_redis
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock(return_value=True)

        async def async_loader():
            return {"async": "result"}

        result = await c.get_or_set("k", async_loader)
        assert result == {"async": "result"}
        mock.set.assert_called_once()


# ============================================================
# 9. 健康检查 + 关闭
# ============================================================
class TestHealthAndClose:
    async def test_ping_success(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.ping = AsyncMock(return_value=True)
        assert await c.ping() is True

    async def test_ping_failure_returns_false(self, client_with_mock_redis):
        """ping 抛异常——返 False（不抛）。"""
        c, mock = client_with_mock_redis
        mock.ping = AsyncMock(side_effect=Exception("conn lost"))
        assert await c.ping() is False

    async def test_close(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.aclose = AsyncMock()
        await c.close()
        mock.aclose.assert_called_once()

    async def test_close_with_disconnect_pool(self, client_with_mock_redis):
        """close 调 pool.disconnect（await）。"""
        c, mock = client_with_mock_redis
        mock.aclose = AsyncMock()
        # pool.disconnect 是 await 调用——必须用 AsyncMock
        c._pool.disconnect = AsyncMock()
        await c.close()
        c._pool.disconnect.assert_called_once()
        mock.aclose.assert_called_once()


# ============================================================
# 10. 全局 init_redis / get_redis / close_redis
# ============================================================
class TestGlobalRegistry:
    def setup_method(self):
        """每个测试清空全局。"""
        import infra.redis.client as rc
        rc._redis_client = None

    async def test_init_redis_returns_client(self, monkeypatch):
        """init_redis 构造 RedisClient 并存到全局。"""
        with patch("infra.redis.client.aioredis.Redis"):
            client = await init_redis(config={"host": "h", "port": 1})
        assert isinstance(client, RedisClient)
        # get_redis 返同一实例
        assert get_redis() is client

    def test_get_redis_before_init_returns_none(self):
        """未 init 时 get_redis 返 None。"""
        import infra.redis.client as rc
        rc._redis_client = None
        assert get_redis() is None

    async def test_init_redis_replaces_existing(self, monkeypatch):
        """重复 init_redis 替换前一个。"""
        with patch("infra.redis.client.aioredis.Redis"):
            c1 = await init_redis(config={"host": "h1"})
            c2 = await init_redis(config={"host": "h2"})
        assert get_redis() is c2
        assert c1 is not c2

    async def test_close_redis(self, monkeypatch):
        """close_redis 调 close + 清全局。"""
        with patch("infra.redis.client.aioredis.Redis"):
            c = await init_redis(config={"host": "h"})
            c.close = AsyncMock()
            await close_redis()
        c.close.assert_called_once()
        assert get_redis() is None

    async def test_close_redis_when_uninitialized(self):
        """未 init 时 close_redis——不抛。"""
        import infra.redis.client as rc
        rc._redis_client = None
        await close_redis()  # 不抛


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
