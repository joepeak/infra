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
        # 默认不带任何标志
        mock.zadd.assert_called_once_with(
            "z", {"a": 1.0, "b": 2.0}, nx=False, xx=False, ch=False, gt=False, lt=False
        )

    async def test_zadd_with_flags(self, client_with_mock_redis):
        """zadd 的 nx / xx / ch / gt / lt 标志。"""
        c, mock = client_with_mock_redis
        mock.zadd = AsyncMock(return_value=1)
        result = await c.zadd("z", {"a": 1.0}, nx=True)
        assert result == 1
        mock.zadd.assert_called_once_with(
            "z", {"a": 1.0}, nx=True, xx=False, ch=False, gt=False, lt=False
        )

    async def test_zadd_empty_mapping(self, client_with_mock_redis):
        """空 mapping 直接返回 0，不发命令。"""
        c, mock = client_with_mock_redis
        mock.zadd = AsyncMock(return_value=0)
        assert await c.zadd("z", {}) == 0
        mock.zadd.assert_not_called()

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
        """set_nx 成功——SET key token NX EX ttl 原子加锁，不再用 setnx。"""
        c, mock = client_with_mock_redis
        c.set_nx = AsyncMock(return_value=True)
        result = await c.lock("mykey", ttl=60)
        assert result is True
        c.set_nx.assert_awaited_once()
        args, kwargs = c.set_nx.call_args
        assert args[0] == "lock:mykey"
        assert isinstance(args[1], str) and len(args[1]) > 0   # 唯一 token
        assert kwargs["ttl"] == 60
        # token 被记录以备 unlock 校验
        assert c._lock_tokens["mykey"] == args[1]

    async def test_lock_already_held(self, client_with_mock_redis):
        """set_nx 失败——返回 False，不记录 token。"""
        c, mock = client_with_mock_redis
        c.set_nx = AsyncMock(return_value=False)
        result = await c.lock("k", ttl=30)
        assert result is False
        assert "k" not in c._lock_tokens

    async def test_unlock_success(self, client_with_mock_redis):
        """unlock——Lua 脚本校验 token 匹配才 DEL。"""
        c, mock = client_with_mock_redis
        c.set_nx = AsyncMock(return_value=True)
        mock.eval = AsyncMock(return_value=1)
        await c.lock("mykey", ttl=60)
        token = c._lock_tokens["mykey"]
        result = await c.unlock("mykey")
        assert result is True
        mock.eval.assert_awaited_once()
        args = mock.eval.call_args.args
        assert args[1] == 1                      # numkeys
        assert args[2] == "lock:mykey"             # KEYS[1]
        assert args[3] == token                   # ARGV[1] token

    async def test_unlock_wrong_token(self, client_with_mock_redis):
        """unlock 显式 token 不匹配——Lua 返回 0，锁不被删除。"""
        c, mock = client_with_mock_redis
        c._lock_tokens["mykey"] = "owner-token"
        mock.eval = AsyncMock(return_value=0)
        result = await c.unlock("mykey", token="wrong-token")
        assert result is False
        mock.eval.assert_awaited_once()
        assert "wrong-token" in mock.eval.call_args.args

    async def test_unlock_no_lock(self, client_with_mock_redis):
        """unlock 无持有 token——不调 eval，返回 False。"""
        c, mock = client_with_mock_redis
        mock.eval = AsyncMock()
        result = await c.unlock("nonexistent")
        assert result is False
        mock.eval.assert_not_called()

    async def test_lock_context_acquire(self, client_with_mock_redis):
        """lock_context 获得锁——yield True，exit 时 Lua unlock。"""
        c, mock = client_with_mock_redis
        c.set_nx = AsyncMock(return_value=True)
        mock.eval = AsyncMock(return_value=1)
        async with c.lock_context("k", ttl=30) as acquired:
            assert acquired is True
        mock.eval.assert_awaited_once()
        assert "lock:k" in mock.eval.call_args.args

    async def test_lock_context_fail_no_unlock(self, client_with_mock_redis):
        """lock_context 拿不到锁——不调 unlock（防误删别人锁）。"""
        c, mock = client_with_mock_redis
        c.set_nx = AsyncMock(return_value=False)
        mock.eval = AsyncMock()
        async with c.lock_context("k") as acquired:
            assert acquired is False
        mock.eval.assert_not_called()

    async def test_lock_concurrent_single_winner(self, client_with_mock_redis):
        """多个协程抢同一 key，只有一个能拿到（SET NX 原子语义）。"""
        c, mock = client_with_mock_redis
        store = {}

        async def fake_set_nx(k, value, ttl=None):
            if k in store:
                return False
            store[k] = value
            return True

        c.set_nx = fake_set_nx
        mock.eval = AsyncMock(return_value=1)
        winners = []

        async def worker():
            ok = await c.lock("race", ttl=10)
            if ok:
                winners.append(True)
                await c.unlock("race")

        await asyncio.gather(worker(), worker(), worker())
        assert winners.count(True) == 1
        # 获胜者释放后实例 token 被清空
        assert c._lock_tokens.get("race") is None


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
# 10. Key 扫描 + 批量操作（20261021 新增）
# ============================================================
class TestKeyScanning:
    """keys / scan_iter / delete_pattern——客户需求 1。

    背景：消费方（crypto-watcher）原先被迫用 ``redis._get_client().scan_iter(...)``
    绕过私有方法；这里验证公开 API 可用且返回 str。
    """

    async def test_keys_returns_str_list(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.keys = AsyncMock(return_value=["a:1", "a:2"])
        result = await c.keys("a:*")
        assert result == ["a:1", "a:2"]
        mock.keys.assert_awaited_once_with("a:*")

    async def test_keys_default_pattern_is_star(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.keys = AsyncMock(return_value=[])
        await c.keys()
        mock.keys.assert_awaited_once_with("*")

    async def test_keys_decodes_bytes(self, client_with_mock_redis):
        """bytes key 也统一成 str（自定义连接池 / mock 场景的兜底）。"""
        c, mock = client_with_mock_redis
        mock.keys = AsyncMock(return_value=[b"a:1", "a:2"])
        assert await c.keys() == ["a:1", "a:2"]

    async def test_scan_iter_yields_str(self, client_with_mock_redis):
        c, mock = client_with_mock_redis

        async def fake_scan_iter(match=None, count=None):
            yield b"llm:article_body:1"
            yield "llm:article_body:2"

        mock.scan_iter = fake_scan_iter
        got = [k async for k in c.scan_iter("llm:article_body:*")]
        assert got == ["llm:article_body:1", "llm:article_body:2"]

    async def test_scan_iter_passes_match_and_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        captured = {}

        async def fake_scan_iter(match=None, count=None):
            captured["match"] = match
            captured["count"] = count
            if False:
                yield  # pragma: no cover

        mock.scan_iter = fake_scan_iter
        _ = [k async for k in c.scan_iter("x:*", count=50)]
        assert captured == {"match": "x:*", "count": 50}

    async def test_scan_iter_defaults(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        captured = {}

        async def fake_scan_iter(match=None, count=None):
            captured["match"] = match
            captured["count"] = count
            if False:
                yield  # pragma: no cover

        mock.scan_iter = fake_scan_iter
        _ = [k async for k in c.scan_iter()]
        assert captured == {"match": "*", "count": 100}

    async def test_delete_pattern_batches(self, client_with_mock_redis):
        """按模式删除——分批 delete，返回总数。"""
        c, mock = client_with_mock_redis

        async def fake_scan_iter(match=None, count=None):
            for i in range(5):
                yield f"tmp:{i}"

        mock.scan_iter = fake_scan_iter
        mock.delete = AsyncMock(return_value=2)
        # batch=2 → 3 批（2+2+1）
        assert await c.delete_pattern("tmp:*", batch=2) == 6
        assert mock.delete.await_count == 3

    async def test_delete_pattern_no_match_returns_zero(self, client_with_mock_redis):
        c, mock = client_with_mock_redis

        async def fake_scan_iter(match=None, count=None):
            if False:
                yield  # pragma: no cover

        mock.scan_iter = fake_scan_iter
        mock.delete = AsyncMock()
        assert await c.delete_pattern("none:*") == 0
        mock.delete.assert_not_awaited()


# ============================================================
# 11. 批量操作 + 公开 client 访问器（20261021 新增）
# ============================================================
class TestBatchOps:
    async def test_pipeline_returns_underlying_pipeline(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        sentinel = object()
        mock.pipeline.return_value = sentinel
        assert c.pipeline() is sentinel
        mock.pipeline.assert_called_once_with(transaction=True)

    async def test_pipeline_accepts_transaction_false(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pipeline.return_value = object()
        c.pipeline(transaction=False)
        mock.pipeline.assert_called_once_with(transaction=False)

    async def test_mget(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.mget = AsyncMock(return_value=["v1", None, "v3"])
        assert await c.mget("k1", "k2", "k3") == ["v1", None, "v3"]
        mock.mget.assert_awaited_once_with("k1", "k2", "k3")

    async def test_mget_no_keys_returns_empty(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.mget = AsyncMock()
        assert await c.mget() == []
        mock.mget.assert_not_awaited()

    async def test_mset_serializes_dict(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.mset = AsyncMock(return_value=True)
        assert await c.mset({"a": {"x": 1}, "b": [1, 2]}) is True
        mock.mset.assert_awaited_once_with({"a": '{"x": 1}', "b": "[1, 2]"})

    async def test_mset_converts_non_str(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.mset = AsyncMock(return_value=True)
        await c.mset({"a": 1})
        mock.mset.assert_awaited_once_with({"a": "1"})

    async def test_mset_empty_returns_true(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.mset = AsyncMock()
        assert await c.mset({}) is True
        mock.mset.assert_not_awaited()


# ============================================================
# 12. 公开 client 访问器（替代 _get_client 私有绕过）
# ============================================================
class TestPublicClientAccessor:
    def test_client_property_returns_underlying(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        assert c.client is mock

    def test_client_property_raises_when_uninitialized(self):
        with patch("infra.redis.client.aioredis.Redis"):
            c = RedisClient()
        c._client = None
        with pytest.raises(RedisError, match="客户端未初始化"):
            _ = c.client


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])