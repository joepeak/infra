#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.redis.client 列表 / 集合 / 排序集合能力测试（20261021 新增）

背景：消费方（crypto-watcher）在本轮改造中遇到过真实障碍——
`jin10_service.py` 里有一行注释「infra 的 RedisClient **没有 list 操作**
→ 用 zset + hash」绕道实现，本轮把 List 能力补齐后即可简化。

覆盖：
1. List 基础：lpush / rpush / lpushx / rpushx / llen
2. List 读取：lrange / lindex
3. List 弹出：lpop / rpop（单条与 count 模式）
4. List 阻塞：blpop / brpop（含超时返回 None）
5. List 修改：lset / lrem / ltrim
6. List 移动：rpoplpush / lmove
7. List 序列化：dict/list 自动 JSON
8. Set 集合运算：sinter / sunion / sdiff / scard / *store
9. ZSet 排名：zrank / zrevrank / zremrangebyrank / zrevrange / zrevrangebyscore

aioredis 全部用 AsyncMock——不连真 redis。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.redis import RedisClient


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
# 1. List 基础操作
# ============================================================
class TestListBasicOps:
    async def test_rpush(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpush = AsyncMock(return_value=3)
        assert await c.rpush("q", "a", "b", "c") == 3
        mock.rpush.assert_awaited_once_with("q", "a", "b", "c")

    async def test_lpush(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpush = AsyncMock(return_value=2)
        assert await c.lpush("q", "a", "b") == 2
        mock.lpush.assert_awaited_once_with("q", "a", "b")

    async def test_lpushx(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpushx = AsyncMock(return_value=1)
        assert await c.lpushx("q", "a") == 1
        mock.lpushx.assert_awaited_once_with("q", "a")

    async def test_rpushx(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpushx = AsyncMock(return_value=1)
        assert await c.rpushx("q", "a") == 1
        mock.rpushx.assert_awaited_once_with("q", "a")

    async def test_llen(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.llen = AsyncMock(return_value=5)
        assert await c.llen("q") == 5

    async def test_push_no_values_returns_length(self, client_with_mock_redis):
        """不传值时不发命令——返回当前长度。"""
        c, mock = client_with_mock_redis
        mock.llen = AsyncMock(return_value=3)
        mock.rpush = AsyncMock()
        assert await c.rpush("q") == 3
        mock.rpush.assert_not_awaited()


# ============================================================
# 2. List 读取
# ============================================================
class TestListRead:
    async def test_lrange(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lrange = AsyncMock(return_value=["a", "b", "c"])
        assert await c.lrange("q", 0, -1) == ["a", "b", "c"]
        mock.lrange.assert_awaited_once_with("q", 0, -1)

    async def test_lrange_recent_n(self, client_with_mock_redis):
        """取最近 N 条的常用写法。"""
        c, mock = client_with_mock_redis
        mock.lrange = AsyncMock(return_value=["b", "c"])
        assert await c.lrange("q", -2, -1) == ["b", "c"]
        mock.lrange.assert_awaited_once_with("q", -2, -1)

    async def test_lrange_empty(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lrange = AsyncMock(return_value=[])
        assert await c.lrange("q") == []

    async def test_lindex(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lindex = AsyncMock(return_value="a")
        assert await c.lindex("q", 0) == "a"

    async def test_lindex_missing_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lindex = AsyncMock(return_value=None)
        assert await c.lindex("q", 99) is None


# ============================================================
# 3. List 弹出
# ============================================================
class TestListPop:
    async def test_lpop_single(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpop = AsyncMock(return_value="a")
        assert await c.lpop("q") == "a"
        mock.lpop.assert_awaited_once_with("q")

    async def test_lpop_with_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpop = AsyncMock(return_value=["a", "b"])
        assert await c.lpop("q", 2) == ["a", "b"]
        mock.lpop.assert_awaited_once_with("q", 2)

    async def test_lpop_count_zero_returns_empty(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpop = AsyncMock()
        assert await c.lpop("q", 0) == []
        mock.lpop.assert_not_awaited()

    async def test_lpop_count_empty_result(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpop = AsyncMock(return_value=[])
        assert await c.lpop("q", 3) == []

    async def test_rpop_single(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpop = AsyncMock(return_value="c")
        assert await c.rpop("q") == "c"

    async def test_rpop_with_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpop = AsyncMock(return_value=["c", "b"])
        assert await c.rpop("q", 2) == ["c", "b"]


# ============================================================
# 4. List 阻塞操作
# ============================================================
class TestListBlocking:
    async def test_blpop_returns_tuple(self, client_with_mock_redis):
        """redis-py 的 blpop 直接返回 (key, value) **元组**（不是 [(key, value)]）。"""
        c, mock = client_with_mock_redis
        mock.blpop = AsyncMock(return_value=("q", "a"))
        assert await c.blpop("q", timeout=5) == ("q", "a")
        mock.blpop.assert_awaited_once_with(["q"], timeout=5)

    async def test_blpop_timeout_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.blpop = AsyncMock(return_value=None)
        assert await c.blpop("q", timeout=1) is None

    async def test_brpop_returns_tuple(self, client_with_mock_redis):
        """redis-py 的 brpop 同样直接返回 (key, value) 元组。"""
        c, mock = client_with_mock_redis
        mock.brpop = AsyncMock(return_value=("q", "a"))
        assert await c.brpop("q", timeout=5) == ("q", "a")
        mock.brpop.assert_awaited_once_with(["q"], timeout=5)

    async def test_brpop_timeout_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.brpop = AsyncMock(return_value=None)
        assert await c.brpop("q", timeout=1) is None

    async def test_blpop_decodes_bytes(self, client_with_mock_redis):
        """bytes 也统一成 str。"""
        c, mock = client_with_mock_redis
        mock.blpop = AsyncMock(return_value=(b"q", b"a"))
        assert await c.blpop("q", timeout=1) == ("q", "a")

    async def test_blpop_long_key_not_char_unpacked(self, client_with_mock_redis):
        """回归：key 长度 > 2 时不能被逐字符解包。

        旧实现写成 ``k, v = result[0]``——redis-py 实际返回元组 ``(key, value)``，
        ``result[0]`` 取到的是 key **字符串**，逐字符解包会炸
        （``too many values to unpack``）或静默取错值。
        """
        c, mock = client_with_mock_redis
        long_key = "infra_test_lists:8f3a-uuid:queue"
        mock.blpop = AsyncMock(return_value=(long_key, "payload"))
        assert await c.blpop(long_key, timeout=1) == (long_key, "payload")

    async def test_brpop_long_key_not_char_unpacked(self, client_with_mock_redis):
        """同上的 brpop 版本。"""
        c, mock = client_with_mock_redis
        long_key = "infra_test_lists:8f3a-uuid:stack"
        mock.brpop = AsyncMock(return_value=(long_key, "payload"))
        assert await c.brpop(long_key, timeout=1) == (long_key, "payload")


# ============================================================
# 5. List 修改
# ============================================================
class TestListModify:
    async def test_lset(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lset = AsyncMock(return_value=True)
        assert await c.lset("q", 0, "x") is True
        mock.lset.assert_awaited_once_with("q", 0, "x")

    async def test_lrem(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lrem = AsyncMock(return_value=2)
        assert await c.lrem("q", 2, "a") == 2
        mock.lrem.assert_awaited_once_with("q", 2, "a")

    async def test_lrem_zero_count(self, client_with_mock_redis):
        """count=0 → 删全部匹配。"""
        c, mock = client_with_mock_redis
        mock.lrem = AsyncMock(return_value=5)
        assert await c.lrem("q", 0, "a") == 5

    async def test_ltrim(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.ltrim = AsyncMock(return_value=True)
        assert await c.ltrim("q", 0, 9) is True
        mock.ltrim.assert_awaited_once_with("q", 0, 9)


# ============================================================
# 6. List 元素移动
# ============================================================
class TestListMove:
    async def test_rpoplpush(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpoplpush = AsyncMock(return_value="job1")
        assert await c.rpoplpush("src", "dst") == "job1"
        mock.rpoplpush.assert_awaited_once_with("src", "dst")

    async def test_rpoplpush_empty_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpoplpush = AsyncMock(return_value=None)
        assert await c.rpoplpush("src", "dst") is None

    async def test_lmove(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lmove = AsyncMock(return_value="job1")
        assert await c.lmove("src", "dst", "right", "left") == "job1"
        mock.lmove.assert_awaited_once_with("src", "dst", "right", "left")

    async def test_lmove_defaults(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lmove = AsyncMock(return_value="job1")
        await c.lmove("src", "dst")
        mock.lmove.assert_awaited_once_with("src", "dst", "left", "left")


# ============================================================
# 7. List 序列化
# ============================================================
class TestListSerialization:
    async def test_dict_value_json_encoded(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpush = AsyncMock(return_value=1)
        payload = {"id": 1, "name": "测试"}
        await c.rpush("q", payload)
        mock.rpush.assert_awaited_once_with("q", json.dumps(payload, ensure_ascii=False))

    async def test_list_value_json_encoded(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpush = AsyncMock(return_value=1)
        await c.rpush("q", [1, 2, 3])
        mock.rpush.assert_awaited_once_with("q", "[1, 2, 3]")

    async def test_int_value_str_converted(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpush = AsyncMock(return_value=1)
        await c.rpush("q", 42)
        mock.rpush.assert_awaited_once_with("q", "42")

    async def test_str_value_unchanged(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rpush = AsyncMock(return_value=1)
        await c.rpush("q", "plain")
        mock.rpush.assert_awaited_once_with("q", "plain")

    async def test_lset_json_encodes(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lset = AsyncMock(return_value=True)
        await c.lset("q", 0, {"a": 1})
        mock.lset.assert_awaited_once_with("q", 0, '{"a": 1}')

    async def test_lrem_json_encodes(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lrem = AsyncMock(return_value=1)
        await c.lrem("q", 0, {"a": 1})
        mock.lrem.assert_awaited_once_with("q", 0, '{"a": 1}')


# ============================================================
# 8. Set 集合运算
# ============================================================
class TestSetOperations:
    async def test_sinter(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sinter = AsyncMock(return_value=[b"b", b"c"])
        assert await c.sinter("s1", "s2") == {"b", "c"}
        # redis-py 的签名是 sinter(keys: list)
        mock.sinter.assert_awaited_once_with(["s1", "s2"])

    async def test_sunion(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sunion = AsyncMock(return_value=["a", "b"])
        assert await c.sunion("s1", "s2") == {"a", "b"}
        mock.sunion.assert_awaited_once_with(["s1", "s2"])

    async def test_sdiff(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sdiff = AsyncMock(return_value=["a"])
        assert await c.sdiff("s1", "s2") == {"a"}
        mock.sdiff.assert_awaited_once_with(["s1", "s2"])

    async def test_set_ops_no_keys_returns_empty(self, client_with_mock_redis):
        """不传 key 时直接返回空集，不发命令。"""
        c, mock = client_with_mock_redis
        mock.sinter = AsyncMock()
        mock.sunion = AsyncMock()
        mock.sdiff = AsyncMock()
        assert await c.sinter() == set()
        assert await c.sunion() == set()
        assert await c.sdiff() == set()
        mock.sinter.assert_not_awaited()
        mock.sunion.assert_not_awaited()
        mock.sdiff.assert_not_awaited()

    async def test_sdiffstore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sdiffstore = AsyncMock(return_value=2)
        assert await c.sdiffstore("dst", "s1", "s2") == 2
        mock.sdiffstore.assert_awaited_once_with("dst", ["s1", "s2"])

    async def test_sinterstore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sinterstore = AsyncMock(return_value=1)
        assert await c.sinterstore("dst", "s1", "s2") == 1
        mock.sinterstore.assert_awaited_once_with("dst", ["s1", "s2"])

    async def test_sunionstore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sunionstore = AsyncMock(return_value=3)
        assert await c.sunionstore("dst", "s1", "s2") == 3
        mock.sunionstore.assert_awaited_once_with("dst", ["s1", "s2"])

    async def test_store_no_keys_returns_zero(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sdiffstore = AsyncMock()
        mock.sinterstore = AsyncMock()
        mock.sunionstore = AsyncMock()
        assert await c.sdiffstore("dst") == 0
        assert await c.sinterstore("dst") == 0
        assert await c.sunionstore("dst") == 0
        mock.sdiffstore.assert_not_awaited()
        mock.sinterstore.assert_not_awaited()
        mock.sunionstore.assert_not_awaited()

    async def test_scard(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.scard = AsyncMock(return_value=7)
        assert await c.scard("s1") == 7


# ============================================================
# 9. ZSet 排名相关
# ============================================================
class TestZSetRanking:
    async def test_zrank(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrank = AsyncMock(return_value=1)
        assert await c.zrank("z", "m") == 1

    async def test_zrank_missing_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrank = AsyncMock(return_value=None)
        assert await c.zrank("z", "nope") is None

    async def test_zrevrank(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrevrank = AsyncMock(return_value=0)
        assert await c.zrevrank("z", "m") == 0

    async def test_zrevrank_missing_returns_none(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrevrank = AsyncMock(return_value=None)
        assert await c.zrevrank("z", "nope") is None

    async def test_zremrangebyrank(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zremrangebyrank = AsyncMock(return_value=2)
        assert await c.zremrangebyrank("z", 0, 1) == 2
        mock.zremrangebyrank.assert_awaited_once_with("z", 0, 1)

    async def test_zrevrange(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrevrange = AsyncMock(return_value=["high", "mid"])
        assert await c.zrevrange("z", 0, 1) == ["high", "mid"]
        mock.zrevrange.assert_awaited_once_with("z", 0, 1, withscores=False)

    async def test_zrevrange_withscores(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrevrange = AsyncMock(return_value=[("high", 10.0)])
        assert await c.zrevrange("z", 0, 0, withscores=True) == [("high", 10.0)]

    async def test_zrevrangebyscore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrevrangebyscore = AsyncMock(return_value=["high"])
        assert await c.zrevrangebyscore("z", 10, 1) == ["high"]
        mock.zrevrangebyscore.assert_awaited_once_with("z", 10, 1, withscores=False)

    async def test_zrevrangebyscore_with_pagination(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrevrangebyscore = AsyncMock(return_value=["b"])
        assert await c.zrevrangebyscore("z", 10, 1, offset=0, count=1) == ["b"]
        mock.zrevrangebyscore.assert_awaited_once_with(
            "z", 10, 1, withscores=False, start=0, num=1
        )


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
