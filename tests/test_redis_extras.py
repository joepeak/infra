#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RedisClient 新增命令的测试（20261021 增强）

覆盖本次新增的常用命令：
- String: pttl / persist / pexpire / expireat / type / strlen / append /
  getrange / setrange / getdel / getex / unlink / dbsize / rename / copy / touch
- Batch: msetnx
- Hash: hlen / hexists / hmget / hkeys / hvals / hincrby / hincrbyfloat /
  hsetnx / hset(mapping=) / hdel(多字段)
- Set: srandmember / spop / smove / sintercard
- List: lpos / linsert
- ZSet: zmscore / zrandmember / zunion / zinter / zdiff /
  zunionstore / zinterstore / zdiffstore
- HyperLogLog: pfadd / pfcount / pfmerge
- 运维: memory_usage / info

同时覆盖本次修复的既有缺陷：
- hgetall / smembers 的 bytes 解码（原来用 str() 会得到 "b'k'"）
- set_nx 自身的参数拼装（此前零直接测试，只在 lock 测试里被 mock 掉）

依赖：mock（不连真 redis）。真 Redis 行为见 tests/test_redis_lists_integration.py。
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.redis.client import RedisClient


@pytest.fixture
def client_with_mock_redis():
    """构造 RedisClient + 注入 mock _client（mock init_connection）。"""
    with patch("infra.redis.client.aioredis.Redis") as mock_redis_class:
        mock_client = MagicMock()
        mock_redis_class.return_value = mock_client
        c = RedisClient(config={"host": "mock", "port": 6379})
        yield c, mock_client


# ============================================================
# 1. String 族新增
# ============================================================
class TestStringExtras:
    async def test_pttl(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pttl = AsyncMock(return_value=1500)
        assert await c.pttl("k") == 1500
        mock.pttl.assert_awaited_once_with("k")

    async def test_persist(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.persist = AsyncMock(return_value=True)
        assert await c.persist("k") is True
        mock.persist.assert_awaited_once_with("k")

    async def test_pexpire(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pexpire = AsyncMock(return_value=True)
        assert await c.pexpire("k", 1000) is True
        mock.pexpire.assert_awaited_once_with("k", 1000)

    async def test_expireat(self, client_with_mock_redis):
        import datetime

        c, mock = client_with_mock_redis
        mock.expireat = AsyncMock(return_value=True)
        when = datetime.datetime(2030, 1, 1)
        assert await c.expireat("k", when) is True
        mock.expireat.assert_awaited_once_with("k", when)

    async def test_type(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.type = AsyncMock(return_value="string")
        assert await c.type("k") == "string"

    async def test_type_decodes_bytes(self, client_with_mock_redis):
        """bytes 类型名也要解成 str。"""
        c, mock = client_with_mock_redis
        mock.type = AsyncMock(return_value=b"hash")
        assert await c.type("k") == "hash"

    async def test_strlen(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.strlen = AsyncMock(return_value=5)
        assert await c.strlen("k") == 5

    async def test_append(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.append = AsyncMock(return_value=8)
        assert await c.append("k", "abc") == 8
        mock.append.assert_awaited_once_with("k", "abc")

    async def test_append_encodes_dict(self, client_with_mock_redis):
        """dict 值走 JSON 编码。"""
        c, mock = client_with_mock_redis
        mock.append = AsyncMock(return_value=10)
        await c.append("k", {"a": 1})
        mock.append.assert_awaited_once_with("k", '{"a": 1}')

    async def test_getrange(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.getrange = AsyncMock(return_value="bcd")
        assert await c.getrange("k", 1, 3) == "bcd"
        mock.getrange.assert_awaited_once_with("k", 1, 3)

    async def test_getrange_empty(self, client_with_mock_redis):
        """key 不存在返回空串（不是 None）。"""
        c, mock = client_with_mock_redis
        mock.getrange = AsyncMock(return_value=None)
        assert await c.getrange("k", 0, -1) == ""

    async def test_setrange(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.setrange = AsyncMock(return_value=10)
        assert await c.setrange("k", 5, "abc") == 10
        mock.setrange.assert_awaited_once_with("k", 5, "abc")

    async def test_getdel(self, client_with_mock_redis):
        """原子取值并删除。"""
        c, mock = client_with_mock_redis
        mock.getdel = AsyncMock(return_value="v")
        assert await c.getdel("k") == "v"
        mock.getdel.assert_awaited_once_with("k")

    async def test_getdel_missing(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.getdel = AsyncMock(return_value=None)
        assert await c.getdel("k") is None

    async def test_getex_with_ttl(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.getex = AsyncMock(return_value="v")
        assert await c.getex("k", ttl=60) == "v"
        mock.getex.assert_awaited_once_with("k", ex=60)

    async def test_getex_without_ttl(self, client_with_mock_redis):
        """不带 ttl 时只取值，不动过期时间。"""
        c, mock = client_with_mock_redis
        mock.getex = AsyncMock(return_value="v")
        assert await c.getex("k") == "v"
        mock.getex.assert_awaited_once_with("k")

    async def test_unlink(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.unlink = AsyncMock(return_value=2)
        assert await c.unlink("k1", "k2") == 2
        mock.unlink.assert_awaited_once_with("k1", "k2")

    async def test_unlink_no_keys(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.unlink = AsyncMock(return_value=0)
        assert await c.unlink() == 0
        mock.unlink.assert_not_called()

    async def test_dbsize(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.dbsize = AsyncMock(return_value=42)
        assert await c.dbsize() == 42

    async def test_rename(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.rename = AsyncMock(return_value=True)
        assert await c.rename("a", "b") is True
        mock.rename.assert_awaited_once_with("a", "b")

    async def test_copy(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.copy = AsyncMock(return_value=True)
        assert await c.copy("a", "b") is True
        mock.copy.assert_awaited_once_with("a", "b")

    async def test_touch(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.touch = AsyncMock(return_value=2)
        assert await c.touch("k1", "k2") == 2
        mock.touch.assert_awaited_once_with("k1", "k2")

    async def test_touch_no_keys(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.touch = AsyncMock(return_value=0)
        assert await c.touch() == 0
        mock.touch.assert_not_called()


# ============================================================
# 2. msetnx
# ============================================================
class TestMsetnx:
    async def test_msetnx_success(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.msetnx = AsyncMock(return_value=True)
        assert await c.msetnx({"a": 1, "b": 2}) is True
        mock.msetnx.assert_awaited_once_with({"a": "1", "b": "2"})

    async def test_msetnx_encodes_dict(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.msetnx = AsyncMock(return_value=True)
        await c.msetnx({"a": {"k": "v"}})
        mock.msetnx.assert_awaited_once_with({"a": '{"k": "v"}'})

    async def test_msetnx_empty(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.msetnx = AsyncMock(return_value=True)
        assert await c.msetnx({}) is True
        mock.msetnx.assert_not_called()


# ============================================================
# 3. Hash 族新增
# ============================================================
class TestHashExtras:
    async def test_hlen(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hlen = AsyncMock(return_value=3)
        assert await c.hlen("h") == 3

    async def test_hexists(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hexists = AsyncMock(return_value=1)
        assert await c.hexists("h", "f") is True
        mock.hexists.assert_awaited_once_with("h", "f")

    async def test_hmget(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hmget = AsyncMock(return_value=["a", None, "c"])
        result = await c.hmget("h", "f1", "f2", "f3")
        assert result == ["a", None, "c"]
        mock.hmget.assert_awaited_once_with("h", ["f1", "f2", "f3"])

    async def test_hmget_no_fields(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hmget = AsyncMock(return_value=[])
        assert await c.hmget("h") == []
        mock.hmget.assert_not_called()

    async def test_hkeys(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hkeys = AsyncMock(return_value=["a", "b"])
        assert await c.hkeys("h") == ["a", "b"]

    async def test_hvals(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hvals = AsyncMock(return_value=["1", "2"])
        assert await c.hvals("h") == ["1", "2"]

    async def test_hincrby(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hincrby = AsyncMock(return_value=6)
        assert await c.hincrby("h", "f", 5) == 6
        mock.hincrby.assert_awaited_once_with("h", "f", 5)

    async def test_hincrbyfloat(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hincrbyfloat = AsyncMock(return_value=1.5)
        assert await c.hincrbyfloat("h", "f", 1.5) == 1.5
        mock.hincrbyfloat.assert_awaited_once_with("h", "f", 1.5)

    async def test_hsetnx(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hsetnx = AsyncMock(return_value=1)
        assert await c.hsetnx("h", "f", "v") is True
        mock.hsetnx.assert_awaited_once_with("h", "f", "v")

    async def test_hset_mapping(self, client_with_mock_redis):
        """批量写 hash 字段。"""
        c, mock = client_with_mock_redis
        mock.hset = AsyncMock(return_value=2)
        assert await c.hset("h", mapping={"f1": "v1", "f2": "v2"}) == 2
        mock.hset.assert_awaited_once_with("h", mapping={"f1": "v1", "f2": "v2"})

    async def test_hset_mapping_encodes_dict_value(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hset = AsyncMock(return_value=1)
        await c.hset("h", mapping={"f": {"k": "v"}})
        mock.hset.assert_awaited_once_with("h", mapping={"f": '{"k": "v"}'})

    async def test_hset_mapping_plus_field(self, client_with_mock_redis):
        """mapping + 单字段混用。"""
        c, mock = client_with_mock_redis
        mock.hset = AsyncMock(return_value=2)
        await c.hset("h", "f3", "v3", mapping={"f1": "v1"})
        mock.hset.assert_awaited_once_with(
            "h", mapping={"f1": "v1", "f3": "v3"}
        )

    async def test_hset_requires_field_or_mapping(self, client_with_mock_redis):
        """既没 field 也没 mapping —— 抛 ValueError。"""
        c, mock = client_with_mock_redis
        mock.hset = AsyncMock(return_value=0)
        with pytest.raises(ValueError, match="field 或 mapping"):
            await c.hset("h")
        mock.hset.assert_not_called()

    async def test_hdel_multiple_fields(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hdel = AsyncMock(return_value=2)
        assert await c.hdel("h", "f1", "f2") == 2
        mock.hdel.assert_awaited_once_with("h", "f1", "f2")

    async def test_hdel_no_fields(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.hdel = AsyncMock(return_value=0)
        assert await c.hdel("h") == 0
        mock.hdel.assert_not_called()

    async def test_hgetall_decodes_bytes(self, client_with_mock_redis):
        """回归：bytes key/value 不能变成 "b'k'"。"""
        c, mock = client_with_mock_redis
        mock.hgetall = AsyncMock(return_value={b"f1": b"v1", "f2": "v2"})
        result = await c.hgetall("h")
        assert result == {"f1": "v1", "f2": "v2"}

    async def test_smembers_decodes_bytes(self, client_with_mock_redis):
        """回归：bytes 成员不能变成 "b'm'"。"""
        c, mock = client_with_mock_redis
        mock.smembers = AsyncMock(return_value={b"m1", "m2"})
        assert await c.smembers("s") == {"m1", "m2"}


# ============================================================
# 4. Set 族新增
# ============================================================
class TestSetExtras:
    async def test_srandmember_single(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.srandmember = AsyncMock(return_value="m1")
        assert await c.srandmember("s") == "m1"
        mock.srandmember.assert_awaited_once_with("s")

    async def test_srandmember_with_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.srandmember = AsyncMock(return_value=["m1", "m2"])
        assert await c.srandmember("s", 2) == ["m1", "m2"]
        mock.srandmember.assert_awaited_once_with("s", 2)

    async def test_srandmember_empty_set(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.srandmember = AsyncMock(return_value=None)
        assert await c.srandmember("s") is None

    async def test_spop_single(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.spop = AsyncMock(return_value="m1")
        assert await c.spop("s") == "m1"
        mock.spop.assert_awaited_once_with("s")

    async def test_spop_with_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.spop = AsyncMock(return_value=["m1", "m2"])
        assert await c.spop("s", 2) == ["m1", "m2"]

    async def test_smove(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.smove = AsyncMock(return_value=True)
        assert await c.smove("src", "dst", "m") is True
        mock.smove.assert_awaited_once_with("src", "dst", "m")

    async def test_sintercard(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sintercard = AsyncMock(return_value=3)
        assert await c.sintercard(["s1", "s2"]) == 3
        # redis-py 签名是 (numkeys, keys, limit)
        mock.sintercard.assert_awaited_once_with(2, ["s1", "s2"], 0)

    async def test_sintercard_no_keys(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.sintercard = AsyncMock(return_value=0)
        assert await c.sintercard([]) == 0
        mock.sintercard.assert_not_called()


# ============================================================
# 5. List 族新增
# ============================================================
class TestListExtras:
    async def test_lpos_single(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpos = AsyncMock(return_value=2)
        assert await c.lpos("l", "v") == 2
        mock.lpos.assert_awaited_once_with("l", "v", rank=1)

    async def test_lpos_with_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpos = AsyncMock(return_value=[0, 2])
        assert await c.lpos("l", "v", count=0) == [0, 2]
        mock.lpos.assert_awaited_once_with("l", "v", rank=1, count=0, maxlen=None)

    async def test_lpos_not_found(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.lpos = AsyncMock(return_value=None)
        assert await c.lpos("l", "v") is None

    async def test_linsert(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.linsert = AsyncMock(return_value=3)
        assert await c.linsert("l", "before", "ref", "new") == 3
        mock.linsert.assert_awaited_once_with("l", "before", "ref", "new")

    async def test_linsert_invalid_where(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.linsert = AsyncMock(return_value=1)
        with pytest.raises(ValueError, match="before"):
            await c.linsert("l", "middle", "ref", "new")
        mock.linsert.assert_not_called()


# ============================================================
# 6. ZSet 族新增
# ============================================================
class TestZSetExtras:
    async def test_zmscore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zmscore = AsyncMock(return_value=[1.0, None, 3.0])
        assert await c.zmscore("z", "a", "b", "c") == [1.0, None, 3.0]
        mock.zmscore.assert_awaited_once_with("z", ["a", "b", "c"])

    async def test_zmscore_no_members(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zmscore = AsyncMock(return_value=[])
        assert await c.zmscore("z") == []
        mock.zmscore.assert_not_called()

    async def test_zrandmember_single(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrandmember = AsyncMock(return_value="m1")
        assert await c.zrandmember("z") == "m1"
        mock.zrandmember.assert_awaited_once_with("z")

    async def test_zrandmember_with_count(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zrandmember = AsyncMock(return_value=["m1", "m2"])
        assert await c.zrandmember("z", 2) == ["m1", "m2"]

    async def test_zunion(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zunion = AsyncMock(return_value=["a", "b"])
        assert await c.zunion(["z1", "z2"]) == ["a", "b"]
        mock.zunion.assert_awaited_once_with(["z1", "z2"], aggregate=None, withscores=False)

    async def test_zunion_with_aggregate(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zunion = AsyncMock(return_value=[("a", 3.0)])
        assert await c.zunion(["z1", "z2"], aggregate="MAX", withscores=True) == [("a", 3.0)]
        mock.zunion.assert_awaited_once_with(["z1", "z2"], aggregate="MAX", withscores=True)

    async def test_zinter(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zinter = AsyncMock(return_value=["a"])
        assert await c.zinter(["z1", "z2"]) == ["a"]

    async def test_zdiff(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zdiff = AsyncMock(return_value=["b"])
        assert await c.zdiff(["z1", "z2"]) == ["b"]

    async def test_zunionstore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zunionstore = AsyncMock(return_value=3)
        assert await c.zunionstore("dst", ["z1", "z2"]) == 3
        mock.zunionstore.assert_awaited_once_with("dst", ["z1", "z2"], aggregate=None)

    async def test_zinterstore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zinterstore = AsyncMock(return_value=1)
        assert await c.zinterstore("dst", ["z1", "z2"]) == 1

    async def test_zdiffstore(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zdiffstore = AsyncMock(return_value=2)
        assert await c.zdiffstore("dst", ["z1", "z2"]) == 2
        mock.zdiffstore.assert_awaited_once_with("dst", ["z1", "z2"])

    async def test_zstore_no_keys(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.zunionstore = AsyncMock(return_value=0)
        mock.zinterstore = AsyncMock(return_value=0)
        mock.zdiffstore = AsyncMock(return_value=0)
        assert await c.zunionstore("dst", []) == 0
        assert await c.zinterstore("dst", []) == 0
        assert await c.zdiffstore("dst", []) == 0
        mock.zunionstore.assert_not_called()
        mock.zinterstore.assert_not_called()
        mock.zdiffstore.assert_not_called()


# ============================================================
# 7. HyperLogLog
# ============================================================
class TestHyperLogLog:
    async def test_pfadd(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pfadd = AsyncMock(return_value=1)
        assert await c.pfadd("hll", "u1", "u2") == 1
        mock.pfadd.assert_awaited_once_with("hll", "u1", "u2")

    async def test_pfadd_no_values(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pfadd = AsyncMock(return_value=0)
        assert await c.pfadd("hll") == 0
        mock.pfadd.assert_not_called()

    async def test_pfcount(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pfcount = AsyncMock(return_value=1000)
        assert await c.pfcount("hll") == 1000
        mock.pfcount.assert_awaited_once_with("hll")

    async def test_pfcount_multiple(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pfcount = AsyncMock(return_value=2000)
        assert await c.pfcount("h1", "h2") == 2000
        mock.pfcount.assert_awaited_once_with("h1", "h2")

    async def test_pfmerge(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pfmerge = AsyncMock(return_value=True)
        assert await c.pfmerge("dst", "h1", "h2") is True
        mock.pfmerge.assert_awaited_once_with("dst", "h1", "h2")

    async def test_pfmerge_no_sources(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.pfmerge = AsyncMock(return_value=False)
        assert await c.pfmerge("dst") is False
        mock.pfmerge.assert_not_called()


# ============================================================
# 8. 运维 / 诊断
# ============================================================
class TestOpsCommands:
    async def test_memory_usage(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.memory_usage = AsyncMock(return_value=1024)
        assert await c.memory_usage("k") == 1024
        mock.memory_usage.assert_awaited_once_with("k", 5)

    async def test_memory_usage_missing_key(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.memory_usage = AsyncMock(return_value=None)
        assert await c.memory_usage("k") is None

    async def test_info(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.info = AsyncMock(return_value={"redis_version": "7.4.0"})
        result = await c.info()
        assert result["redis_version"] == "7.4.0"
        mock.info.assert_awaited_once_with(None)

    async def test_info_with_section(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.info = AsyncMock(return_value={"used_memory": "1M"})
        result = await c.info("memory")
        assert result["used_memory"] == "1M"
        mock.info.assert_awaited_once_with("memory")


# ============================================================
# 9. 既有缺陷修复：set_nx 自身的参数拼装
# ============================================================
class TestSetNxFix:
    """set_nx 此前零直接测试——它承载分布式锁的原子性语义。"""

    async def test_set_nx_with_ttl(self, client_with_mock_redis):
        """SET key value NX EX ttl —— 单命令原子。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        assert await c.set_nx("k", "v", ttl=30) is True
        mock.set.assert_awaited_once_with("k", "v", nx=True, ex=30)

    async def test_set_nx_without_ttl(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=False)
        assert await c.set_nx("k", "v") is False
        mock.set.assert_awaited_once_with("k", "v", nx=True)

    async def test_set_nx_encodes_dict(self, client_with_mock_redis):
        """dict 值走 JSON 编码（与 set 一致）。"""
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        await c.set_nx("k", {"a": 1})
        mock.set.assert_awaited_once_with("k", '{"a": 1}', nx=True)

    async def test_set_nx_encodes_non_str(self, client_with_mock_redis):
        c, mock = client_with_mock_redis
        mock.set = AsyncMock(return_value=True)
        await c.set_nx("k", 42)
        mock.set.assert_awaited_once_with("k", "42", nx=True)
