#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra redis **List / Set / ZSet 新增方法**的真 Redis 集成测试

背景（20261021）：
    `tests/test_redis_lists.py` 用 mock 覆盖了新增方法，但 mock 验证不了：
      1. 命令参数在真实服务端是否被接受
         （关键：``lmove`` 传的是小写 ``"left"/"right"``，
          redis-py 默认是大写 ``"LEFT"/"RIGHT"``——需服务端确认大小写不敏感）
      2. 返回值类型（bytes vs str）——连接池开了 ``decode_responses=True``
      3. 空值 / 边界行为（``None`` vs ``[]``、``count=0``、``count<0``）

    本文件用**真 Redis** 跑一遍，锁住这些 mock 测不到的行为。

前置条件（手动触发）：
    方式 A（环境变量）：
        export REDIS_INTEGRATION_HOST=127.0.0.1
        export REDIS_INTEGRATION_PORT=6379
        pytest -m integration tests/test_redis_lists_integration.py

    方式 B（.env.test 文件——自动加载）：
        REDIS_INTEGRATION_HOST=127.0.0.1
        REDIS_INTEGRATION_PORT=6379
        REDIS_INTEGRATION_PASSWORD=   # 无密码的本地 redis 可留空
        然后跑：
            pytest -m integration tests/test_redis_lists_integration.py

默认 ``pytest tests/`` 会跳过（未设 REDIS_INTEGRATION_HOST）。

与 ``test_redis_integration.py`` 的差异：
    那个文件要求 PASSWORD 非空（防连错服务）；本文件**允许无密码**，
    因为本地/容器化的 redis 常常没有密码，用它来跑新增方法的真实验证。

所有 key 用 ``infra_test_lists:<uuid>:`` 前缀隔离，测试结束清理。
"""

from __future__ import annotations

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
        "db": int(os.getenv("REDIS_INTEGRATION_DB", "0")),
    }


_INTEGRATION_CONFIG = _env_config()
if _INTEGRATION_CONFIG is None:
    pytest.skip(
        "REDIS_INTEGRATION_HOST 未设——跳过 List/Set/ZSet 集成测试",
        allow_module_level=True,
    )


def _redis_reachable(config: Dict[str, Any]) -> Optional[str]:
    """同步探活——连不上返回原因说明，连上返回 None。

    模块级做一次：连不上就**整个文件 skip**（与 test_redis_integration.py
    的模块级 skip 行为一致），避免每个用例都产出一条 skip 噪音。
    """
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
        f"（{_UNREACHABLE}）——跳过 List/Set/ZSet 集成测试",
        allow_module_level=True,
    )


@pytest.fixture
async def real_redis():
    """真 RedisClient——测试结束清理本次用到的所有 key + close 连接。

    连不上（服务未起 / 密码错 / 网络不通）时 **skip 而不是 fail**——
    集成测试依赖外部服务，不该让默认的全量 ``pytest tests/`` 挂掉。
    """
    from infra.redis import RedisClient

    prefix = f"infra_test_lists:{uuid.uuid4()}"
    client = RedisClient(config=_INTEGRATION_CONFIG)
    try:
        # 探活——连不上就 skip
        try:
            await client._get_client().ping()
        except Exception as e:
            pytest.skip(
                f"连不上 redis {_INTEGRATION_CONFIG['host']}:"
                f"{_INTEGRATION_CONFIG['port']}（{type(e).__name__}）——跳过集成测试"
            )
        yield client, prefix
    finally:
        try:
            keys = await client._get_client().keys(f"{prefix}:*")
            if keys:
                await client._get_client().delete(*keys)
        except Exception:
            pass
        await client.close()


# ============================================================
# 1. Set 运算——真服务端
# ============================================================
class TestSetOperationsIntegration:
    async def test_sinter_sunion_sdiff_real(self, real_redis):
        """交 / 并 / 差集——真 Redis 计算结果。"""
        client, prefix = real_redis
        s1, s2 = f"{prefix}:s1", f"{prefix}:s2"
        await client._get_client().sadd(s1, "a", "b", "c")
        await client._get_client().sadd(s2, "b", "c", "d")

        assert await client.sinter(s1, s2) == {"b", "c"}
        assert await client.sunion(s1, s2) == {"a", "b", "c", "d"}
        assert await client.sdiff(s1, s2) == {"a"}

    async def test_set_ops_return_str_not_bytes(self, real_redis):
        """返回值是 str（不是 bytes）——连接池 decode_responses=True。"""
        client, prefix = real_redis
        s1 = f"{prefix}:s1"
        await client._get_client().sadd(s1, "x", "y")
        result = await client.sinter(s1)
        assert all(isinstance(v, str) for v in result), f"应返回 str，实际 {result}"

    async def test_set_ops_no_keys_returns_empty(self, real_redis):
        """无参调用返回空集（不报错、不建 key）。"""
        client, _ = real_redis
        assert await client.sinter() == set()
        assert await client.sunion() == set()
        assert await client.sdiff() == set()

    async def test_set_store_variants_real(self, real_redis):
        """sdiffstore / sinterstore / sunionstore——真写入 + 返回数量。"""
        client, prefix = real_redis
        s1, s2 = f"{prefix}:s1", f"{prefix}:s2"
        await client._get_client().sadd(s1, "a", "b", "c")
        await client._get_client().sadd(s2, "b", "c", "d")

        d = f"{prefix}:d"
        assert await client.sdiffstore(d, s1, s2) == 1
        assert await client._get_client().smembers(d) == {"a"}

        i = f"{prefix}:i"
        assert await client.sinterstore(i, s1, s2) == 2
        assert await client._get_client().smembers(i) == {"b", "c"}

        u = f"{prefix}:u"
        assert await client.sunionstore(u, s1, s2) == 4
        assert await client._get_client().smembers(u) == {"a", "b", "c", "d"}

    async def test_scard_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:s"
        await client._get_client().sadd(key, "a", "b")
        assert await client.scard(key) == 2
        assert await client.scard(f"{prefix}:nonexistent") == 0


# ============================================================
# 2. List 基本操作——真服务端
# ============================================================
class TestListOperationsIntegration:
    async def test_lpush_rpush_order(self, real_redis):
        """lpush 左推 / rpush 右推——真实顺序。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        assert await client.rpush(key, "a", "b", "c") == 3
        assert await client.lrange(key, 0, -1) == ["a", "b", "c"]

        key2 = f"{prefix}:l2"
        assert await client.lpush(key2, "x", "y") == 2
        # lpush 后插入的在**前面**
        assert await client.lrange(key2, 0, -1) == ["y", "x"]

    async def test_lrange_variants(self, real_redis):
        """lrange 区间 / 最近 N 条 / 空 key。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "a", "b", "c", "d")
        assert await client.lrange(key, 0, 1) == ["a", "b"]
        assert await client.lrange(key, -2, -1) == ["c", "d"]
        assert await client.lrange(key, 0, -1) == ["a", "b", "c", "d"]
        assert await client.lrange(f"{prefix}:nonexistent", 0, -1) == []

    async def test_llen_lindex(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "a", "b", "c")
        assert await client.llen(key) == 3
        assert await client.lindex(key, 1) == "b"
        assert await client.lindex(key, -1) == "c"
        assert await client.lindex(key, 99) is None

    async def test_lpop_rpop_single_and_count(self, real_redis):
        """lpop / rpop——单个返回 str，count 返回 list。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "a", "b", "c")

        v = await client.lpop(key)
        assert v == "a"
        assert isinstance(v, str)

        vs = await client.lpop(key, 2)
        assert vs == ["b", "c"]
        assert all(isinstance(x, str) for x in vs)

        key2 = f"{prefix}:l2"
        await client.rpush(key2, "x", "y", "z")
        assert await client.rpop(key2) == "z"
        assert await client.rpop(key2, 2) == ["y", "x"]

    async def test_lpop_empty_returns_none(self, real_redis):
        """空 key 弹出返回 None（不是 [] / 不报错）。"""
        client, prefix = real_redis
        assert await client.lpop(f"{prefix}:nonexistent") is None
        assert await client.rpop(f"{prefix}:nonexistent") is None
        assert await client.lpop(f"{prefix}:nonexistent", 0) == []
        assert await client.rpop(f"{prefix}:nonexistent", 0) == []

    async def test_lset_lrem_ltrim(self, real_redis):
        """lset / lrem(count 语义) / ltrim。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "a", "b", "a", "c")
        assert await client.lset(key, 0, "z") is True
        assert await client.lindex(key, 0) == "z"

        # count>0：从头往尾删 N 个
        assert await client.lrem(key, 1, "a") == 1
        assert await client.lrange(key, 0, -1) == ["z", "b", "c"]

        # count=0：删全部匹配
        key2 = f"{prefix}:l2"
        await client.rpush(key2, "a", "x", "a", "a", "y")
        assert await client.lrem(key2, 0, "a") == 3
        assert await client.lrange(key2, 0, -1) == ["x", "y"]

        # count<0：从尾往头删 |count| 个
        key3 = f"{prefix}:l3"
        await client.rpush(key3, "a", "b", "a", "b", "a")
        assert await client.lrem(key3, -2, "a") == 2
        assert await client.lrange(key3, 0, -1) == ["a", "b", "b"]

        # ltrim：只保留区间
        assert await client.ltrim(key3, 0, 0) is True
        assert await client.lrange(key3, 0, -1) == ["a"]

    async def test_lpushx_rpushx(self, real_redis):
        """lpushx / rpushx——key 不存在时不创建。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        assert await client.lpushx(key, "a") == 0
        assert await client.llen(key) == 0, "lpushx 不该创建 key"
        assert await client.rpushx(key, "a") == 0
        assert await client.llen(key) == 0, "rpushx 不该创建 key"

        await client.rpush(key, "first")
        assert await client.lpushx(key, "second") == 2
        assert await client.rpushx(key, "third") == 3
        assert await client.lrange(key, 0, -1) == ["second", "first", "third"]

    async def test_list_value_encoding(self, real_redis):
        """dict / list 元素编码为 JSON；str 原样；int 转 str。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, {"k": "v"}, [1, 2], "plain", 42)
        vals = await client.lrange(key, 0, -1)
        assert vals[0] == '{"k": "v"}'
        assert vals[1] == "[1, 2]"
        assert vals[2] == "plain"
        assert vals[3] == "42"

    async def test_list_value_encoding_unicode(self, real_redis):
        """中文元素——JSON 不转义（ensure_ascii=False）。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, {"city": "上海"})
        vals = await client.lrange(key, 0, -1)
        assert "上海" in vals[0]
        assert "\\u" not in vals[0], "不应出现 unicode 转义"


# ============================================================
# 3. 阻塞弹出——真服务端
# ============================================================
class TestBlockingListIntegration:
    async def test_blpop_timeout_returns_none(self, real_redis):
        """空 key 阻塞超时——返回 None（不挂死）。"""
        client, prefix = real_redis
        result = await client.blpop(f"{prefix}:nonexistent", timeout=1)
        assert result is None

    async def test_blpop_gets_value(self, real_redis):
        """有值时 blpop 立即返回 (key, value)。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "task1")
        result = await client.blpop(key, timeout=2)
        assert result is not None
        k, v = result
        assert v == "task1"
        assert isinstance(k, str) and isinstance(v, str)

    async def test_brpop_gets_value(self, real_redis):
        """brpop 从尾部弹。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "task1", "task2")
        result = await client.brpop(key, timeout=2)
        assert result is not None
        _k, v = result
        assert v == "task2"


# ============================================================
# 4. 原子移动——真服务端（重点验证 lmove 参数大小写）
# ============================================================
class TestAtomicMoveIntegration:
    async def test_rpoplpush_real(self, real_redis):
        """rpoplpush——从源尾部弹、压入目标头部。"""
        client, prefix = real_redis
        src, dst = f"{prefix}:src", f"{prefix}:dst"
        await client.rpush(src, "job1", "job2")
        moved = await client.rpoplpush(src, dst)
        assert moved == "job2"
        assert await client.lrange(src, 0, -1) == ["job1"]
        assert await client.lrange(dst, 0, -1) == ["job2"]

    async def test_lmove_lowercase_direction_accepted(self, real_redis):
        """**lmove 传小写 "left"/"right" 服务端必须接受**。

        redis-py 的默认参数是大写 ``"LEFT"/"RIGHT"``；本项目用小写。
        Redis 服务端解析方向用 ``strcasecmp``（大小写不敏感）——本用例锁住
        这个行为，若服务端将来改成大小写敏感就会失败。
        """
        client, prefix = real_redis
        src, dst = f"{prefix}:src", f"{prefix}:dst"
        await client.rpush(src, "m1", "m2")

        # right → left：尾部弹出、头部压入
        moved = await client.lmove(src, dst, "right", "left")
        assert moved == "m2"
        assert await client.lrange(dst, 0, -1) == ["m2"]

    async def test_lmove_left_right(self, real_redis):
        """lmove left→right：头部弹出、尾部压入。"""
        client, prefix = real_redis
        src, dst = f"{prefix}:src", f"{prefix}:dst"
        await client.rpush(src, "n1", "n2")
        moved = await client.lmove(src, dst, "left", "right")
        assert moved == "n1"
        assert await client.lrange(dst, 0, -1) == ["n1"]

    async def test_lmove_defaults(self, real_redis):
        """不传方向时用默认 left/left。"""
        client, prefix = real_redis
        src, dst = f"{prefix}:src", f"{prefix}:dst"
        await client.rpush(src, "d1", "d2")
        moved = await client.lmove(src, dst)
        assert moved is not None
        assert moved in {"d1", "d2"}

    async def test_rpoplpush_empty_source(self, real_redis):
        """源为空——返回 None。"""
        client, prefix = real_redis
        result = await client.rpoplpush(f"{prefix}:nonexistent", f"{prefix}:dst")
        assert result is None


# ============================================================
# 5. ZSet 排名——真服务端
# ============================================================
class TestZSetRankIntegration:
    async def test_zrank_zrevrank(self, real_redis):
        """zrank 升序 / zrevrank 降序。"""
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"low": 1.0, "mid": 5.0, "high": 10.0})
        assert await client.zrank(key, "mid") == 1
        assert await client.zrevrank(key, "mid") == 1
        assert await client.zrank(key, "high") == 2
        assert await client.zrevrank(key, "high") == 0

    async def test_zrank_missing_member_returns_none(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"a": 1.0})
        assert await client.zrank(key, "nope") is None
        assert await client.zrevrank(key, "nope") is None

    async def test_zrevrange_descending(self, real_redis):
        """zrevrange 按分数降序。"""
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"low": 1.0, "mid": 5.0, "high": 10.0})
        assert await client.zrevrange(key, 0, -1) == ["high", "mid", "low"]
        assert await client.zrevrange(key, 0, 1) == ["high", "mid"]

    async def test_zrevrange_withscores(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"low": 1.0, "high": 10.0})
        result = await client.zrevrange(key, 0, 0, withscores=True)
        assert result == [("high", 10.0)]

    async def test_zrevrangebyscore(self, real_redis):
        """zrevrangebyscore 区间 + 分页。"""
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"a": 1.0, "b": 5.0, "c": 10.0})
        assert await client.zrevrangebyscore(key, 10.0, 1.0) == ["c", "b", "a"]
        # 分页：offset/count
        assert await client.zrevrangebyscore(key, 10.0, 1.0, offset=0, count=2) == ["c", "b"]
        assert await client.zrevrangebyscore(key, 10.0, 1.0, offset=1, count=1) == ["b"]

    async def test_zremrangebyrank(self, real_redis):
        """按排名区间删除。"""
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"a": 1.0, "b": 5.0, "c": 10.0})
        # 删最低分的 1 个（升序排名 0）
        assert await client.zremrangebyrank(key, 0, 0) == 1
        assert await client.zrange(key, 0, -1) == ["b", "c"]

    async def test_zadd_flags_real(self, real_redis):
        """zadd 的 nx / gt / lt / ch 标志——真服务端行为。

        注意：ZADD 返回的是**新增**元素数（更新已有成员返回 0）；
        要看「实际变更数」必须加 ch=True。
        """
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"m": 5.0})

        # nx：已存在则不更新
        assert await client.zadd(key, {"m": 100.0}, nx=True) == 0
        assert await client.zscore(key, "m") == 5.0

        # gt：只有更大才更新（Redis 6.2+）
        assert await client.zadd(key, {"m": 10.0}, gt=True) == 0
        assert await client.zscore(key, "m") == 10.0
        # 更小的分数不更新
        assert await client.zadd(key, {"m": 1.0}, gt=True) == 0
        assert await client.zscore(key, "m") == 10.0

        # lt：只有更小才更新
        assert await client.zadd(key, {"m": 3.0}, lt=True) == 0
        assert await client.zscore(key, "m") == 3.0

        # ch：返回实际变更数
        assert await client.zadd(key, {"m": 3.0}, ch=True) == 0
        assert await client.zadd(key, {"m": 4.0}, ch=True) == 1

    async def test_zmscore_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"a": 1.0, "b": 2.0})
        assert await client.zmscore(key, "a", "b", "nope") == [1.0, 2.0, None]

    async def test_zset_set_operations_real(self, real_redis):
        """zunion / zinter / zdiff + 落盘三件套。"""
        client, prefix = real_redis
        z1, z2, dest = f"{prefix}:z1", f"{prefix}:z2", f"{prefix}:zd"
        await client.zadd(z1, {"a": 1.0, "b": 2.0})
        await client.zadd(z2, {"b": 3.0, "c": 4.0})

        assert sorted(await client.zunion([z1, z2])) == ["a", "b", "c"]
        assert await client.zinter([z1, z2]) == ["b"]
        assert await client.zdiff([z1, z2]) == ["a"]

        assert await client.zunionstore(dest, [z1, z2]) == 3
        assert await client.zinterstore(dest, [z1, z2]) == 1
        assert await client.zdiffstore(dest, [z1, z2]) == 1

    async def test_zrandmember_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:z"
        await client.zadd(key, {"a": 1.0, "b": 2.0, "c": 3.0})
        assert (await client.zrandmember(key)) in {"a", "b", "c"}
        assert len(await client.zrandmember(key, 2)) == 2


# ============================================================
# 6. String / Hash / Set / List / HLL 新增命令——真 Redis
# ============================================================
class TestNewCommandsIntegration:
    async def test_string_extras_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:s"
        await client.set(key, "hello")
        assert await client.strlen(key) == 5
        assert await client.type(key) == "string"
        assert await client.append(key, " world") == 11
        assert await client.getrange(key, 0, 4) == "hello"
        assert await client.setrange(key, 0, "HELLO") == 11
        assert await client.getrange(key, 0, 4) == "HELLO"

    async def test_getdel_getex_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:s"
        await client.set(key, "v")
        assert await client.getdel(key) == "v"
        assert await client.exists(key) is False

        await client.set(key, "v", ttl=100)
        assert await client.getex(key, ttl=200) == "v"
        assert (await client.ttl(key)) > 100
        assert await client.persist(key) is True
        assert (await client.ttl(key)) == -1

    async def test_unlink_rename_copy_touch_real(self, real_redis):
        client, prefix = real_redis
        k1, k2, k3 = f"{prefix}:k1", f"{prefix}:k2", f"{prefix}:k3"
        await client.set(k1, "v")
        assert await client.copy(k1, k2) is True
        assert await client.exists(k2) is True
        assert await client.rename(k2, k3) is True
        assert await client.exists(k3) is True
        assert await client.touch(k1, k3) == 2
        assert await client.unlink(k1) == 1
        assert await client.exists(k1) is False

    async def test_msetnx_real(self, real_redis):
        """msetnx——全部不存在才写（原子）。"""
        client, prefix = real_redis
        m1, m2, m3 = f"{prefix}:m1", f"{prefix}:m2", f"{prefix}:m3"
        assert await client.msetnx({m1: "a", m2: "b"}) is True
        assert await client.get(m1) == "a"
        # m1 已存在 → 整组不写
        assert await client.msetnx({m1: "x", m3: "y"}) is False
        assert await client.exists(m3) is False

    async def test_hash_extras_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:h"
        await client.hset(key, "f1", "v1")
        await client.hset(key, mapping={"f2": "v2", "f3": "v3"})
        assert await client.hlen(key) == 3
        assert await client.hexists(key, "f1") is True
        assert await client.hexists(key, "nope") is False
        assert sorted(await client.hkeys(key)) == ["f1", "f2", "f3"]
        assert sorted(await client.hvals(key)) == ["v1", "v2", "v3"]
        assert await client.hmget(key, "f1", "nope") == ["v1", None]
        assert await client.hincrby(key, "cnt", 5) == 5
        assert await client.hincrbyfloat(key, "price", 1.5) == 1.5
        assert await client.hsetnx(key, "new", "v") is True
        assert await client.hsetnx(key, "new", "v2") is False
        assert await client.hdel(key, "f1", "f2") == 2

    async def test_set_extras_real(self, real_redis):
        client, prefix = real_redis
        key, key2 = f"{prefix}:s", f"{prefix}:s2"
        await client.sadd(key, "a", "b", "c")
        assert (await client.srandmember(key)) in {"a", "b", "c"}
        assert len(await client.srandmember(key, 2)) == 2
        assert await client.scard(key) == 3  # srandmember 不删除

        # 先测 smove——spop 是随机的，可能把待移动的成员删掉
        assert await client.smove(key, key2, "a") is True
        assert await client.sismember(key, "a") is False
        assert await client.sismember(key2, "a") is True

        assert isinstance(await client.spop(key), str)
        assert await client.scard(key) == 1  # spop 删除了

    async def test_sintercard_real(self, real_redis):
        client, prefix = real_redis
        s1, s2 = f"{prefix}:si1", f"{prefix}:si2"
        await client.sadd(s1, "x", "y", "z")
        await client.sadd(s2, "y", "z", "w")
        assert await client.sintercard([s1, s2]) == 2
        assert await client.sintercard([s1, s2], limit=1) == 1

    async def test_lpos_linsert_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:l"
        await client.rpush(key, "a", "b", "a", "c")
        assert await client.lpos(key, "a") == 0
        assert await client.lpos(key, "a", rank=2) == 2
        assert await client.lpos(key, "zzz") is None
        assert await client.lpos(key, "a", count=0) == [0, 2]
        assert await client.linsert(key, "before", "c", "new") == 5
        assert await client.lrange(key, 0, -1) == ["a", "b", "a", "new", "c"]
        assert await client.linsert(key, "before", "nope", "x") == -1

    async def test_hyperloglog_real(self, real_redis):
        """HyperLogLog——UV 统计。"""
        client, prefix = real_redis
        hll, hll2, merged = f"{prefix}:hll", f"{prefix}:hll2", f"{prefix}:hllm"
        assert await client.pfadd(hll, "u1", "u2", "u3") == 1
        assert await client.pfcount(hll) == 3
        await client.pfadd(hll2, "u3", "u4")
        assert await client.pfcount(hll, hll2) == 4
        assert await client.pfmerge(merged, hll, hll2) is True
        assert await client.pfcount(merged) == 4

    async def test_memory_usage_info_real(self, real_redis):
        client, prefix = real_redis
        key = f"{prefix}:m"
        await client.set(key, "x" * 100)
        usage = await client.memory_usage(key)
        assert usage is not None and usage > 0
        assert await client.memory_usage(f"{prefix}:nope") is None
        info = await client.info("memory")
        assert isinstance(info, dict) and len(info) > 0



# ============================================================
# 6. pipeline 事务下使用新方法
# ============================================================
class TestPipelineIntegration:
    async def test_pipeline_with_new_methods(self, real_redis):
        """pipeline 里用新方法——真 Redis 事务执行。"""
        client, prefix = real_redis
        key = f"{prefix}:l"
        async with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, "a")
            pipe.lpush(key, "b")
            pipe.llen(key)
            result = await pipe.execute()
        assert result[2] == 2
        assert await client.lrange(key, 0, -1) == ["b", "a"]
