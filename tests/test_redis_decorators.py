#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.redis.decorators 深度测试

infra 是下游项目的基础设施类库——cache 装饰器是业务常用的缓存工具。
覆盖 @cache 的所有路径：命中、未命中、序列化、未初始化、key 模板。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.redis.decorators import cache


# ============================================================
# Fixture：mock get_redis 返 fake client
# ============================================================
@pytest.fixture
def fake_redis_client():
    """mock get_redis() 返 fake client——不连真 redis。"""
    fake = MagicMock()
    fake.get = AsyncMock(return_value=None)
    fake.set = AsyncMock(return_value=True)
    with patch("infra.redis.decorators.get_redis", return_value=fake):
        yield fake


# ============================================================
# 1. 缓存未初始化
# ============================================================
class TestCacheRedisUninitialized:
    async def test_uninitialized_redis_executes_func_directly(self):
        """Redis 未初始化——直接执行原函数，不调 redis。"""
        with patch("infra.redis.decorators.get_redis", return_value=None):
            @cache(ttl=60)
            async def my_func():
                return {"data": "value"}

            result = await my_func()
            assert result == {"data": "value"}

    async def test_uninitialized_redis_passes_args(self):
        """Redis 未初始化——args/kwargs 透传给原函数。"""
        with patch("infra.redis.decorators.get_redis", return_value=None):
            @cache()
            async def add(a, b, c=10):
                return a + b + c

            result = await add(1, 2, c=3)
            assert result == 6


# ============================================================
# 2. 缓存命中
# ============================================================
class TestCacheHit:
    async def test_hit_returns_cached_dict(self, fake_redis_client):
        """缓存命中——JSON 反序列化返 dict。"""
        fake_redis_client.get = AsyncMock(return_value=json.dumps({"a": 1}))

        @cache()
        async def my_func():
            return {"should": "not call"}

        result = await my_func()
        assert result == {"a": 1}

    async def test_hit_returns_cached_list(self, fake_redis_client):
        fake_redis_client.get = AsyncMock(return_value=json.dumps([1, 2, 3]))

        @cache()
        async def my_func():
            return []

        result = await my_func()
        assert result == [1, 2, 3]

    async def test_hit_returns_cached_string(self, fake_redis_client):
        """缓存命中——非 JSON 字符串原样返回（fallback）。"""
        fake_redis_client.get = AsyncMock(return_value="raw_string")

        @cache()
        async def my_func():
            return "fresh"

        result = await my_func()
        assert result == "raw_string"

    async def test_hit_corrupt_json_falls_back_to_raw(self, fake_redis_client):
        """缓存命中——损坏 JSON 返 raw string（不抛）。"""
        fake_redis_client.get = AsyncMock(return_value="not valid json{")

        @cache()
        async def my_func():
            return "fresh"

        result = await my_func()
        assert result == "not valid json{"

    async def test_hit_does_not_call_function(self, fake_redis_client):
        """缓存命中——原函数不应被调。"""
        fake_redis_client.get = AsyncMock(return_value=json.dumps("cached"))

        call_count = 0

        @cache()
        async def my_func():
            nonlocal call_count
            call_count += 1
            return "should not happen"

        await my_func()
        await my_func()
        await my_func()
        assert call_count == 0


# ============================================================
# 3. 缓存未命中
# ============================================================
class TestCacheMiss:
    async def test_miss_executes_function(self, fake_redis_client):
        """缓存未命中——执行原函数，结果 set 后返。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache(ttl=60)
        async def my_func():
            return {"result": "computed"}

        result = await my_func()
        assert result == {"result": "computed"}
        fake_redis_client.set.assert_called_once()

    async def test_miss_set_uses_default_key(self, fake_redis_client):
        """未指定 key——使用 f"cache:{module}:{name}" 默认 key。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache()
        async def my_func():
            return "x"

        await my_func()
        call_args = fake_redis_client.set.call_args
        # key 默认是 cache:<module>:<name>
        assert call_args.args[0].startswith("cache:")
        assert "my_func" in call_args.args[0]

    async def test_miss_set_uses_custom_key(self, fake_redis_client):
        """指定 key——使用自定义 key。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache(key="my:custom:key", ttl=120)
        async def my_func():
            return "x"

        await my_func()
        call_args = fake_redis_client.set.call_args
        # 验证 key 用了自定义的
        assert call_args.args[0] == "my:custom:key"
        # 验证 ttl 传过去了
        assert call_args.kwargs.get("ttl") == 120 or call_args.args[2] == 120

    async def test_miss_set_with_no_ttl(self, fake_redis_client):
        """不传 ttl——调 set（不调 setex）。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache()  # 无 ttl
        async def my_func():
            return "x"

        await my_func()
        # redis.set 调了（不带 ttl=）
        fake_redis_client.set.assert_called_once()
        # 不应传 ttl 关键字
        call_args = fake_redis_client.set.call_args
        assert "ttl" not in call_args.kwargs

    async def test_miss_set_with_ttl_uses_setex(self, fake_redis_client):
        """ttl 指定——redis.set 调时带 ttl（不实际分 setex/set；set 内部会处理）。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache(ttl=60)
        async def my_func():
            return "x"

        await my_func()
        call_args = fake_redis_client.set.call_args
        # 验证 ttl=60 被传
        assert call_args.kwargs.get("ttl") == 60 or call_args.args[2] == 60


# ============================================================
# 4. functools.wraps 元信息保留
# ============================================================
class TestFunctoolsWraps:
    async def test_func_name_preserved(self, fake_redis_client):
        @cache()
        async def my_special_func():
            return "x"
        assert my_special_func.__name__ == "my_special_func"

    async def test_docstring_preserved(self, fake_redis_client):
        @cache()
        async def my_func():
            """My docstring."""
            return "x"
        assert "My docstring." in (my_func.__doc__ or "")


# ============================================================
# 5. 装饰器参数组合
# ============================================================
class TestDecoratorArgs:
    async def test_key_only(self, fake_redis_client):
        """只传 key，不传 ttl——走默认 key，无 ttl。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache(key="custom")
        async def my_func():
            return "x"

        await my_func()
        assert fake_redis_client.set.call_args.args[0] == "custom"

    async def test_ttl_only(self, fake_redis_client):
        """只传 ttl，不传 key——走默认 key。"""
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache(ttl=300)
        async def my_func():
            return "x"

        await my_func()
        # 走默认 key
        key = fake_redis_client.set.call_args.args[0]
        assert key.startswith("cache:") and "my_func" in key
        # ttl=300——可能是 kwargs 或第三个位置参数
        call_args = fake_redis_client.set.call_args
        ttl_value = call_args.kwargs.get("ttl")
        if ttl_value is None:
            # 位置参数：args 是 (key, value, ttl) 或 (key, value)
            if len(call_args.args) >= 3:
                ttl_value = call_args.args[2]
        assert ttl_value == 300

    @pytest.mark.xfail(reason="@cache 源码必须带括号（装饰器工厂），不支持 @cache 直接用")
    async def test_no_args_uses_full_defaults(self, fake_redis_client):
        """无参——全默认。

        注意：源码 cache(key=None, ttl=None) 是装饰器工厂——必须 @cache() 带括号。
        @cache 不带括号会 TypeError（装饰器工厂需要先被调用返 decorator）。
        """
        fake_redis_client.get = AsyncMock(return_value=None)
        fake_redis_client.set = AsyncMock(return_value=True)

        @cache
        async def my_func():
            return "x"

        result = await my_func()
        assert result == "x"


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
