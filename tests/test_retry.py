#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.utils.retry 全功能测试

覆盖：
1. RetryConfig 默认值 / .copy() / .default/aggressive/conservative
2. RetryExecutor.should_retry：deny 优先于 allow / 未命中任何时不重试 / allow 为空时一律 False
3. RetryExecutor.execute_sync：成功直接返 / 重试至成功 / 重试至耗尽 raise 原异常 / 不可重试时立即 raise
4. RetryExecutor.execute_async：同步/异步 func 分流 / 同上失败行为
5. 函数式 retry_sync / retry_async
6. 装饰器 retry_sync_deco / retry_async_deco
7. _calc_delay 的 jitter / max_delay 上限
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from infra.utils.retry import (
    RetryConfig,
    RetryExecutor,
    retry_sync,
    retry_async,
    retry_sync_deco,
    retry_async_deco,
)


# ============================================================
# 1. RetryConfig
# ============================================================
class TestRetryConfig:
    def test_default_values(self):
        c = RetryConfig()
        assert c.max_retries == 3
        assert c.base_delay == 1.0
        assert c.max_delay == 30.0
        assert c.exponential_base == 2.0
        assert c.jitter is True
        assert ConnectionError in c.allow_exceptions
        assert TimeoutError in c.allow_exceptions
        assert OSError in c.allow_exceptions
        assert c.deny_exceptions == ()

    def test_default_classmethod(self):
        c = RetryConfig.default()
        assert c == RetryConfig() or c.max_retries == 3

    def test_aggressive_classmethod(self):
        c = RetryConfig.aggressive()
        assert c.max_retries == 10
        assert c.base_delay == 0.5
        assert c.max_delay == 10.0
        assert c.exponential_base == 1.5

    def test_conservative_classmethod(self):
        c = RetryConfig.conservative()
        assert c.max_retries == 2
        assert c.base_delay == 5.0
        assert c.max_delay == 60.0
        assert c.exponential_base == 2.0

    def test_copy_with_override(self):
        base = RetryConfig()
        new = base.copy(max_retries=10, base_delay=0.1)
        assert new.max_retries == 10
        assert new.base_delay == 0.1
        # 未改动的字段保持
        assert new.max_delay == base.max_delay
        assert new.jitter == base.jitter

    def test_copy_replaces_tuple_fields(self):
        """copy 会**替换**元组字段而非合并（已知行为）。"""
        base = RetryConfig(allow_exceptions=(ValueError,))
        new = base.copy(allow_exceptions=(KeyError,))
        assert new.allow_exceptions == (KeyError,)


# ============================================================
# 2. RetryExecutor.should_retry
# ============================================================
class TestShouldRetry:
    def test_deny_wins_over_allow(self):
        """同一异常同时在 deny 和 allow，应 deny。"""
        cfg = RetryConfig(allow_exceptions=(ValueError,), deny_exceptions=(ValueError,))
        exe = RetryExecutor(cfg)
        assert exe.should_retry(ValueError("x")) is False

    def test_allow_only(self):
        cfg = RetryConfig(allow_exceptions=(ValueError,), deny_exceptions=())
        exe = RetryExecutor(cfg)
        assert exe.should_retry(ValueError("x")) is True
        assert exe.should_retry(KeyError("x")) is False

    def test_empty_allow_returns_false_for_anything(self):
        """allow_exceptions 为空时，isinstance 永远 False。"""
        cfg = RetryConfig(allow_exceptions=(), deny_exceptions=())
        exe = RetryExecutor(cfg)
        # 任何异常都返回 False
        assert exe.should_retry(ValueError("x")) is False
        assert exe.should_retry(ConnectionError("x")) is False
        assert exe.should_retry(Exception("x")) is False

    def test_subclass_of_allow_passes(self):
        """isinstance 匹配子类。"""
        class CustomError(ConnectionError):
            pass

        cfg = RetryConfig.default()  # allow=(ConnectionError,...)
        exe = RetryExecutor(cfg)
        assert exe.should_retry(CustomError("x")) is True


# ============================================================
# 3. execute_sync
# ============================================================
class TestExecuteSync:
    def test_success_no_retry(self):
        cfg = RetryConfig(max_retries=3, base_delay=0.001, jitter=False)
        exe = RetryExecutor(cfg)
        calls = []

        def fn(x, y=0):
            calls.append((x, y))
            return x + y

        result = exe.execute_sync(fn, 1, y=2)
        assert result == 3
        assert calls == [(1, 2)]  # 只调用一次

    def test_retry_until_success(self):
        cfg = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        result = exe.execute_sync(fn)
        assert result == "ok"
        assert attempts["n"] == 3  # 调用了 3 次（前 2 次失败）

    def test_retry_exhausted_raises_last_exception(self):
        cfg = RetryConfig(
            max_retries=2,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            raise ValueError(f"fail {attempts['n']}")

        with pytest.raises(ValueError, match="fail 3"):
            exe.execute_sync(fn)
        # max_retries=2 → 最多 3 次调用
        assert attempts["n"] == 3

    def test_non_retryable_raises_immediately(self):
        """未在 allow 列表中的异常立即 raise，不重试。"""
        cfg = RetryConfig(
            max_retries=5,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ConnectionError,),
        )
        exe = RetryExecutor(cfg)
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            exe.execute_sync(fn)
        assert attempts["n"] == 1  # 只调一次

    def test_deny_exception_raises_immediately(self):
        cfg = RetryConfig(
            max_retries=5,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
            deny_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            raise ValueError("denied")

        with pytest.raises(ValueError):
            exe.execute_sync(fn)
        assert attempts["n"] == 1

    def test_sleep_called_between_retries(self, monkeypatch):
        cfg = RetryConfig(
            max_retries=2,
            base_delay=0.1,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        sleep_calls = []
        monkeypatch.setattr("infra.utils.retry.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            raise ValueError("x")

        with pytest.raises(ValueError):
            exe.execute_sync(fn)
        # max_retries=2 → 调用 3 次 → sleep 2 次
        assert len(sleep_calls) == 2
        # 第一次 sleep = base_delay * exp_base^0 = 0.1
        # 第二次 sleep = base_delay * exp_base^1 = 0.2
        assert sleep_calls[0] == pytest.approx(0.1)
        assert sleep_calls[1] == pytest.approx(0.2)

    def test_sleep_capped_at_max_delay(self, monkeypatch):
        cfg = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=0.5,  # 低于 base_delay * 2
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        sleep_calls = []
        monkeypatch.setattr("infra.utils.retry.time.sleep", lambda d: sleep_calls.append(d))

        def fn():
            raise ValueError("x")

        with pytest.raises(ValueError):
            exe.execute_sync(fn)
        # 所有 sleep 都不应超过 max_delay
        for d in sleep_calls:
            assert d <= 0.5


# ============================================================
# 4. execute_async
# ============================================================
class TestExecuteAsync:
    async def test_async_func_success(self):
        cfg = RetryConfig(max_retries=3, base_delay=0.001, jitter=False)
        exe = RetryExecutor(cfg)
        calls = []

        async def fn(x):
            calls.append(x)
            return x * 2

        result = await exe.execute_async(fn, 5)
        assert result == 10
        assert calls == [5]

    async def test_sync_func_runs_in_executor(self):
        """同步函数走 run_in_executor 分支。"""
        cfg = RetryConfig(max_retries=2, base_delay=0.001, jitter=False)
        exe = RetryExecutor(cfg)
        calls = []

        def fn():
            calls.append(1)
            return 42

        result = await exe.execute_async(fn)
        assert result == 42
        assert calls == [1]

    async def test_async_func_retry_until_success(self):
        cfg = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        attempts = {"n": 0}

        async def fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("first")
            return "done"

        result = await exe.execute_async(fn)
        assert result == "done"
        assert attempts["n"] == 2

    async def test_async_func_retry_exhausted_raises(self):
        cfg = RetryConfig(
            max_retries=1,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)

        async def fn():
            raise ValueError("always fail")

        with pytest.raises(ValueError, match="always fail"):
            await exe.execute_async(fn)

    async def test_async_sleep_uses_asyncio_sleep(self, monkeypatch):
        cfg = RetryConfig(
            max_retries=2,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = RetryExecutor(cfg)
        sleep_calls = []
        real_sleep = asyncio.sleep

        async def fake_sleep(d):
            sleep_calls.append(d)
            await real_sleep(0)  # 真正让出但不阻塞

        monkeypatch.setattr("infra.utils.retry.asyncio.sleep", fake_sleep)

        async def fn():
            raise ValueError("x")

        with pytest.raises(ValueError):
            await exe.execute_async(fn)
        assert len(sleep_calls) == 2


# ============================================================
# 5. 函数式 API
# ============================================================
class TestFunctionalApi:
    def test_retry_sync_success(self):
        cfg = RetryConfig(max_retries=2, base_delay=0.001, jitter=False)
        result = retry_sync(lambda: 100, config=cfg)
        assert result == 100

    def test_retry_sync_raises(self):
        cfg = RetryConfig(
            max_retries=1,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(KeyError,),
        )
        with pytest.raises(KeyError):
            retry_sync(lambda: (_ for _ in ()).throw(KeyError("x")), config=cfg)

    async def test_retry_async_success(self):
        cfg = RetryConfig(max_retries=2, base_delay=0.001, jitter=False)
        result = await retry_async(lambda: 7, config=cfg)
        assert result == 7

    async def test_retry_async_raises(self):
        cfg = RetryConfig(
            max_retries=1,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(KeyError,),
        )

        async def boom():
            raise KeyError("k")

        with pytest.raises(KeyError):
            await retry_async(boom, config=cfg)


# ============================================================
# 6. 装饰器
# ============================================================
class TestDecorators:
    def test_retry_sync_deco_success(self):
        @retry_sync_deco(config=RetryConfig(max_retries=2, base_delay=0.001, jitter=False))
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_retry_sync_deco_retries(self):
        attempts = {"n": 0}

        @retry_sync_deco(
            config=RetryConfig(
                max_retries=3,
                base_delay=0.001,
                jitter=False,
                allow_exceptions=(ValueError,),
            )
        )
        def fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("x")
            return "ok"

        assert fn() == "ok"
        assert attempts["n"] == 2

    def test_retry_sync_deco_raises(self):
        @retry_sync_deco(
            config=RetryConfig(
                max_retries=1,
                base_delay=0.001,
                jitter=False,
                allow_exceptions=(ValueError,),
            )
        )
        def fn():
            raise ValueError("x")

        with pytest.raises(ValueError):
            fn()

    async def test_retry_async_deco_success(self):
        @retry_async_deco(config=RetryConfig(max_retries=2, base_delay=0.001, jitter=False))
        async def fn():
            return "ok"

        assert await fn() == "ok"

    async def test_retry_async_deco_retries(self):
        attempts = {"n": 0}

        @retry_async_deco(
            config=RetryConfig(
                max_retries=3,
                base_delay=0.001,
                jitter=False,
                allow_exceptions=(ValueError,),
            )
        )
        async def fn():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("x")
            return "done"

        result = await fn()
        assert result == "done"
        assert attempts["n"] == 3


# ============================================================
# 7. _calc_delay 行为
# ============================================================
class TestCalcDelay:
    def test_no_jitter_exponential_growth(self):
        cfg = RetryConfig(
            base_delay=0.1,
            max_delay=100.0,
            exponential_base=2.0,
            jitter=False,
        )
        exe = RetryExecutor(cfg)
        # attempt 0 → 0.1
        # attempt 1 → 0.2
        # attempt 2 → 0.4
        assert exe._calc_delay(0) == pytest.approx(0.1)
        assert exe._calc_delay(1) == pytest.approx(0.2)
        assert exe._calc_delay(2) == pytest.approx(0.4)

    def test_jitter_within_range(self):
        cfg = RetryConfig(
            base_delay=1.0,
            max_delay=100.0,
            exponential_base=2.0,
            jitter=True,
        )
        exe = RetryExecutor(cfg)
        # attempt 0: base=1.0, jitter 0.5-1.5 → 0.5-1.5
        for _ in range(50):
            d = exe._calc_delay(0)
            assert 0.5 <= d <= 1.5

    def test_max_delay_cap(self):
        cfg = RetryConfig(
            base_delay=1.0,
            max_delay=2.0,
            exponential_base=10.0,
            jitter=False,
        )
        exe = RetryExecutor(cfg)
        # attempt 0 → 1.0
        # attempt 1 → min(10.0, 2.0) = 2.0
        # attempt 2 → min(100.0, 2.0) = 2.0
        assert exe._calc_delay(0) == pytest.approx(1.0)
        assert exe._calc_delay(1) == pytest.approx(2.0)
        assert exe._calc_delay(2) == pytest.approx(2.0)


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
