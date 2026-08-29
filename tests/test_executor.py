#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.utils.executor 测试

覆盖：
1. TaskExecutor.execute 同步/异步分流
2. execute_task 函数式：每次新建 TaskExecutor
3. retry_config 透传
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from infra.utils.executor import TaskExecutor, execute_task
from infra.utils.retry import RetryConfig


# ============================================================
# 1. TaskExecutor.execute
# ============================================================
class TestTaskExecutor:
    async def test_sync_func_runs(self):
        exe = TaskExecutor()
        result = await exe.execute(lambda x: x + 1, 41)
        assert result == 42

    async def test_async_func_runs(self):
        exe = TaskExecutor()

        async def fn(x):
            await asyncio.sleep(0)
            return x * 2

        result = await exe.execute(fn, 21)
        assert result == 42

    async def test_retry_on_retryable_exception(self):
        cfg = RetryConfig(
            max_retries=2,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = TaskExecutor(default_retry_config=cfg)
        attempts = {"n": 0}

        async def fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("x")
            return "done"

        result = await exe.execute(fn)
        assert result == "done"
        assert attempts["n"] == 2

    async def test_no_retry_on_non_retryable(self):
        """未在 allow 列表的异常立即 raise。"""
        cfg = RetryConfig(
            max_retries=5,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ConnectionError,),
        )
        exe = TaskExecutor(default_retry_config=cfg)
        attempts = {"n": 0}

        async def fn():
            attempts["n"] += 1
            raise ValueError("x")

        with pytest.raises(ValueError):
            await exe.execute(fn)
        assert attempts["n"] == 1

    async def test_retry_config_override(self):
        """execute 的 retry_config 覆盖 default。"""
        default_cfg = RetryConfig(
            max_retries=10,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        override_cfg = RetryConfig(
            max_retries=1,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        exe = TaskExecutor(default_retry_config=default_cfg)
        attempts = {"n": 0}

        async def fn():
            attempts["n"] += 1
            raise ValueError("x")

        with pytest.raises(ValueError):
            await exe.execute(fn, retry_config=override_cfg)
        # override_cfg.max_retries=1 → 2 次调用
        assert attempts["n"] == 2

    async def test_default_retry_config_is_none_safe(self):
        """不传 default_retry_config 时走默认 RetryConfig。"""
        exe = TaskExecutor()
        # 即使不传配置，func 成功就直接返
        assert await exe.execute(lambda: 1) == 1


# ============================================================
# 2. execute_task 函数式
# ============================================================
class TestExecuteTask:
    async def test_sync_func(self):
        result = await execute_task(lambda: "x")
        assert result == "x"

    async def test_async_func(self):
        async def fn():
            return "y"

        assert await execute_task(fn) == "y"

    async def test_retry_config_accepted(self):
        """execute_task 的 retry_config 形参会被接收（注意：实际传给 TaskExecutor.__init__）。"""
        cfg = RetryConfig(
            max_retries=2,
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )
        attempts = {"n": 0}

        async def fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise ValueError("x")
            return "ok"

        result = await execute_task(fn, retry_config=cfg)
        assert result == "ok"
        assert attempts["n"] == 2

    async def test_each_call_creates_new_executor(self):
        """每次调用 execute_task 都新建 TaskExecutor（互不影响）。"""
        # 两次调用，第一次的 retry_config 不应影响第二次
        cfg = RetryConfig(
            max_retries=0,  # 不重试
            base_delay=0.001,
            jitter=False,
            allow_exceptions=(ValueError,),
        )

        async def fail():
            raise ValueError("x")

        async def succeed():
            return 1

        # 第一次：retry_config=max_retries=0 立即 raise
        with pytest.raises(ValueError):
            await execute_task(fail, retry_config=cfg)
        # 第二次：不传 retry_config，走默认
        result = await execute_task(succeed)
        assert result == 1


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
