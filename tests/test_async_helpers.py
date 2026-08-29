#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.utils.async_helpers 测试

覆盖：
1. sync_retry：无参 func / 一直失败最终 raise / 成功路径
2. merge_sync_results：空参 / 单个 dict / 多个 dict 合并 / 部分 None / 全部 None
3. merge_sync_results：非 dict 返回被跳过 / 函数抛异常被吞
4. merge_sync_results：超时控制
"""

from __future__ import annotations

import asyncio
import time

import pytest

from infra.utils.async_helpers import sync_retry, merge_sync_results


# ============================================================
# 1. sync_retry
# ============================================================

@pytest.fixture(autouse=True, scope="class")
def _patch_sleep_class(request):
    """
    sync_retry 的 time.sleep 替换成 no-op——只作用 TestSyncRetry 类。
    默认 base_delay=2.0 + 多次重试可达 60s+，没这个 monkeypatch 会 hang。

    注意：只 patch TestSyncRetry，不影响 TestMergeSyncResults。merge_sync_results
    的 hang 风险由"短 timeout + boom" 模式独立控制（见 test_exceptions_swallowed 等）。
    """
    if request.cls is not TestSyncRetry:
        yield
        return
    import infra.utils.async_helpers as ah
    original = ah.time.sleep
    ah.time.sleep = lambda d: None
    try:
        yield
    finally:
        ah.time.sleep = original


class TestSyncRetry:
    def test_success_no_retry(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return "ok"

        assert sync_retry(fn) == "ok"
        assert calls["n"] == 1

    def test_retry_until_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        assert sync_retry(fn, max_retries=5, base_delay=0.001) == "ok"
        assert calls["n"] == 3

    def test_exhausted_raises(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("always fail")

        with pytest.raises(ValueError, match="always fail"):
            sync_retry(fn, max_retries=2, base_delay=0.001)
        # max_retries=2 → 3 次调用
        assert calls["n"] == 3

    def test_signature_accepts_retry_params_positionally(self):
        """sync_retry 签名 (func, max_retries, base_delay, max_delay) 都可用位置参数。"""
        def fn():
            return "x"

        # 位置参数：max_retries=1, base_delay=0.001, max_delay=0.002
        assert sync_retry(fn, 1, 0.001, 0.002) == "x"

        # 关键字参数
        assert sync_retry(fn, max_retries=1, base_delay=0.001, max_delay=0.002) == "x"

    def test_callable_invoked_without_args(self, monkeypatch):
        """sync_retry 内部用 fn() 不传参——fn 必须是零参可调用。

        注意：要 monkeypatch sleep，否则 fn_with_arg 6 次 TypeError 重试会真 sleep ~60s。
        """
        # 把 sleep 替换成 no-op，避免 retry 间的 sleep 把测试拖到分钟级
        monkeypatch.setattr(
            "infra.utils.async_helpers.time.sleep",
            lambda d: None,
        )
        def fn_with_arg(x):
            return x

        with pytest.raises(TypeError):
            sync_retry(fn_with_arg, max_retries=2)  # 2 次重试 = 3 次尝试

    def test_retries_any_exception(self):
        """sync_retry 对任何 Exception 都重试（无 allow/deny 过滤）。"""
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise KeyError("k")
            return "ok"

        assert sync_retry(fn, max_retries=5, base_delay=0.001) == "ok"
        assert calls["n"] == 2

    def test_max_delay_caps_base_delay_before_jitter(self):
        """
        源码行为：`delay = min(base_delay * 2**attempt, max_delay)` 在 jitter 之前；
        jitter 系数 = 0.5 + random.random() ∈ [0.5, 1.5]。

        验证：把 random.random() 锁成 0.0 → jitter=0.5 → 实际 sleep = max_delay*0.5，
        即"被截断"的效果稳定可断言。
        """
        from infra.utils import async_helpers as ah
        sleep_calls = []
        original_sleep = ah.time.sleep
        original_random = ah.random.random
        ah.time.sleep = lambda d: sleep_calls.append(d)
        ah.random.random = lambda: 0.0  # jitter = 0.5 + 0 = 0.5
        try:
            calls = {"n": 0}

            def fn():
                calls["n"] += 1
                raise ValueError("x")

            with pytest.raises(ValueError):
                sync_retry(fn, max_retries=5, base_delay=10.0, max_delay=0.5)
        finally:
            ah.time.sleep = original_sleep
            ah.random.random = original_random

        # jitter 锁到 0.5：每次 sleep = min(base*2^i, max_delay) * 0.5 = 0.5 * 0.5 = 0.25
        assert len(sleep_calls) == 5  # max_retries=5 → 5 次重试 = 5 次 sleep
        for d in sleep_calls:
            assert d == pytest.approx(0.25)


# ============================================================
# 2. merge_sync_results
# ============================================================
class TestMergeSyncResults:
    async def test_no_args_returns_none(self):
        assert await merge_sync_results() is None

    async def test_single_dict(self):
        result = await merge_sync_results(lambda: {"a": 1})
        assert result == {"a": 1}

    async def test_multiple_dicts_merged(self):
        def a():
            return {"a": 1}

        def b():
            return {"b": 2}

        result = await merge_sync_results(a, b)
        assert result == {"a": 1, "b": 2}

    async def test_later_overrides_earlier(self):
        def a():
            return {"x": 1, "y": 1}

        def b():
            return {"y": 2, "z": 2}

        result = await merge_sync_results(a, b)
        assert result == {"x": 1, "y": 2, "z": 2}

    async def test_none_results_are_skipped(self):
        def a():
            return None

        def b():
            return {"b": 2}

        result = await merge_sync_results(a, b)
        assert result == {"b": 2}

    async def test_all_none_returns_none(self):
        result = await merge_sync_results(lambda: None, lambda: None)
        assert result is None

    async def test_exceptions_swallowed(self, monkeypatch):
        """merge_sync_results 吞掉所有 Exception，返回 None。

        用 monkeypatch 锁住 sync_retry 的 time.sleep（noop），避免 boom 触发
        6 次重试 sleep 几十秒撑满 ThreadPoolExecutor 阻塞后续测试。
        """
        monkeypatch.setattr("infra.utils.async_helpers.time.sleep", lambda d: None)

        def boom():
            raise ValueError("boom")

        def ok():
            return {"k": 1}

        result = await merge_sync_results(boom, ok, timeout=0.1)
        assert result == {"k": 1}

    async def test_all_exceptions_returns_none(self, monkeypatch):
        """所有函数都抛异常 → 返回 None。

        同上：monkeypatch 锁 sleep 防 hang。
        """
        monkeypatch.setattr("infra.utils.async_helpers.time.sleep", lambda d: None)

        def boom():
            raise ValueError("boom")

        result = await merge_sync_results(boom, boom, timeout=0.1)
        assert result is None

    async def test_non_dict_returned_value_skipped(self):
        """非 dict 的返回值会被跳过（仅打 warning）。"""
        result = await merge_sync_results(
            lambda: "not a dict",
            lambda: {"k": 1},
        )
        assert result == {"k": 1}

    async def test_empty_dict_also_merged_as_empty(self):
        """返回空 dict 也会参与合并，但不会改变结果。"""
        result = await merge_sync_results(
            lambda: {},
            lambda: {"k": 1},
        )
        assert result == {"k": 1}

    async def test_timeout_raises_returned_as_none(self):
        """超时会被捕获为 None。"""
        def slow():
            time.sleep(0.3)
            return {"slow": 1}

        def fast():
            return {"fast": 1}

        # 极小 timeout：0.05s < slow 的 0.3s sleep
        result = await merge_sync_results(slow, fast, timeout=0.05)
        # slow 超时 → None；fast 成功 → 合并
        assert result == {"fast": 1}

    async def test_concurrent_execution(self):
        """两个函数并发执行（不是串行）。"""
        def slow():
            time.sleep(0.3)
            return {"slow": 1}

        def slow2():
            time.sleep(0.3)
            return {"slow2": 1}

        start = time.time()
        result = await merge_sync_results(slow, slow2, timeout=2.0)
        elapsed = time.time() - start
        # 并发执行：总耗时 ≈ 0.3s，而非 0.6s
        assert result == {"slow": 1, "slow2": 1}
        assert elapsed < 0.5  # 远小于 0.6s

    async def test_runs_in_thread_pool(self):
        """函数运行在独立线程，不阻塞事件循环。"""
        def blocking():
            time.sleep(0.2)
            return {"k": 1}

        loop = asyncio.get_running_loop()
        # 在 await 期间应能调度其他任务
        start = time.time()
        result = await merge_sync_results(blocking, timeout=2.0)
        elapsed = time.time() - start
        assert result == {"k": 1}
        # 应该真的 sleep 了 0.2s（说明不是被异步魔法加速了）
        assert elapsed >= 0.15


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
