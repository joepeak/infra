#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm 三个新增 API 能力的回归测试（20261021 晚间）

覆盖：
1. 批量 / 并发调用（abatch / batch）——限流、顺序保证、异常策略
2. 流式聚合（astream_collect / stream_collect）——一站式拿完整响应
3. 重试退避可观测（on_retry 回调）——退避过程可见、异常不阻断
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from infra.llm.client import OpenAIClient
from infra.llm.models import LLMRequest
from infra.utils.retry import RetryAttempt, RetryConfig, retry_async, retry_sync


# ============================================================
# 公共 helper
# ============================================================
def make_response(content: str = "hi", model: str = "m", usage: Any = None) -> MagicMock:
    """构造最小可用的 chat completion 响应。

    mock 形态必须对齐真实 openai SDK——``model`` / ``finish_reason``
    必须是 **str**（用 MagicMock 会导致 pydantic 校验失败）。
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.model = model
    resp.usage = usage
    return resp


def make_chunk(
    content: str = "",
    finish_reason: Any = None,
    usage: Any = None,
    model: str = "m",
) -> MagicMock:
    """构造一个流式 chunk。"""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].finish_reason = finish_reason
    chunk.model = model
    chunk.usage = usage
    return chunk


def make_usage(prompt: int, completion: int, total: int) -> MagicMock:
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = total
    return u


@pytest.fixture
def client() -> OpenAIClient:
    """一个不触发能力探测的客户端。"""
    c = OpenAIClient(
        api_key="k", base_url="http://x", model="m", capability_probe=False
    )
    c._async_client = MagicMock()
    c._sync_client = MagicMock()
    return c


# ============================================================
# 1. 批量 / 并发调用
# ============================================================
class TestBatch:
    """abatch / batch——并发执行多个请求。"""

    async def test_abatch_returns_all_results(self, client):
        async def create(**kwargs: Any) -> Any:
            return make_response()

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": f"q{i}"}])
            for i in range(3)
        ]
        results = await client.abatch(reqs)
        assert len(results) == 3
        assert all(r.content == "hi" for r in results)

    async def test_abatch_preserves_input_order(self, client):
        """结果顺序必须与输入一致——不能因完成先后而乱序。"""
        async def create(**kwargs: Any) -> Any:
            # 越靠前的请求越慢，模拟完成顺序与输入顺序相反
            content = kwargs["messages"][0]["content"]
            delay = 0.05 - 0.01 * int(content[1:])
            await asyncio.sleep(delay)
            return make_response(content)

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": f"q{i}"}])
            for i in range(3)
        ]
        results = await client.abatch(reqs)
        assert [r.content for r in results] == ["q0", "q1", "q2"]

    async def test_abatch_runs_concurrently(self, client):
        """并发应显著快于串行——否则并发没生效。"""
        async def create(**kwargs: Any) -> Any:
            await asyncio.sleep(0.05)
            return make_response()

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": "q"}])
            for _ in range(5)
        ]
        t0 = time.monotonic()
        await client.abatch(reqs, concurrency=5)
        elapsed = time.monotonic() - t0
        # 串行需 ~0.25s；并发应接近 0.05s
        assert elapsed < 0.2, f"耗时 {elapsed:.2f}s，未体现并发"

    async def test_abatch_respects_concurrency_limit(self, client):
        """并发上限必须生效——不能打爆 provider 限流。"""
        active = {"now": 0, "max": 0}

        async def create(**kwargs: Any) -> Any:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.02)
            active["now"] -= 1
            return make_response()

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": "q"}])
            for _ in range(8)
        ]
        await client.abatch(reqs, concurrency=2)
        assert active["max"] <= 2, f"实际并发 {active['max']} 超过上限 2"

    async def test_abatch_empty_list(self, client):
        assert await client.abatch([]) == []

    async def test_abatch_invalid_concurrency(self, client):
        reqs = [LLMRequest(messages=[{"role": "user", "content": "q"}])]
        with pytest.raises(ValueError):
            await client.abatch(reqs, concurrency=0)

    async def test_abatch_raises_on_failure_by_default(self, client):
        """默认：任一失败立即抛出（fail-fast）。"""

        async def create(**kwargs: Any) -> Any:
            raise ConnectionError("boom")

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": "q"}])
            for _ in range(2)
        ]
        with pytest.raises(ConnectionError):
            await client.abatch(reqs)

    async def test_abatch_return_exceptions(self, client):
        """return_exceptions=True：失败项以异常对象返回，不中断整体。"""

        async def create(**kwargs: Any) -> Any:
            content = kwargs["messages"][0]["content"]
            if content == "bad":
                raise ConnectionError("boom")
            return make_response(content)

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": "ok1"}]),
            LLMRequest(messages=[{"role": "user", "content": "bad"}]),
            LLMRequest(messages=[{"role": "user", "content": "ok2"}]),
        ]
        results = await client.abatch(reqs, return_exceptions=True)
        assert results[0].content == "ok1"
        assert isinstance(results[1], ConnectionError)
        assert results[2].content == "ok2"

    async def test_abatch_invokes_usage_callback(self, client):
        seen: List[Dict[str, Any]] = []

        async def create(**kwargs: Any) -> Any:
            return make_response(usage=make_usage(10, 5, 15))

        client._async_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": "q"}])
            for _ in range(2)
        ]
        await client.abatch(reqs, usage_callback=lambda info: seen.append(info))
        assert len(seen) == 2
        assert seen[0]["total_tokens"] == 15

    def test_batch_sync_wrapper(self, client):
        """同步版 batch 也能工作。"""

        def create(**kwargs: Any) -> Any:
            return make_response()

        client._sync_client.chat.completions.create = create
        reqs = [
            LLMRequest(messages=[{"role": "user", "content": f"q{i}"}])
            for i in range(3)
        ]
        results = client.batch(reqs)
        assert len(results) == 3
        assert [r.content for r in results] == ["hi", "hi", "hi"]

    def test_batch_empty_list(self, client):
        assert client.batch([]) == []

    async def test_batch_inside_event_loop(self, client):
        """在事件循环里调同步 batch 不应报 "loop already running"。"""

        def create(**kwargs: Any) -> Any:
            return make_response()

        client._sync_client.chat.completions.create = create
        reqs = [LLMRequest(messages=[{"role": "user", "content": "q"}])]
        results = client.batch(reqs)
        assert len(results) == 1


# ============================================================
# 2. 流式聚合
# ============================================================
class TestStreamCollect:
    """astream_collect / stream_collect——流式调用但拿完整响应。"""

    async def test_collects_content(self, client):
        chunks = [make_chunk("Hello "), make_chunk("World"), make_chunk(finish_reason="stop")]

        async def create(**kwargs: Any) -> Any:
            resp = MagicMock()

            async def _aiter():
                for ch in chunks:
                    yield ch

            resp.__aiter__ = lambda self: _aiter()
            return resp

        client._async_client.chat.completions.create = create
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await client.astream_collect(req)
        assert result.content == "Hello World"
        assert result.finish_reason == "stop"

    async def test_collects_usage(self, client):
        chunks = [
            make_chunk("a"),
            make_chunk(finish_reason="stop", usage=make_usage(10, 5, 15)),
        ]

        async def create(**kwargs: Any) -> Any:
            resp = MagicMock()

            async def _aiter():
                for ch in chunks:
                    yield ch

            resp.__aiter__ = lambda self: _aiter()
            return resp

        client._async_client.chat.completions.create = create
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await client.astream_collect(req)
        assert result.usage["total_tokens"] == 15
        assert result.usage["prompt_tokens"] == 10

    async def test_post_processor_applied_to_final_content(self, client):
        """post_processor 应作用于**拼接后**的完整内容。

        逐 chunk 处理会破坏跨 chunk 的标记（如 "Final answer:" 跨两个 chunk）。
        """
        chunks = [make_chunk("Final "), make_chunk("answer: 42")]

        async def create(**kwargs: Any) -> Any:
            resp = MagicMock()

            async def _aiter():
                for ch in chunks:
                    yield ch

            resp.__aiter__ = lambda self: _aiter()
            return resp

        client._async_client.chat.completions.create = create

        def processor(content: str, reasoning: Any = None) -> str:
            return content.replace("Final answer:", "").strip()

        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            post_processor=processor,
        )
        result = await client.astream_collect(req)
        assert result.content == "42"

    async def test_closes_stream_deterministically(self, client):
        """必须用 async with 保证连接释放。"""
        closed = {"value": False}
        chunks = [make_chunk("a")]

        async def create(**kwargs: Any) -> Any:
            resp = MagicMock()

            async def _aiter():
                for ch in chunks:
                    yield ch

            resp.__aiter__ = lambda self: _aiter()

            async def _close() -> None:
                closed["value"] = True

            resp.aclose = _close
            return resp

        client._async_client.chat.completions.create = create
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        await client.astream_collect(req)
        assert closed["value"] is True, "流未被关闭"

    async def test_collect_empty_stream(self, client):
        async def create(**kwargs: Any) -> Any:
            resp = MagicMock()

            async def _aiter():
                return
                yield  # pragma: no cover

            resp.__aiter__ = lambda self: _aiter()
            return resp

        client._async_client.chat.completions.create = create
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await client.astream_collect(req)
        assert result.content == ""

    def test_stream_collect_sync(self, client):
        chunks = [make_chunk("Sync "), make_chunk("OK"), make_chunk(finish_reason="stop")]

        def create(**kwargs: Any) -> Any:
            resp = MagicMock()
            resp.__iter__ = lambda self: iter(chunks)
            return resp

        client._sync_client.chat.completions.create = create
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = client.stream_collect(req)
        assert result.content == "Sync OK"


# ============================================================
# 3. 重试退避可观测
# ============================================================
class TestRetryObservability:
    """on_retry 回调——让退避过程可见。"""

    async def test_async_callback_fires_on_each_retry(self):
        events: List[RetryAttempt] = []
        cfg = RetryConfig.default().copy(
            max_retries=2,
            base_delay=0.01,
            max_delay=0.05,
            jitter=False,
            on_retry=events.append,
        )
        calls = {"n": 0}

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return "ok"

        result = await retry_async(flaky, config=cfg)
        assert result == "ok"
        assert calls["n"] == 3
        assert len(events) == 2, "两次失败应各触发一次回调"

    async def test_callback_receives_attempt_context(self):
        events: List[RetryAttempt] = []
        cfg = RetryConfig.default().copy(
            max_retries=2,
            base_delay=0.01,
            jitter=False,
            on_retry=events.append,
        )
        calls = {"n": 0}

        async def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("slow")
            return "ok"

        await retry_async(flaky, config=cfg)
        first = events[0]
        assert first.attempt == 1
        assert first.max_attempts == 3
        assert first.error_type == "TimeoutError"
        assert isinstance(first.exception, TimeoutError)
        assert first.delay > 0
        assert first.is_last is False
        assert events[1].attempt == 2

    async def test_delay_grows_exponentially(self):
        events: List[RetryAttempt] = []
        cfg = RetryConfig.default().copy(
            max_retries=3,
            base_delay=0.01,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False,
            on_retry=events.append,
        )

        async def always_fail() -> str:
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            await retry_async(always_fail, config=cfg)
        assert len(events) == 3
        assert events[0].delay < events[1].delay < events[2].delay

    async def test_no_callback_means_no_events(self):
        events: List[RetryAttempt] = []
        cfg = RetryConfig.default().copy(
            max_retries=1, base_delay=0.01, jitter=False, on_retry=events.append
        )

        async def always_fail() -> str:
            raise ConnectionError("boom")

        with pytest.raises(ConnectionError):
            await retry_async(always_fail, config=cfg)
        assert len(events) == 1

    def test_sync_callback_fires(self):
        events: List[RetryAttempt] = []
        cfg = RetryConfig.default().copy(
            max_retries=1, base_delay=0.01, jitter=False, on_retry=events.append
        )
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("boom")
            return "ok"

        assert retry_sync(flaky, config=cfg) == "ok"
        assert len(events) == 1

    def test_callback_exception_does_not_break_retry(self):
        """观测回调抛异常不该影响重试主流程。"""

        def bad_callback(attempt: RetryAttempt) -> None:
            raise RuntimeError("callback exploded")

        cfg = RetryConfig.default().copy(
            max_retries=2, base_delay=0.01, jitter=False, on_retry=bad_callback
        )
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return "recovered"

        assert retry_sync(flaky, config=cfg) == "recovered"
        assert calls["n"] == 3

    def test_deny_exception_does_not_retry(self):
        """黑名单异常不重试——回调也不应触发。"""
        events: List[RetryAttempt] = []
        cfg = RetryConfig.default().copy(
            max_retries=3,
            base_delay=0.01,
            jitter=False,
            deny_exceptions=(ValueError,),
            on_retry=events.append,
        )

        def fail() -> str:
            raise ValueError("nope")

        with pytest.raises(ValueError):
            retry_sync(fail, config=cfg)
        assert events == []

    async def test_llm_client_injects_on_retry_into_retry_config(self):
        """LLMClient 构造时传的 on_retry 应落到 retry_config 上。"""
        events: List[RetryAttempt] = []
        c = OpenAIClient(
            api_key="k",
            base_url="http://x",
            model="m",
            capability_probe=False,
            on_retry=events.append,
        )
        assert c.retry_config.on_retry is not None
        c.retry_config.on_retry(
            RetryAttempt(1, 3, 0.1, ConnectionError("x"), "ConnectionError")
        )
        assert len(events) == 1

    async def test_llm_client_retry_triggers_callback(self, client):
        """LLM 调用失败重试时，退避回调应被触发。"""
        events: List[RetryAttempt] = []
        client.retry_config = client.retry_config.copy(
            max_retries=2, base_delay=0.01, jitter=False, on_retry=events.append
        )

        async def create(**kwargs: Any) -> Any:
            raise ConnectionError("network down")

        client._async_client.chat.completions.create = create
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(ConnectionError):
            await client.ainvoke(req)
        assert len(events) == 2
