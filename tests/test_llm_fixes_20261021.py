#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm 深度缺陷修复的回归测试（20261021）

覆盖本轮实测发现并修复的问题：
1. 流式提前 break → HTTP 连接不释放（P0-1）
2. total_tokens=0 时成本统计漏记（P0-2）
3. 能力注册表缓存键含 api_key（P1-1）
4. 能力注册表无上限（P1-2）
5. CoT 提取截断错位置（P1-3）
6. poll_video 吞掉主端点错误（P2-1）
7. retry_config 类型不校验（P2-2）
8. probe_capabilities 零覆盖（P2-3）
9. total_tokens 构造时回退（O1）
10. 能力参数适配逻辑统一（O2）
11. DECISION_FIELDS 应为 ClassVar（O4）
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from infra.llm.capabilities import (
    ProviderCapabilities,
    ProviderCapabilityRegistry,
    get_registry,
    reset_registry,
)
from infra.llm.client import (
    AsyncStreamIterator,
    OpenAIClient,
    OpenAIMediaClient,
    SyncStreamIterator,
)
from infra.llm.models import LLMRequest, LLMResponse
from infra.llm.postprocess import _extract_after_final_answer, strip_cot_after_answer
from infra.utils.retry import RetryConfig


# ============================================================
# 公共 helper
# ============================================================
def make_chunk(
    text: str = "c",
    usage: Any = None,
    finish_reason: Optional[str] = None,
) -> MagicMock:
    """构造一个 OpenAI 风格的流式 chunk。"""
    ch = MagicMock()
    ch.choices = [MagicMock()]
    ch.choices[0].delta = MagicMock()
    ch.choices[0].delta.content = text
    ch.choices[0].delta.tool_calls = None
    ch.choices[0].delta.reasoning_content = None
    ch.choices[0].delta.reasoning_details = None
    ch.choices[0].finish_reason = finish_reason
    ch.model = "gpt-4o"
    ch.usage = usage
    return ch


def make_usage(prompt: int, completion: int, total: int) -> MagicMock:
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = total
    return u


class FakeAsyncStream:
    """假的异步流——记录 aclose 是否被调用。"""

    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = chunks
        self.closed = 0

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for c in self._chunks:
                yield c

        return gen()

    async def aclose(self) -> None:
        self.closed += 1


class FakeSyncStream:
    """假的同步流——记录 close 是否被调用。"""

    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = chunks
        self.closed = 0

    def __iter__(self) -> Any:
        return iter(self._chunks)

    def close(self) -> None:
        self.closed += 1


class OpenAIShapedAsyncStream:
    """**严格对齐 openai 3.x ``AsyncStream`` 形态**的假流。

    关键差异：只有协程 ``close()``，**没有** ``aclose()``。
    openai 的实现::

        async def close(self) -> None:
            await self.response.aclose()

    用这个类才能测出「用 ``hasattr(stream, 'aclose')`` 判断会漏关」的真实 bug。
    """

    def __init__(self, chunks: List[Any], raise_at_end: Optional[Exception] = None) -> None:
        self._chunks = chunks
        self._raise_at_end = raise_at_end
        self.closed = 0

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for c in self._chunks:
                yield c
            if self._raise_at_end is not None:
                raise self._raise_at_end

        return gen()

    async def close(self) -> None:
        """openai 3.x 的真实签名——协程 close，无 aclose。"""
        self.closed += 1

    async def __aenter__(self) -> "OpenAIShapedAsyncStream":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class OpenAIShapedSyncStream:
    """对齐 openai 同步 ``Stream`` 形态的假流（``close()`` 是同步方法）。"""

    def __init__(self, chunks: List[Any]) -> None:
        self._chunks = chunks
        self.closed = 0

    def __iter__(self) -> Any:
        return iter(self._chunks)

    def close(self) -> None:
        self.closed += 1


def make_client(mock: MagicMock) -> OpenAIClient:
    """构造一个不触发 provider 初始化的 OpenAIClient。"""
    c = OpenAIClient.__new__(OpenAIClient)
    c._async_client = mock
    c._fallback_async_client = mock
    c._sync_client = mock
    c._fallback_sync_client = mock
    c.default_model = "gpt-4o"
    c.fallback_model = None
    c.base_url = None
    c.fallback_base_url = None
    c.retry_config = None
    c.capability_probe = False
    return c


@pytest.fixture(autouse=True)
def _reset_registry():
    """能力注册表是进程内全局——每个测试后还原。"""
    reset_registry()
    yield
    reset_registry()


# ============================================================
# 1. P0-1：流式提前 break → 连接不释放
# ============================================================
class TestStreamEarlyBreakReleasesConnection:
    """流式连接的确定性释放。

    背景：裸 async generator 的 finally 只在 GC 时执行，消费方提前 break
    且持有引用时底层 HTTP 连接不释放。

    注：CPython 的 ``async for ... break`` **不会**通知迭代器（语言限制），
    所以确定性释放靠 ``async with`` / 显式 ``aclose()``；GC 兜底由
    ``__del__`` 提供。
    """

    async def test_astream_async_with_releases(self):
        """async with —— 确定性释放（推荐用法）。"""
        stream_obj = FakeAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        async with c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}])) as stream:
            async for _ in stream:
                break
        assert stream_obj.closed == 1, "async with 退出后底层流未关闭"

    async def test_astream_explicit_aclose_releases(self):
        """显式 aclose() —— 确定性释放。"""
        stream_obj = FakeAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        gen = c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        async for _ in gen:
            break
        await gen.aclose()
        assert stream_obj.closed == 1

    async def test_astream_aclose_is_idempotent(self):
        """aclose() 幂等——多次调用不报错、不重复关闭。"""
        stream_obj = FakeAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        gen = c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        async for _ in gen:
            break
        await gen.aclose()
        await gen.aclose()
        assert stream_obj.closed == 1

    async def test_astream_full_consumption_closes(self):
        """完整消费也要关闭底层流。"""
        stream_obj = FakeAsyncStream([make_chunk(f"c{i}") for i in range(3)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        n = 0
        async for _ in c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}])):
            n += 1
        assert n == 3
        assert stream_obj.closed == 1

    async def test_astream_mid_stream_error_closes(self):
        """流中途抛异常也要关闭底层流。"""

        class BadStream:
            def __init__(self) -> None:
                self.closed = 0

            def __aiter__(self) -> Any:
                async def gen() -> Any:
                    yield make_chunk("a")
                    raise RuntimeError("mid-stream error")

                return gen()

            async def aclose(self) -> None:
                self.closed += 1

        bad = BadStream()
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=bad)
        c = make_client(mock)

        with pytest.raises(RuntimeError, match="mid-stream error"):
            async for _ in c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}])):
                pass
        assert bad.closed == 1

    async def test_astream_gc_fallback_closes(self):
        """迭代器被 GC 时兜底关闭（裸 break + 变量不再持有时生效）。"""
        stream_obj = FakeAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        gen = c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        async for _ in gen:
            break
        del gen
        gc.collect()
        await asyncio.sleep(0)
        assert stream_obj.closed >= 1

    def test_sync_stream_with_releases(self):
        """同步版 with —— 确定性释放。"""
        stream_obj = FakeSyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = MagicMock(return_value=stream_obj)
        c = make_client(mock)

        with c.stream(LLMRequest(messages=[{"role": "user", "content": "x"}])) as stream:
            for _ in stream:
                break
        assert stream_obj.closed == 1

    def test_sync_stream_explicit_close(self):
        stream_obj = FakeSyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = MagicMock(return_value=stream_obj)
        c = make_client(mock)

        gen = c.stream(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        for _ in gen:
            break
        gen.close()
        assert stream_obj.closed == 1

    def test_sync_stream_full_consumption_closes(self):
        stream_obj = FakeSyncStream([make_chunk(f"c{i}") for i in range(3)])
        mock = MagicMock()
        mock.chat.completions.create = MagicMock(return_value=stream_obj)
        c = make_client(mock)

        n = 0
        for _ in c.stream(LLMRequest(messages=[{"role": "user", "content": "x"}])):
            n += 1
        assert n == 3
        assert stream_obj.closed == 1

    def test_stream_iterator_classes_are_exported(self):
        """两个迭代器类是公开的（消费方可以 isinstance 判断）。"""
        assert AsyncStreamIterator is not None
        assert SyncStreamIterator is not None


# ============================================================
# 1b. 回归：真实 openai SDK 的流只有协程 close()，没有 aclose()
# ============================================================
class TestRealOpenAIStreamShape:
    """**回归测试**——锁住「真实 openai 流形态下连接必须释放」。

    背景（20261021 实测发现）：openai 3.x 的 ``AsyncStream`` **只有**
    ``close()``（且是 ``async def`` 协程），**没有** ``aclose()``。

    旧实现用 ``hasattr(stream, "aclose")`` 判断是否关闭，导致在真实
    openai SDK 下**底层 HTTP 流从未被关闭**——正是本次要修的 P0 泄漏。
    之前的测试用假流实现了 ``aclose()``，所以全绿但生产仍泄漏。

    本组测试的假流**严格对齐 openai 3.x 的方法签名**：
    有协程 ``close()``，**没有** ``aclose()``。
    """

    async def test_async_with_releases_real_shape(self):
        """async with —— 真实形态下也必须关闭。"""
        stream_obj = OpenAIShapedAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        async with c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}])) as stream:
            async for _ in stream:
                break
        assert stream_obj.closed == 1, "真实 openai 形态下底层流未关闭——连接泄漏"

    async def test_explicit_aclose_releases_real_shape(self):
        """显式 aclose() —— 真实形态下也必须关闭。"""
        stream_obj = OpenAIShapedAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        gen = c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        async for _ in gen:
            break
        await gen.aclose()
        assert stream_obj.closed == 1, "显式 aclose 未关闭真实形态的流"

    async def test_full_consumption_releases_real_shape(self):
        """完整消费 —— 真实形态下也必须关闭。"""
        stream_obj = OpenAIShapedAsyncStream([make_chunk(f"c{i}") for i in range(3)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        n = 0
        async for _ in c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}])):
            n += 1
        assert n == 3
        assert stream_obj.closed == 1

    async def test_mid_stream_error_releases_real_shape(self):
        """流中途异常 —— 真实形态下也必须关闭。"""
        stream_obj = OpenAIShapedAsyncStream([make_chunk("a")], raise_at_end=RuntimeError("boom"))
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        with pytest.raises(RuntimeError, match="boom"):
            async for _ in c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}])):
                pass
        assert stream_obj.closed == 1

    async def test_gc_fallback_releases_real_shape(self):
        """GC 兜底 —— 真实形态下也必须关闭（且不产生 never-awaited 警告）。"""
        stream_obj = OpenAIShapedAsyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        gen = c.astream(LLMRequest(messages=[{"role": "user", "content": "x"}]))
        async for _ in gen:
            break
        del gen
        gc.collect()
        await asyncio.sleep(0.05)
        assert stream_obj.closed >= 1, "GC 兜底未关闭真实形态的流"

    def test_sync_stream_real_shape(self):
        """同步版——openai 同步 ``Stream.close()`` 是同步方法。"""
        stream_obj = OpenAIShapedSyncStream([make_chunk(f"c{i}") for i in range(50)])
        mock = MagicMock()
        mock.chat.completions.create = MagicMock(return_value=stream_obj)
        c = make_client(mock)

        with c.stream(LLMRequest(messages=[{"role": "user", "content": "x"}])) as stream:
            for _ in stream:
                break
        assert stream_obj.closed == 1

    def test_openai_actual_stream_has_no_aclose(self):
        """守住前提：真实 openai SDK 的流确实没有 ``aclose()``。

        若未来 openai 改版加上了 ``aclose()``，这条会失败，提醒更新实现。
        """
        import openai

        assert hasattr(openai.AsyncStream, "close"), "openai AsyncStream 应有 close()"
        assert not hasattr(openai.AsyncStream, "aclose"), (
            "openai AsyncStream 现在有 aclose() 了——请复核 AsyncStreamIterator 的关闭逻辑"
        )


# ============================================================
# 2. P0-2：total_tokens=0 时成本统计漏记
# ============================================================
class TestStreamUsageWithZeroTotal:
    """回归 bug：``if aggregated_total > 0`` 门禁导致 provider 返回
    total_tokens=0 时 usage_callback 完全不触发——成本统计漏记。"""

    async def test_callback_fires_when_total_is_zero(self):
        stream_obj = FakeAsyncStream([
            make_chunk("a"),
            make_chunk("b", usage=make_usage(10, 5, 0)),
        ])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        calls: List[Dict[str, Any]] = []
        async for _ in c.astream(
            LLMRequest(messages=[{"role": "user", "content": "x"}]),
            usage_callback=lambda info: calls.append(info),
        ):
            pass

        assert len(calls) == 1, "total_tokens=0 时 usage_callback 未触发 —— 成本漏记"
        assert calls[0]["prompt_tokens"] == 10
        assert calls[0]["completion_tokens"] == 5
        # total 回退到 prompt + completion
        assert calls[0]["total_tokens"] == 15

    async def test_callback_fires_with_normal_usage(self):
        stream_obj = FakeAsyncStream([
            make_chunk("b", usage=make_usage(10, 5, 15)),
        ])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        calls: List[Dict[str, Any]] = []
        async for _ in c.astream(
            LLMRequest(messages=[{"role": "user", "content": "x"}]),
            usage_callback=lambda info: calls.append(info),
        ):
            pass
        assert len(calls) == 1
        assert calls[0]["total_tokens"] == 15

    async def test_no_callback_when_no_usage_at_all(self):
        """完全没有 usage 时不该触发（避免虚假记录）。"""
        stream_obj = FakeAsyncStream([make_chunk("a"), make_chunk("b")])
        mock = MagicMock()
        mock.chat.completions.create = AsyncMock(return_value=stream_obj)
        c = make_client(mock)

        calls: List[Dict[str, Any]] = []
        async for _ in c.astream(
            LLMRequest(messages=[{"role": "user", "content": "x"}]),
            usage_callback=lambda info: calls.append(info),
        ):
            pass
        assert calls == []


# ============================================================
# 3. P1-1：能力注册表缓存键含 api_key
# ============================================================
class TestRegistryKeySanitization:
    """回归 bug：base_url 形如 ``https://sk-xxx@host`` 时密钥进缓存键，
    与模块 docstring「缓存键不含 api_key」的约束矛盾。"""

    def test_key_strips_userinfo(self):
        key = ProviderCapabilityRegistry.key("https://sk-SECRET@api.example.com/v1", "gpt-4o")
        assert "SECRET" not in key
        assert "api.example.com" in key

    def test_key_strips_user_pass(self):
        key = ProviderCapabilityRegistry.key("https://user:pass@api.example.com/v1", "gpt-4o")
        assert "pass" not in key
        assert "api.example.com" in key

    def test_key_strips_token_without_scheme(self):
        key = ProviderCapabilityRegistry.key("token@api.example.com/v1", "gpt-4o")
        assert "token" not in key
        assert "api.example.com" in key

    def test_normal_url_unchanged(self):
        key = ProviderCapabilityRegistry.key("https://api.example.com/v1", "gpt-4o")
        assert key == "https://api.example.com/v1::gpt-4o"

    def test_trailing_slash_normalized(self):
        a = ProviderCapabilityRegistry.key("https://api.example.com/v1/", "m")
        b = ProviderCapabilityRegistry.key("https://api.example.com/v1", "m")
        assert a == b

    def test_snapshot_contains_no_secret(self):
        """snapshot() 是给日志/诊断用的——绝不能含密钥。"""
        r = ProviderCapabilityRegistry()
        r.update("https://sk-SECRETKEY@api.example.com/v1", "gpt-4o", source="learn", supports_tools=False)
        assert "SECRETKEY" not in str(r.snapshot())

    def test_same_host_different_credentials_shares_entry(self):
        """同 host 不同凭据应命中同一条缓存。"""
        r = ProviderCapabilityRegistry()
        r.update("https://sk-A@api.example.com/v1", "m", source="learn", supports_tools=True)
        r.update("https://sk-B@api.example.com/v1", "m", source="learn", supports_tools=False)
        assert len(r.snapshot()) == 1


# ============================================================
# 4. P1-2：能力注册表无上限
# ============================================================
class TestRegistryLruBound:
    """回归 bug：缓存无上限/TTL/LRU——多端点多模型场景下无限增长。"""

    def test_entries_capped(self):
        r = ProviderCapabilityRegistry(max_entries=5)
        for i in range(20):
            r.update(f"https://h{i}.example.com/v1", f"m{i}", source="learn", supports_tools=True)
        assert len(r.snapshot()) == 5

    def test_evicts_oldest_first(self):
        r = ProviderCapabilityRegistry(max_entries=3)
        for i in range(3):
            r.update(f"https://y{i}.com/v1", "m", source="learn", supports_tools=True)
        r.update("https://new.com/v1", "m", source="learn", supports_tools=True)
        keys = list(r.snapshot().keys())
        assert "https://y0.com/v1::m" not in keys
        assert "https://new.com/v1::m" in keys

    def test_access_refreshes_lru_order(self):
        """访问过的条目不该被优先淘汰。"""
        r = ProviderCapabilityRegistry(max_entries=3)
        for i in range(3):
            r.update(f"https://y{i}.com/v1", "m", source="learn", supports_tools=True)
        r.get("https://y0.com/v1", "m")  # 刷新 y0
        r.update("https://new.com/v1", "m", source="learn", supports_tools=True)
        keys = list(r.snapshot().keys())
        assert "https://y0.com/v1::m" in keys
        assert "https://y1.com/v1::m" not in keys

    def test_learn_from_error_also_respects_cap(self):
        r = ProviderCapabilityRegistry(max_entries=4)
        for i in range(10):
            r.learn_from_error(f"https://h{i}.com/v1", "m", "thinking not supported")
        assert len(r.snapshot()) == 4

    def test_default_max_entries_is_bounded(self):
        assert 0 < ProviderCapabilityRegistry.MAX_ENTRIES <= 4096


# ============================================================
# 5. P1-3：CoT 提取截断错位置
# ============================================================
class TestFinalAnswerExtraction:
    """回归 bug：取**最后一个** ``answer:`` —— 答案正文里再次出现
    "answer:" 时会把真正答案内容吃掉。"""

    def test_answer_inside_content_not_truncated(self):
        content = (
            "Let me think about this.\n"
            "Step 1: consider the data.\n"
            "Final answer: The answer: 42 dollars."
        )
        extracted = _extract_after_final_answer(content)
        assert extracted == "The answer: 42 dollars."

    def test_strong_marker_on_own_line(self):
        content = "Let me think.\nStep 1: blah.\nFinal answer:\n42 dollars."
        assert _extract_after_final_answer(content) == "42 dollars."

    def test_chinese_strong_marker(self):
        content = "让我分析一下。\n结论：\n答案如下"
        assert _extract_after_final_answer(content) == "答案如下"

    def test_weak_marker_takes_first(self):
        content = "Let me think. answer: 42 is correct."
        assert _extract_after_final_answer(content) == "42 is correct."

    def test_no_marker_returns_none(self):
        assert _extract_after_final_answer("just some text") is None

    def test_strip_cot_after_answer_keeps_full_answer(self):
        content = (
            "Let me think about this.\n"
            "Step 1: consider the data.\n"
            "Final answer: The answer: 42 dollars."
        )
        stripped = strip_cot_after_answer(content)
        assert stripped == "The answer: 42 dollars."

    def test_strip_cot_normal_answer_untouched(self):
        """非 CoT 的正常回答原样返回。"""
        content = "The answer: 42 dollars."
        assert strip_cot_after_answer(content) == content


# ============================================================
# 6. P2-1：poll_video 吞掉主端点真实错误
# ============================================================
class TestPollVideoErrorPropagation:
    """回归 bug：主端点异常被 ``except Exception`` 吞掉，fallback 也失败时
    只暴露 fallback 的错误——主因（如余额不足）丢失。"""

    @staticmethod
    def make_media_client() -> OpenAIMediaClient:
        c = OpenAIMediaClient.__new__(OpenAIMediaClient)
        c._async_client = MagicMock()
        c.image_model = "dall-e-3"
        c.video_model = "seedance"
        c.base_url = None
        c._provider = "openai"
        c.retry_config = None
        return c

    async def test_both_endpoints_fail_keeps_primary_reason(self):
        c = self.make_media_client()

        async def fail_retrieve(tid: str) -> Any:
            raise RuntimeError("余额不足")

        async def fail_create(**kwargs: Any) -> Any:
            raise RuntimeError("不支持该端点")

        c._async_client.video.tasks.retrieve = fail_retrieve
        c._async_client.chat.completions.create = fail_create

        with pytest.raises(RuntimeError) as exc:
            await c.poll_video("task-1", interval=0.01, max_wait=0.05)

        assert "余额不足" in str(exc.value)
        assert "不支持该端点" in str(exc.value)

    async def test_fallback_success_still_works(self):
        c = self.make_media_client()

        async def fail_retrieve(tid: str) -> Any:
            raise RuntimeError("no dedicated endpoint")

        resp = MagicMock()
        resp.status = "completed"
        resp.completed = True
        resp.video_url = "http://v.mp4"
        resp.duration_seconds = 5
        resp.cost = 0.1
        resp.currency = "CNY"
        resp.cost_metadata = {}

        async def ok_create(**kwargs: Any) -> Any:
            return resp

        c._async_client.video.tasks.retrieve = fail_retrieve
        c._async_client.chat.completions.create = ok_create

        result = await c.poll_video("task-2", interval=0.01, max_wait=1)
        assert result.videos[0].status == "completed"
        assert result.videos[0].video_url == "http://v.mp4"


# ============================================================
# 7. P2-2：retry_config 类型不校验
# ============================================================
class TestRetryConfigTyping:
    """回归 bug：``retry_config: Optional[Any]`` —— pydantic 不校验，
    传 dict 会一路带到 retry_async 才炸出难懂的 AttributeError。"""

    def test_dict_is_converted_to_retry_config(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            retry_config={"max_retries": 5},
        )
        assert isinstance(req.retry_config, RetryConfig)
        assert req.retry_config.max_retries == 5

    def test_invalid_type_rejected_at_construction(self):
        """错误在构造时就报，而不是运行时才炸。"""
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[{"role": "user", "content": "x"}],
                retry_config="not-a-config",
            )

    def test_none_still_allowed(self):
        req = LLMRequest(messages=[{"role": "user", "content": "x"}])
        assert req.retry_config is None


# ============================================================
# 8. P2-3：probe_capabilities（原零覆盖）
# ============================================================
class TestProbeCapabilities:
    """能力主动探测——四条分支此前完全没有测试。"""

    @staticmethod
    def make_client() -> OpenAIClient:
        mock = MagicMock()
        c = OpenAIClient.__new__(OpenAIClient)
        c._async_client = mock
        c._fallback_async_client = mock
        c.default_model = "gpt-4o"
        c.fallback_model = None
        c.base_url = "https://api.example.com/v1"
        c.fallback_base_url = "https://api.example.com/v1"
        c.retry_config = None
        c.capability_probe = True
        return c

    async def test_probe_all_success(self):
        """全部探测通过 → 能力齐备。"""
        c = self.make_client()
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"ok": true}'
        c._async_client.chat.completions.create = AsyncMock(return_value=resp)

        changes = await c.probe_capabilities("gpt-4o")

        assert changes["supports_tools"] is True
        assert changes["supports_response_format"] is True
        assert changes["json_schema"] is True
        assert changes["reasoning_effort_supported"] is True
        assert changes["rejects_cn_suffix"] is False

    async def test_probe_thinking_conflicts_with_tools(self):
        """400 提示 tool_choice + thinking 冲突 → 学会该结论。"""
        from openai import BadRequestError

        c = self.make_client()

        def make_400(msg: str) -> BadRequestError:
            request = MagicMock()
            response = MagicMock()
            response.status_code = 400
            response.headers = {}
            return BadRequestError(msg, response=response, body=None)

        c._async_client.chat.completions.create = AsyncMock(
            side_effect=make_400("tool_choice cannot be used with thinking mode")
        )

        changes = await c.probe_capabilities("glm-4")
        assert changes["thinking_conflicts_with_tools"] is True
        assert changes["supports_tools"] is True

    async def test_probe_thinking_param_rejected(self):
        """400 提示 thinking 被拒收 → 学会彻底移除 thinking。"""
        from openai import BadRequestError

        c = self.make_client()

        def make_400(msg: str) -> BadRequestError:
            request = MagicMock()
            response = MagicMock()
            response.status_code = 400
            response.headers = {}
            return BadRequestError(msg, response=response, body=None)

        c._async_client.chat.completions.create = AsyncMock(
            side_effect=make_400("parameter thinking is not supported")
        )

        changes = await c.probe_capabilities("glm-5")
        assert changes["thinking_param_rejected"] is True

    async def test_probe_response_format_unsupported(self):
        """400 提示 response_format 不支持 → 学会该结论。"""
        from openai import BadRequestError

        c = self.make_client()

        def make_400(msg: str) -> BadRequestError:
            request = MagicMock()
            response = MagicMock()
            response.status_code = 400
            response.headers = {}
            return BadRequestError(msg, response=response, body=None)

        c._async_client.chat.completions.create = AsyncMock(
            side_effect=make_400("response_format json_object is not supported by this model")
        )

        changes = await c.probe_capabilities("old-model")
        assert changes["supports_response_format"] is False

    async def test_probe_network_error_returns_empty(self):
        """网络错误 → 返回空字典，绝不抛出（探测失败不影响主流程）。"""
        c = self.make_client()
        c._async_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("network down")
        )

        changes = await c.probe_capabilities("gpt-4o")
        assert changes == {}

    async def test_probe_detects_empty_cn_suffix_response(self):
        """中文 JSON 后缀导致空响应 → 标记 rejects_cn_suffix。"""
        c = self.make_client()
        call_count = {"n": 0}

        async def create(**kwargs: Any) -> Any:
            call_count["n"] += 1
            resp = MagicMock()
            resp.choices = [MagicMock()]
            # 最后一次（中文后缀探测）返回空内容
            resp.choices[0].message.content = "" if call_count["n"] >= 4 else '{"ok": true}'
            return resp

        c._async_client.chat.completions.create = AsyncMock(side_effect=create)

        changes = await c.probe_capabilities("deepseek-flash")
        assert changes["rejects_cn_suffix"] is True


# ============================================================
# 9. O1：total_tokens 构造时回退
# ============================================================
class TestUsageTotalTokensFallback:
    """provider 不返回 total_tokens 时，usage dict 里也应回退——
    否则调用方直接读 ``resp.usage["total_tokens"]`` 会拿到 0。"""

    @staticmethod
    def make_response(usage: Any) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = MagicMock()
        resp.choices[0].message.content = "hi"
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.reasoning_content = None
        resp.choices[0].message.reasoning_details = None
        resp.choices[0].finish_reason = "stop"
        resp.model = "gpt-4o"
        resp.usage = usage
        return resp

    def test_total_falls_back_to_sum(self):
        resp = self.make_response(make_usage(10, 5, 0))
        result = LLMResponse.from_openai_response(resp)
        assert result.usage["total_tokens"] == 15

    def test_total_kept_when_provided(self):
        resp = self.make_response(make_usage(10, 5, 20))
        result = LLMResponse.from_openai_response(resp)
        assert result.usage["total_tokens"] == 20

    def test_chunk_total_falls_back(self):
        """流式 chunk 同样回退。"""
        chunk = make_chunk("x", usage=make_usage(3, 4, 0))
        result = LLMResponse.from_openai_chunk(chunk)
        assert result.usage["total_tokens"] == 7


# ============================================================
# 10. O2：能力参数适配逻辑统一
# ============================================================
class TestAdaptCapabilityParams:
    """四条调用路径（ainvoke/invoke/astream/stream）共用同一份适配逻辑。"""

    def test_thinking_stripped_when_rejected(self):
        r = get_registry()
        r.update("https://api.example.com/v1", "gpt-4o", source="learn", thinking_param_rejected=True)

        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            extra_body={"thinking": {"type": "enabled"}},
        )
        kwargs = OpenAIClient._adapt_capability_params(
            req, req.to_openai_kwargs(), "https://api.example.com/v1", "gpt-4o"
        )
        assert "thinking" not in (kwargs.get("extra_body") or {})

    def test_json_mode_json_object(self):
        req = LLMRequest(messages=[{"role": "user", "content": "x"}], json_mode="json_object")
        kwargs = OpenAIClient._adapt_capability_params(req, req.to_openai_kwargs(), None, "m")
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_json_mode_prompt_only_drops_response_format(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            json_mode="prompt_only",
            response_format={"type": "json_object"},
        )
        kwargs = OpenAIClient._adapt_capability_params(req, req.to_openai_kwargs(), None, "m")
        assert "response_format" not in kwargs

    def test_response_format_dropped_when_unsupported(self):
        """已知端点不支持 response_format 且 json_mode=None → 自动剔除。"""
        r = get_registry()
        r.update("https://api.example.com/v1", "m", source="learn", supports_response_format=False)

        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )
        kwargs = OpenAIClient._adapt_capability_params(
            req, req.to_openai_kwargs(), "https://api.example.com/v1", "m"
        )
        assert "response_format" not in kwargs

    def test_sync_and_async_share_same_logic(self):
        """同步 / 异步走的是同一个方法（不会分叉）。"""
        import inspect

        assert isinstance(
            inspect.getattr_static(OpenAIClient, "_adapt_capability_params"), staticmethod
        )
        ainvoke_src = inspect.getsource(OpenAIClient.ainvoke)
        invoke_src = inspect.getsource(OpenAIClient.invoke)
        astream_src = inspect.getsource(OpenAIClient.astream)
        stream_src = inspect.getsource(OpenAIClient.stream)
        for src in (ainvoke_src, invoke_src, astream_src, stream_src):
            assert "_adapt_capability_params" in src


# ============================================================
# 11. O4：DECISION_FIELDS 应为 ClassVar
# ============================================================
class TestDecisionFieldsIsClassVar:
    def test_not_a_dataclass_field(self):
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(ProviderCapabilities)]
        assert "DECISION_FIELDS" not in field_names

    def test_still_accessible_from_class_and_instance(self):
        assert "supports_tools" in ProviderCapabilities.DECISION_FIELDS
        assert "supports_tools" in ProviderCapabilities().DECISION_FIELDS

    def test_as_dict_unaffected(self):
        d = ProviderCapabilities(supports_tools=True).as_dict()
        assert d["supports_tools"] is True
        assert "DECISION_FIELDS" not in d
