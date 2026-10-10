#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm usage_callback 单元测试

覆盖：
1. OpenAIClient.ainvoke — usage_callback 在成功时调用，含 token 字段
2. OpenAIClient.ainvoke — 响应无 usage 时不调用回调
3. OpenAIClient.ainvoke — 回调抛出异常不向上抛
4. OpenAIClient.ainvoke — 不传 usage_callback 行为不变
5. OpenAIClient.invoke (sync) — 同 ainvoke
6. OpenAIClient.astream — 流结束时聚合 usage + 回调一次
7. OpenAIClient.stream (sync) — 同 astream
8. OpenAIMMediaClient.generate_image — 回调收到 image info
9. OpenAIMMediaClient.synthesize_audio — 回调收到 audio info
10. OpenIAMMediaClient.generate_video — 回调收到 pending 状态
11. OpenIAMMediaClient.poll_video — 完成时回调收到 duration + completed
"""
from __future__ import annotations

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from infra.llm import LLMRequest, LLMResponse, MediaRequest, MediaResponse
from infra.llm.client import OpenAIClient, OpenAIMediaClient
from infra.llm.models import ImageResult, AudioResult, VideoResult


# ============================================================
# Helpers
# ============================================================

def _make_openai_client() -> OpenAIClient:
    """创建 OpenAIClient，用虚拟 key（不会真连网）。"""
    return OpenAIClient(
        api_key="sk-test",
        base_url="http://localhost:8000",
        model="gpt-4o",
        max_retries=1,
        timeout=10.0,
    )


def _make_media_client() -> OpenAIMediaClient:
    """创建 OpenAIMediaClient，用虚拟 key。"""
    return OpenAIMediaClient(
        api_key="sk-test",
        base_url="http://localhost:8000",
        image_model="dall-e-3",
        video_model="video-gen",
        audio_model="tts-1",
        max_retries=1,
        timeout=10.0,
        provider="openai",
    )


def _make_fake_openai_response(content: str = "hello", usage: dict | None = None, model: str = "gpt-4o") -> MagicMock:
    """构造 fake OpenAI chat.completions.create 返回对象。"""
    msg = MagicMock(spec=[])
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None
    msg.reasoning = None
    msg.reasoning_details = None

    choice = MagicMock(spec=[])
    choice.message = msg
    choice.finish_reason = "stop"

    resp = MagicMock(spec=[])
    resp.choices = [choice]
    resp.model = model
    if usage is not None:
        u = MagicMock(spec=[])
        u.prompt_tokens = usage.get("prompt_tokens", 0)
        u.completion_tokens = usage.get("completion_tokens", 0)
        u.total_tokens = usage.get("total_tokens", 0)
        resp.usage = u
    else:
        resp.usage = None
    return resp


def _make_fake_stream_chunks(content_chunks: List[str], usage: dict) -> List[MagicMock]:
    """构造 fake OpenAI stream chunk 列表。最后一个 chunk 带 usage。"""
    chunks = []
    for i, text in enumerate(content_chunks):
        delta = MagicMock(spec=[])
        delta.content = text
        delta.tool_calls = None
        delta.reasoning_content = None
        delta.reasoning = None
        delta.reasoning_details = None

        choice = MagicMock(spec=[])
        choice.delta = delta
        choice.finish_reason = "stop" if i == len(content_chunks) - 1 else None

        chunk = MagicMock(spec=[])
        chunk.choices = [choice]
        chunk.model = "gpt-4o"
        chunk.usage = None  # only last chunk has usage
        chunks.append(chunk)

    # 最后一个 chunk 带 usage
    last = chunks[-1]
    u = MagicMock(spec=[])
    u.prompt_tokens = usage.get("prompt_tokens", 0)
    u.completion_tokens = usage.get("completion_tokens", 0)
    u.total_tokens = usage.get("total_tokens", 0)
    last.usage = u
    return chunks


# ============================================================
# 1. OpenAIClient.ainvoke — callback called with tokens on success
# ============================================================
class TestAinvokeUsageCallback:
    @pytest.mark.asyncio
    async def test_callback_called_on_success(self):
        client = _make_openai_client()
        fake_resp = _make_fake_openai_response(
            content="hello", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        client._async_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        calls: list[dict] = []
        callback = lambda info: calls.append(info)  # noqa: E731

        await client.ainvoke(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=callback,
        )

        assert len(calls) == 1
        info = calls[0]
        assert info["modality"] == "text"
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o"
        assert info["prompt_tokens"] == 10
        assert info["completion_tokens"] == 20
        assert info["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_callback_not_called_when_no_usage(self):
        client = _make_openai_client()
        fake_resp = _make_fake_openai_response(content="hello", usage=None)
        client._async_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        await client.ainvoke(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=callback,
        )
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_swallowed(self):
        client = _make_openai_client()
        fake_resp = _make_fake_openai_response(
            content="hello", usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        )
        client._async_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        def bad_callback(info):
            raise RuntimeError("boom")

        # Should not raise — exception is swallowed
        result = await client.ainvoke(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=bad_callback,
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_without_callback_unchanged(self):
        """不传 usage_callback —— 进程内聚合仍记录，行为不变。"""
        client = _make_openai_client()
        fake_resp = _make_fake_openai_response(
            content="hello", usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
        )
        client._async_client.chat.completions.create = AsyncMock(return_value=fake_resp)

        from infra.llm import get_usage_summary, reset_usage_summary
        reset_usage_summary()

        result = await client.ainvoke(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
        )
        assert result.content == "hello"

        summary = get_usage_summary()
        assert "unknown" in summary
        assert summary["unknown"]["total_tokens"] == 10

        reset_usage_summary()


# ============================================================
# 2. OpenAIClient.invoke (sync) — callback called
# ============================================================
class TestInvokeUsageCallback:
    def test_sync_callback_called(self):
        client = _make_openai_client()
        fake_resp = _make_fake_openai_response(
            content="ack", usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
        )
        client._sync_client.chat.completions.create = MagicMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        result = client.invoke(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=callback,
        )
        assert result.content == "ack"
        assert len(calls) == 1
        assert calls[0]["total_tokens"] == 5

    def test_sync_callback_exception_swallowed(self):
        client = _make_openai_client()
        fake_resp = _make_fake_openai_response(
            content="ack", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        )
        client._sync_client.chat.completions.create = MagicMock(return_value=fake_resp)

        def bad_callback(info):
            raise ValueError("callback boom")

        result = client.invoke(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=bad_callback,
        )
        assert result.content == "ack"


# ============================================================
# 3. OpenAIClient.astream — callback at end with aggregated usage
# ============================================================
class TestAstreamUsageCallback:
    @pytest.mark.asyncio
    async def test_astream_callback_called_once_at_end(self):
        client = _make_openai_client()
        chunks = _make_fake_stream_chunks(["h", "i"], {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12})

        class _FakeAsyncStream:
            def __init__(self, items):
                self._items = list(items)
            def __aiter__(self):
                async def _gen():
                    for item in self._items:
                        yield item
                return _gen()
            async def aclose(self):
                pass

        client._async_client.chat.completions.create = AsyncMock(
            return_value=_FakeAsyncStream(chunks)
        )

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        collected = []
        async for resp in client.astream(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=callback,
        ):
            collected.append(resp)

        assert any(r.content for r in collected)
        assert len(calls) == 1
        info = calls[0]
        assert info["modality"] == "text"
        assert info["prompt_tokens"] == 8
        assert info["total_tokens"] == 12


# ============================================================
# 4. OpenAIClient.stream (sync) — callback at end
# ============================================================
class TestStreamUsageCallback:
    def test_stream_callback_called_once_at_end(self):
        client = _make_openai_client()
        chunks = _make_fake_stream_chunks(["h", "i"], {"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9})

        sync_iter = MagicMock()
        sync_iter.__iter__ = MagicMock(return_value=iter(chunks))
        client._sync_client.chat.completions.create = MagicMock(return_value=sync_iter)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        collected = list(client.stream(
            LLMRequest(messages=[{"role": "user", "content": "hi"}]),
            usage_callback=callback,
        ))
        assert any(r.content for r in collected)
        assert len(calls) == 1
        assert calls[0]["prompt_tokens"] == 6
        assert calls[0]["total_tokens"] == 9


# ============================================================
# 5. OpenAIMMediaClient.generate_image — callback with image info
# ============================================================
class TestMediaImageCallback:
    @pytest.mark.asyncio
    async def test_image_callback_called(self):
        media = _make_media_client()
        fake_resp = MagicMock(spec=[])
        fake_item = MagicMock(spec=[])
        fake_item.url = "https://example.com/img.png"
        fake_item.b64_json = None
        fake_item.revised_prompt = "a cat"
        fake_resp.data = [fake_item]
        media._async_client.images.generate = AsyncMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        req = MediaRequest(
            prompt="a cat",
            model="dall-e-3",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        resp = await media.generate_image(req, usage_callback=callback)
        assert resp.modality == "image"
        assert len(calls) == 1
        info = calls[0]
        assert info["modality"] == "image"
        assert info["image_count"] == 1
        assert info["size"] == "1024x1024"
        assert info["quality"] == "standard"

    def test_image_sync_callback_called(self):
        media = _make_media_client()
        fake_resp = MagicMock(spec=[])
        fake_item = MagicMock(spec=[])
        fake_item.url = "https://example.com/img.png"
        fake_item.b64_json = None
        fake_item.revised_prompt = None
        fake_resp.data = [fake_item]
        media._sync_client.images.generate = MagicMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        req = MediaRequest(prompt="a cat", model="dall-e-3", n=1)
        resp = media.generate_image_sync(req, usage_callback=callback)
        assert resp.modality == "image"
        assert len(calls) == 1
        assert calls[0]["modality"] == "image"
        assert calls[0]["image_count"] == 1


# ============================================================
# 6. OpenAIMMediaClient.synthesize_audio — callback with input_chars
# ============================================================
class TestMediaAudioCallback:
    @pytest.mark.asyncio
    async def test_audio_callback_called(self):
        media = _make_media_client()
        fake_resp = MagicMock(spec=[])
        fake_resp.url = "https://example.com/audio.mp3"
        media._async_client.audio.speech.create = AsyncMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        req = MediaRequest(prompt="hello world", model="tts-1")
        resp = await media.synthesize_audio(req, usage_callback=callback)
        assert resp.modality == "audio"
        assert len(calls) == 1
        info = calls[0]
        assert info["modality"] == "audio"
        assert info["input_chars"] == 11


# ============================================================
# 7. OpenIAMMediaClient.generate_video — callback with pending status
# ============================================================
class TestMediaVideoCallback:
    @pytest.mark.asyncio
    async def test_video_generate_callback_pending(self):
        media = _make_media_client()
        fake_resp = MagicMock(spec=[])
        fake_resp.id = "task-123"
        media._async_client = MagicMock()
        media._async_client.video.generate = AsyncMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        req = MediaRequest(prompt="a dancing cat", model="video-gen")
        resp = await media.generate_video(req, usage_callback=callback)
        assert resp.modality == "video"
        assert len(calls) == 1
        info = calls[0]
        assert info["modality"] == "video"
        assert info["video_status"] == "pending"

    @pytest.mark.asyncio
    async def test_poll_video_callback_completed(self):
        media = _make_media_client()
        fake_resp = MagicMock(spec=[])
        fake_resp.status = "completed"
        fake_resp.video_url = "https://example.com/video.mp4"
        fake_resp.result_url = None
        fake_resp.duration_seconds = 10.5
        fake_resp.completed = True
        fake_resp.task_id = "task-abc"

        media._async_client = MagicMock()
        media._async_client.video.tasks.retrieve = AsyncMock(return_value=fake_resp)

        calls: list = []
        callback = lambda info: calls.append(info)  # noqa: E731

        resp = await media.poll_video("task-abc", usage_callback=callback)
        assert resp.modality == "video"
        assert resp.videos[0].status == "completed"
        assert len(calls) == 1
        info = calls[0]
        assert info["video_status"] == "completed"
        assert info["video_duration_seconds"] == 10.5
