#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm.postprocess 测试

覆盖：
1. is_likely_cot：CoT 检测（正例 / 负例 / 边界）
2. strip_cot_after_answer：final answer 提取
3. strip_cot_with_fallback：激进清洗
4. make_cot_aware_processor：工厂函数
5. LLMRequest.post_processor 字段存在与 to_openai_kwargs 排除
6. LLMResponse.from_openai_response 应用 post_processor
7. ProviderCapabilities.leaks_cot 字段
8. ProviderCapabilityRegistry.learn_from_successful_response
"""

from __future__ import annotations

from typing import Optional

import pytest

from infra.llm.models import LLMRequest, LLMResponse
from infra.llm.postprocess import (
    is_likely_cot,
    strip_cot_after_answer,
    strip_cot_with_fallback,
    make_cot_aware_processor,
    default_cot_processor,
    aggressive_cot_processor,
)
from infra.llm.capabilities import (
    ProviderCapabilities,
    ProviderCapabilityRegistry,
    get_registry,
    reset_registry,
)


# ============================================================
# 1. is_likely_cot
# ============================================================
class TestIsLikelyCot:
    def test_short_content_not_cot(self):
        """短的正常回答 → False。"""
        assert is_likely_cot("Hello, how can I help you today?") is False

    def test_empty_content(self):
        assert is_likely_cot("") is False
        assert is_likely_cot("   ") is False

    def test_long_content_with_cot_markers_and_no_reasoning(self):
        """长 + CoT 标记 + 无 reasoning_content → True。"""
        cot = "We need to think step by step. The answer is: hello. "
        cot = cot * 20  # 扩长 > 400 chars
        assert len(cot) > 400
        assert is_likely_cot(cot, reasoning_content=None) is True

    def test_long_content_with_cot_start_pattern(self):
        """以 'Let me think' 开头的 CoT → True。"""
        cot = "Let me think about this step by step.\nWe need to consider..."
        cot = cot * 30  # 扩长
        assert len(cot) > 400
        assert is_likely_cot(cot) is True

    def test_short_cot_start_pattern(self):
        """短内容但以 CoT 模式开头 → True。"""
        assert is_likely_cot("Let me think about this...") is True
        assert is_likely_cot("We need to generate a response...") is True

    def test_content_with_reasoning_not_cot(self):
        """即使 content 较长，但 reasoning_content 存在 → 可能不是 CoT 泄露（推理型模型）。"""
        content = "x" * 500
        reasoning = "Here is my thinking process..."
        assert is_likely_cot(content, reasoning) is False

    def test_final_answer_marker_not_cot(self):
        """以 final answer 标记结尾的短内容 → False。"""
        content = "Final answer: The result is 42."
        assert is_likely_cot(content) is False


# ============================================================
# 2. strip_cot_after_answer
# ============================================================
class TestStripCotAfterAnswer:
    def test_normal_content_unchanged(self):
        content = "Hello, this is a normal response."
        assert strip_cot_after_answer(content) == content

    def test_cot_with_final_answer_extracted(self):
        cot = (
            "We need to think step by step. Let me consider the options... " * 20
        ) + "\n\nFinal answer: #MacroMonitor v0.1 {run_123}"
        assert is_likely_cot(cot) is True
        result = strip_cot_after_answer(cot)
        assert result == "#MacroMonitor v0.1 {run_123}"

    def test_cot_no_final_answer_marker_returns_original(self):
        """CoT 但没有 final answer 标记 → 原样返回（保守）。"""
        cot = ("We need to think step by step..." * 20)
        result = strip_cot_after_answer(cot)
        assert result == cot

    def test_empty_content_with_reasoning(self):
        result = strip_cot_after_answer("", reasoning_content="thinking content")
        assert result == "thinking content"

    def test_empty_content_no_reasoning(self):
        result = strip_cot_after_answer("")
        assert result == ""


# ============================================================
# 3. strip_cot_with_fallback
# ============================================================
class TestStripCotWithFallback:
    def test_cot_with_final_answer(self):
        cot = ("We need to think step by step..." * 20) + "\n\nFinal answer: hello"
        result = strip_cot_with_fallback(cot)
        assert result == "hello"

    def test_cot_without_final_answer_uses_reasoning(self):
        cot = ("We need to think step by step..." * 20)
        reasoning = "The final answer is: hello"
        result = strip_cot_with_fallback(cot, reasoning_content=reasoning)
        assert result == "The final answer is: hello"

    def test_cot_without_final_answer_no_reasoning(self):
        cot = ("We need to think step by step... " * 20)
        result = strip_cot_with_fallback(cot, reasoning_content=None)
        assert result == cot


# ============================================================
# 4. make_cot_aware_processor
# ============================================================
class TestMakeCotAwareProcessor:
    def test_default_aggressive_false(self):
        processor = make_cot_aware_processor(aggressive=False)
        content = "Hello world"
        assert processor(content) == content

    def test_aggressive_uses_min_length(self):
        processor = make_cot_aware_processor(aggressive=False, min_length=50)
        content = "x" * 30
        assert processor(content) == content  # 短于阈值，原样返回

    def test_processor_extracts_final_answer(self):
        processor = make_cot_aware_processor(aggressive=True)
        cot = ("We need to think step by step..." * 30) + "Final answer: extracted"
        result = processor(cot, "reasoning")
        assert result == "extracted"


# ============================================================
# 5. default_cot_processor / aggressive_cot_processor aliases
# ============================================================
class TestProcessorAliases:
    def test_default_is_strip_after_answer(self):
        assert default_cot_processor is strip_cot_after_answer

    def test_aggressive_is_strip_with_fallback(self):
        assert aggressive_cot_processor is strip_cot_with_fallback


# ============================================================
# 6. LLMRequest.post_processor field
# ============================================================
class TestLLMRequestPostProcessor:
    def test_field_exists_and_defaults_none(self):
        r = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        assert r.post_processor is None

    def test_post_processor_accepted(self):
        def my_proc(content: str, reasoning_content: Optional[str] = None) -> str:
            return content.strip()
        r = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            post_processor=my_proc,
        )
        assert r.post_processor is my_proc

    def test_to_openai_kwargs_excludes_post_processor(self):
        """post_processor 不应传入 OpenAI API。"""
        def my_proc(content: str, reasoning_content=None) -> str:
            return content
        r = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            post_processor=my_proc,
        )
        kwargs = r.to_openai_kwargs()
        assert "post_processor" not in kwargs

    def test_to_openai_kwargs_excludes_post_processor_when_none(self):
        r = LLMRequest(messages=[{"role": "user", "content": "x"}])
        kwargs = r.to_openai_kwargs()
        assert "post_processor" not in kwargs

    def test_model_dump_excludes_callable(self):
        """model_dump 不应失败（Callable 字段可 dump 但值可能为 None）。"""
        def my_proc(content: str, reasoning=None) -> str:
            return content
        r = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            post_processor=my_proc,
        )
        d = r.model_dump()
        assert "post_processor" in d  # dump 包含字段
        assert "post_processor" not in r.to_openai_kwargs()  # 但 kwargs 排除


# ============================================================
# 7. LLMResponse.from_openai_response applies post_processor
# ============================================================
class _FakeMessage:
    def __init__(self, content=None, tool_calls=None, reasoning=None, reasoning_details=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning
        self.reasoning_details = reasoning_details or []


class _FakeChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt=0, completion=0, total=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _FakeResponse:
    def __init__(self, content=None, model="gpt-4", usage=None, reasoning=None,
                 reasoning_details=None):
        self.choices = [_FakeChoice(_FakeMessage(content, reasoning=reasoning,
                                   reasoning_details=reasoning_details), "stop")]
        self.model = model
        self.usage = usage


class TestFromOpenAIResponseWithPostProcessor:
    def test_no_post_processor(self):
        resp = _FakeResponse(content="hello world")
        r = LLMResponse.from_openai_response(resp)
        assert r.content == "hello world"

    def test_post_processor_applied(self):
        cot_content = ("We need to think step by step. " * 20) + "Final answer: cleaned"
        resp = _FakeResponse(content=cot_content)
        r = LLMResponse.from_openai_response(resp, post_processor=strip_cot_after_answer)
        assert r.content == "cleaned"

    def test_post_processor_receives_reasoning(self):
        resp = _FakeResponse(content="text", reasoning="reasoning text")
        received = {}
        def proc(content: str, reasoning_content=None) -> str:
            received["content"] = content
            received["reasoning"] = reasoning_content
            return content
        r = LLMResponse.from_openai_response(resp, post_processor=proc)
        assert received["content"] == "text"
        assert received["reasoning"] == "reasoning text"


# ============================================================
# 8. from_openai_chunk applies post_processor
# ============================================================
class _FakeDelta:
    def __init__(self, content=None, reasoning=None, tool_calls=None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = tool_calls or []


class _FakeChunkChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeChunk:
    def __init__(self, content=None, reasoning=None, model="gpt-4", usage=None):
        self.choices = [_FakeChunkChoice(_FakeDelta(content, reasoning=reasoning))]
        self.model = model
        self.usage = usage


class TestFromOpenAIChunkWithPostProcessor:
    def test_post_processor_applied(self):
        chunk = _FakeChunk(content=("We need to think step by step. " * 50) + "Final answer: hi")
        r = LLMResponse.from_openai_chunk(chunk, post_processor=strip_cot_after_answer)
        assert r.content == "hi"

    def test_no_post_processor_preserves_content(self):
        chunk = _FakeChunk(content="hello")
        r = LLMResponse.from_openai_chunk(chunk)
        assert r.content == "hello"


# ============================================================
# 9. ProviderCapabilities.leaks_cot field
# ============================================================
class TestCapabilitiesLeaksCot:
    def test_leaks_cot_defaults_none(self):
        caps = ProviderCapabilities()
        assert caps.leaks_cot is None

    def test_leaks_cot_can_be_set(self):
        caps = ProviderCapabilities(leaks_cot=True)
        assert caps.leaks_cot is True

    def test_decision_fields_includes_leaks_cot(self):
        assert "leaks_cot" in ProviderCapabilities.DECISION_FIELDS


# ============================================================
# 10. ProviderCapabilityRegistry.learn_from_successful_response
# ============================================================
class TestLearnFromSuccessfulResponse:
    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_detects_cot_and_sets_leaks_cot(self):
        reg = get_registry()
        cot = ("Let me think step by step..." * 20)
        assert len(cot) > 400
        caps = reg.learn_from_successful_response(
            "https://openrouter.ai/api/v1",
            "cohere/north-mini-code:free",
            content=cot,
            reasoning_content=None,
        )
        assert caps.leaks_cot is True

    def test_clean_content_does_not_set_leaks_cot(self):
        reg = get_registry()
        reg.learn_from_successful_response(
            "https://openrouter.ai/api/v1",
            "nex-agi/nex-n2.5-mini:free",
            content="Hello world",
            reasoning_content=None,
        )
        caps = reg.get("https://openrouter.ai/api/v1", "nex-agi/nex-n2.5-mini:free")
        assert caps.leaks_cot is None

    def test_persisted_in_registry(self):
        reg = get_registry()
        cot = ("We need to think..." * 20)
        reg.learn_from_successful_response(
            "https://api.example.com/v1",
            "leaky-model",
            content=cot,
        )
        caps = reg.get("https://api.example.com/v1", "leaky-model")
        assert caps.leaks_cot is True
        assert caps.source == "learn"


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
