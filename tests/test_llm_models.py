#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm.models 测试

覆盖：
1. LLMRequest 必填 / 默认值
2. LLMRequest 字段验证（temperature/max_tokens/top_p 范围）
3. LLMRequest.to_openai_kwargs 输出
4. LLMResponse 默认值
5. LLMResponse.from_openai_response：纯文本响应
6. LLMResponse.from_openai_response：tool_calls 响应
7. LLMResponse.from_openai_response：usage 为 None
8. LLMResponse.from_openai_response：message.content 为 None
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from infra.llm.models import LLMRequest, LLMResponse


# ============================================================
# 1. LLMRequest 必填 + 默认值
# ============================================================
class TestLLMRequestDefaults:
    def test_minimal_required(self):
        """只 messages 必填。"""
        r = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        assert r.messages == [{"role": "user", "content": "hi"}]
        assert r.model is None
        assert r.temperature == 0.7
        assert r.max_tokens == 4096
        assert r.top_p == 1.0
        assert r.tools is None
        assert r.tool_choice is None
        assert r.response_format is None
        assert r.timeout is None

    def test_missing_messages_raises(self):
        """messages 缺失 → ValidationError。"""
        with pytest.raises(ValidationError):
            LLMRequest()


# ============================================================
# 2. 字段范围验证
# ============================================================
class TestLLMRequestValidation:
    @pytest.mark.parametrize("temp", [-0.1, 2.1, 5.0])
    def test_temperature_out_of_range(self, temp):
        with pytest.raises(ValidationError):
            LLMRequest(messages=[{"role": "user", "content": "x"}], temperature=temp)

    @pytest.mark.parametrize("temp", [0.0, 0.7, 1.0, 2.0])
    def test_temperature_in_range(self, temp):
        r = LLMRequest(messages=[{"role": "user", "content": "x"}], temperature=temp)
        assert r.temperature == temp

    @pytest.mark.parametrize("mt", [0, -1, 100001])
    def test_max_tokens_out_of_range(self, mt):
        with pytest.raises(ValidationError):
            LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=mt)

    @pytest.mark.parametrize("mt", [1, 4096, 100000])
    def test_max_tokens_in_range(self, mt):
        r = LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=mt)
        assert r.max_tokens == mt

    @pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
    def test_top_p_out_of_range(self, p):
        with pytest.raises(ValidationError):
            LLMRequest(messages=[{"role": "user", "content": "x"}], top_p=p)


# ============================================================
# 3. to_openai_kwargs
# ============================================================
class TestToOpenAIKwargs:
    def test_basic_fields(self):
        r = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4",
            temperature=0.5,
        )
        kwargs = r.to_openai_kwargs()
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["model"] == "gpt-4"
        assert kwargs["temperature"] == 0.5

    def test_excludes_none(self):
        """为 None 的字段不进入 kwargs。"""
        r = LLMRequest(messages=[{"role": "user", "content": "x"}])
        kwargs = r.to_openai_kwargs()
        assert "model" not in kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        assert "response_format" not in kwargs
        assert "timeout" not in kwargs

    def test_messages_present(self):
        """messages 始终在输出中。"""
        r = LLMRequest(messages=[{"role": "user", "content": "x"}])
        kwargs = r.to_openai_kwargs()
        assert "messages" in kwargs

    def test_tools_when_provided(self):
        r = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f"}}],
        )
        kwargs = r.to_openai_kwargs()
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "f"}}]


# ============================================================
# 4. LLMResponse 默认值
# ============================================================
class TestLLMResponseDefaults:
    def test_minimal(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.usage == {}
        assert r.model is None
        assert r.finish_reason is None

    def test_with_content(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"

    def test_with_tool_calls(self):
        r = LLMResponse(tool_calls=[{"id": "1", "type": "function", "function": {}}])
        assert len(r.tool_calls) == 1


# ============================================================
# 5-8. from_openai_response（用 mock 对象模拟 OpenAI 响应）
# ============================================================
class _FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id = id_
        self.type = "function"
        self.function = type("F", (), {
            "name": name,
            "arguments": arguments,
        })()


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


class _FakeOpenAIResponse:
    def __init__(self, content=None, tool_calls=None, model="gpt-4",
                 finish_reason="stop", usage=None, reasoning=None,
                 reasoning_details=None):
        self.choices = [_FakeChoice(_FakeMessage(content, tool_calls, reasoning, reasoning_details), finish_reason)]
        self.model = model
        self.usage = usage


class TestFromOpenAIResponse:
    def test_text_only_response(self):
        resp = _FakeOpenAIResponse(
            content="hello world",
            model="gpt-4o",
            finish_reason="stop",
            usage=_FakeUsage(prompt=10, completion=20, total=30),
        )
        r = LLMResponse.from_openai_response(resp)
        assert r.content == "hello world"
        assert r.model == "gpt-4o"
        assert r.finish_reason == "stop"
        assert r.tool_calls == []
        assert r.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_none_content_becomes_empty_string(self):
        """message.content is None → content 字段用 ""（不存 None）。"""
        resp = _FakeOpenAIResponse(content=None)
        r = LLMResponse.from_openai_response(resp)
        assert r.content == ""

    def test_reasoning_content_becomes_content_when_content_is_empty(self):
        resp = _FakeOpenAIResponse(
            content=None,
            reasoning="这是正文",
        )
        r = LLMResponse.from_openai_response(resp)
        assert r.content == "这是正文"
        assert r.reasoning_content == "这是正文"

    def test_reasoning_details_become_content_when_content_is_empty(self):
        resp = _FakeOpenAIResponse(
            content=None,
            reasoning_details=[
                {"type": "reasoning.text", "text": "第一部分"},
                {"type": "reasoning.text", "text": "第二部分"},
            ],
        )
        r = LLMResponse.from_openai_response(resp)
        assert r.content == "第一部分\n第二部分"
        assert r.reasoning_content == "第一部分\n第二部分"

    def test_text_content_takes_precedence_over_reasoning(self):
        resp = _FakeOpenAIResponse(
            content="最终正文",
            reasoning="推理内容",
        )
        r = LLMResponse.from_openai_response(resp)
        assert r.content == "最终正文"
        assert r.reasoning_content == "推理内容"

    def test_response_without_usage(self):
        """response.usage = None → usage = {}。"""
        resp = _FakeOpenAIResponse(content="x", usage=None)
        r = LLMResponse.from_openai_response(resp)
        assert r.usage == {}

    def test_usage_with_zero_tokens(self):
        resp = _FakeOpenAIResponse(
            content="x",
            usage=_FakeUsage(prompt=0, completion=0, total=0),
        )
        r = LLMResponse.from_openai_response(resp)
        assert r.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_response_with_tool_calls(self):
        tool_calls = [
            _FakeToolCall("call_1", "get_weather", '{"city": "Beijing"}'),
            _FakeToolCall("call_2", "get_time", '{"tz": "UTC"}'),
        ]
        resp = _FakeOpenAIResponse(content=None, tool_calls=tool_calls)
        r = LLMResponse.from_openai_response(resp)
        assert len(r.tool_calls) == 2
        assert r.tool_calls[0] == {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Beijing"}',
            },
        }
        assert r.tool_calls[1]["function"]["name"] == "get_time"

    def test_response_model_falls_back_to_none(self):
        """response 没有 model 属性 → None。"""
        resp = _FakeOpenAIResponse(content="x")
        # 移除 model 属性
        del resp.model
        r = LLMResponse.from_openai_response(resp)
        assert r.model is None

    def test_finish_reason_attribute_missing(self):
        """choices[0] 没有 finish_reason → None。"""
        resp = _FakeOpenAIResponse(content="x")
        delattr(resp.choices[0], "finish_reason")
        r = LLMResponse.from_openai_response(resp)
        assert r.finish_reason is None


# ============================================================
# 9. Pydantic v2 模型行为
# ============================================================
class TestPydanticBehavior:
    def test_request_immutable_dict(self):
        """model_dump 返回 plain dict（非 BaseModel 自身）。"""
        r = LLMRequest(messages=[{"role": "user", "content": "x"}])
        d = r.model_dump()
        assert isinstance(d, dict)
        assert d["messages"] == [{"role": "user", "content": "x"}]

    def test_response_default_factory_each_instance(self):
        """tool_calls 和 usage 默认值不共享。"""
        a = LLMResponse()
        b = LLMResponse()
        a.tool_calls.append({"x": 1})
        a.usage["k"] = 1
        assert b.tool_calls == []
        assert b.usage == {}


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
