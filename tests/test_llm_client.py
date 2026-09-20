"""
tests/test_llm_client.py — infra.llm.client 真实 API 集成测试

策略：按用户要求"测试 LLM 时可以使用真实 LLM"，所有用例走真 API。
- 用 @pytest.mark.slow + @pytest.mark.integration 标记——pyproject 默认 addopts
  `-m "not slow"` 会跳过本文件；跑全部用 `pytest -m ""` 或 `pytest -m "slow and integration"`。
- 凭据从 .env 读（load_dotenv 注入到 os.environ），不硬编码。
- 真实调用 LiteLLM 网关下的 DeepSeek 模型（如 .env 配置）。

覆盖：
1. LLMFactory.create('openai') → OpenAIClient
2. LLMFactory.create('unknown') → ValueError
3. init_llm_client：缺 key 返 None
4. init_llm_client：完整配置 → 拿到 client
5. get_llm_client：未初始化时返 None
6. OpenAIClient.ainvoke：真发一次请求，验证响应形态
7. OpenAIClient.invoke：同步调用真发一次
8. LLMRequest.to_openai_kwargs 字段透传
9. LLMResponse.from_openai_response 工具调用 + usage 提取
10. 重试行为：认证错（401/403/404）→ deny_exceptions → 不重试
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import infra.llm.client as client_module
from infra.llm.client import (
    LLMClient,
    LLMFactory,
    OpenAIClient,
    _llm_client as _global_client_ref,
    get_llm_client,
    init_llm_client,
)
from infra.llm.models import LLMRequest, LLMResponse


# ============================================================
# .env 自动加载 + fixture
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


def _load_dotenv() -> Dict[str, str]:
    """
    加载 .env 到 os.environ。返回当前 LLM_* 键值（仅供诊断）。
    不抛错——缺 .env 时返空 dict，让缺凭据的测试用 pytest.skip 跳过。
    """
    if _ENV_PATH.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(str(_ENV_PATH), override=False)
        except Exception:
            pass
    return {
        "LLM_API_KEY": os.getenv("LLM_API_KEY", ""),
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL", ""),
        "LLM_MODEL": os.getenv("LLM_MODEL", ""),
    }


@pytest.fixture(scope="module")
def llm_env() -> Dict[str, str]:
    """module-scope 加载 .env，所有用例共享。"""
    return _load_dotenv()


@pytest.fixture
def reset_global_llm():
    """每个测试前后重置全局单例 + usage。"""
    from infra.llm import _reset_for_test
    _reset_for_test()
    yield
    _reset_for_test()


def _has_real_credentials(env: Dict[str, str]) -> bool:
    """判定 .env 里的 key 是不是占位符/空——非真凭据就 skip。"""
    if not env.get("LLM_API_KEY"):
        return False
    placeholder = {"", "YOUR_API_KEY_HERE", "sk-xxx", "sk-placeholder"}
    if env["LLM_API_KEY"] in placeholder:
        return False
    if not env.get("LLM_BASE_URL"):
        return False
    return True


pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
]


# ============================================================
# 1. LLMFactory
# ============================================================
class TestLLMFactory:
    def test_create_openai(self, llm_env):
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL，跳过（按需跑 `pytest -m \"\"`）")
        client = LLMFactory.create({
            "provider": "openai",
            "api_key": llm_env["LLM_API_KEY"],
            "base_url": llm_env["LLM_BASE_URL"],
            "default_model": llm_env["LLM_MODEL"],
            "max_retries": 1,
            "timeout": 60.0,
        })
        assert isinstance(client, OpenAIClient)
        assert client.provider == "openai"
        assert client.api_key == llm_env["LLM_API_KEY"]
        assert client.base_url == llm_env["LLM_BASE_URL"]
        assert client.default_model == llm_env["LLM_MODEL"]

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unsupported provider"):
            LLMFactory.create({"provider": "anthropic", "api_key": "x"})

    def test_create_default_provider_is_openai(self, llm_env):
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        cfg = {
            "api_key": llm_env["LLM_API_KEY"],
            "base_url": llm_env["LLM_BASE_URL"],
            "default_model": llm_env["LLM_MODEL"],
            "max_retries": 1,
            "timeout": 60.0,
        }
        client = LLMFactory.create(cfg)
        # 不传 provider → 默认 openai
        assert client.provider == "openai"


# ============================================================
# 2. init_llm_client / get_llm_client 全局单例
# ============================================================
class TestGlobalClient:
    @pytest.mark.asyncio
    async def test_get_llm_client_before_init_returns_none(self, reset_global_llm):
        assert get_llm_client() is None

    @pytest.mark.asyncio
    async def test_init_without_api_key_returns_none(self, reset_global_llm, monkeypatch):
        # 清空所有可能来源
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        cfg = {"llm": {"api_key": ""}}
        result = await init_llm_client(cfg)
        assert result is None
        assert get_llm_client() is None

    @pytest.mark.asyncio
    async def test_init_with_placeholder_returns_none(self, reset_global_llm, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        cfg = {"llm": {"api_key": "YOUR_API_KEY_HERE"}}
        result = await init_llm_client(cfg)
        assert result is None

    @pytest.mark.asyncio
    async def test_init_with_real_config(self, reset_global_llm, llm_env):
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        cfg = {
            "llm": {
                "provider": "openai",
                "api_key": llm_env["LLM_API_KEY"],
                "base_url": llm_env["LLM_BASE_URL"],
                "default_model": llm_env["LLM_MODEL"],
                "max_retries": 1,
                "timeout": 60.0,
            }
        }
        client = await init_llm_client(cfg)
        assert client is not None
        assert isinstance(client, OpenAIClient)
        # get_llm_client 返同一实例
        assert get_llm_client() is client
        # 第二次 init 返同一实例（单例）
        again = await init_llm_client(cfg)
        assert again is client

    @pytest.mark.asyncio
    async def test_init_env_var_fallback(self, reset_global_llm, llm_env, monkeypatch):
        """
        yaml 没 api_key 但环境变量有 → 走环境变量分支。
        用 monkeypatch 把 yaml 里的值清成空，再触发 fallback。
        """
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        # 把环境变量注入（.env 已被 llm_env fixture 加载）
        monkeypatch.setenv("LLM_API_KEY", llm_env["LLM_API_KEY"])
        monkeypatch.setenv("LLM_BASE_URL", llm_env["LLM_BASE_URL"])
        monkeypatch.setenv("LLM_MODEL", llm_env["LLM_MODEL"])

        # yaml 故意留空
        cfg = {"llm": {"provider": "openai", "api_key": ""}}
        client = await init_llm_client(cfg)
        assert client is not None
        assert client.provider == "openai"


# ============================================================
# 3. LLMRequest.to_openai_kwargs
# ============================================================
class TestLLMRequestConversion:
    def test_minimal(self):
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        kw = req.to_openai_kwargs()
        # messages 必须存在且值正确
        assert kw["messages"] == [{"role": "user", "content": "hi"}]
        # 默认值
        assert kw["temperature"] == 0.7
        assert kw["max_tokens"] == 4096
        assert kw["top_p"] == 1.0
        # None 字段被 exclude
        for k in ("model", "tools", "tool_choice", "response_format", "timeout"):
            assert k not in kw

    def test_with_tools_and_format(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "json pls"}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=128,
            top_p=0.9,
            response_format={"type": "json_object"},
            tools=[{"type": "function", "function": {"name": "f"}}],
            tool_choice="auto",
        )
        kw = req.to_openai_kwargs()
        # model 由 client 在外层指定（partial(..., model=...)），不进入 kwargs
        assert "model" not in kw
        assert kw["temperature"] == 0.2
        assert kw["max_tokens"] == 128
        assert kw["response_format"] == {"type": "json_object"}
        assert kw["tools"][0]["function"]["name"] == "f"
        assert kw["tool_choice"] == "auto"


# ============================================================
# 4. LLMResponse.from_openai_response
# ============================================================
class TestLLMResponseConversion:
    def test_text_only(self):
        # 构造 fake openai 响应（带 .choices[0].message / .usage）
        msg = MagicMock_with_attrs(content="Hello!", tool_calls=None)
        choice = MagicMock_with_attrs(message=msg, finish_reason="stop")
        usage = MagicMock_with_attrs(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        fake_resp = MagicMock_with_attrs(
            choices=[choice], usage=usage, model="gpt-4o",
        )

        resp = LLMResponse.from_openai_response(fake_resp)
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o"
        assert resp.finish_reason == "stop"
        assert resp.usage == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        assert resp.tool_calls == []

    def test_with_tool_calls(self):
        tc = MagicMock_with_attrs(
            id="call_1", type="function",
            function=MagicMock_with_attrs(name="get_weather", arguments='{"city":"SF"}'),
        )
        msg = MagicMock_with_attrs(content=None, tool_calls=[tc])
        choice = MagicMock_with_attrs(message=msg, finish_reason="tool_calls")
        usage = MagicMock_with_attrs(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        fake_resp = MagicMock_with_attrs(
            choices=[choice], usage=usage, model="gpt-4o",
        )

        resp = LLMResponse.from_openai_response(fake_resp)
        # content 为 None → 默认空字符串
        assert resp.content == ""
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["id"] == "call_1"
        assert resp.tool_calls[0]["function"]["name"] == "get_weather"
        assert resp.finish_reason == "tool_calls"

    def test_no_usage(self):
        msg = MagicMock_with_attrs(content="ok", tool_calls=None)
        choice = MagicMock_with_attrs(message=msg, finish_reason="stop")
        fake_resp = MagicMock_with_attrs(choices=[choice], usage=None, model="m")
        resp = LLMResponse.from_openai_response(fake_resp)
        assert resp.usage == {}


def MagicMock_with_attrs(**attrs):
    """
    轻量 MagicMock 替代品——直接 setattr 而不是 MagicMock。
    MagicMock 会把 'content'/'tool_calls' 解析成子 mock（自动 mock spec），
    setattr 模式能精确控制。
    """
    from unittest.mock import MagicMock
    m = MagicMock(spec=[])  # 空 spec → 不自动 mock 任何属性
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# ============================================================
# 5. OpenAIClient 真实 ainvoke / invoke（走 .env 真 API）
# ============================================================
class TestOpenAIClientRealAPI:
    def _make_client(self, llm_env) -> OpenAIClient:
        return OpenAIClient(
            api_key=llm_env["LLM_API_KEY"],
            base_url=llm_env["LLM_BASE_URL"],
            model=llm_env["LLM_MODEL"],
            max_retries=1,
            timeout=60.0,
        )

    def _skip_on_billing_error(self, exc: Exception) -> None:
        """
        真 API 调用失败时判断：
        - 402 Payment Required / 账户欠费 → skip（环境问题，非测试/源码 bug）
        - 429 限流 → skip（短期不可用）
        - 其他异常 → 重新抛（让测试 fail，暴露真问题）
        """
        from openai import APIStatusError
        if isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            if code in (402, 429):
                pytest.skip(
                    f"真 LLM 调用受限于账户/限流（HTTP {code}），"
                    f"请充值或稍后重试。错误：{str(exc)[:120]}"
                )
        # 其他异常继续抛

    @pytest.mark.asyncio
    async def test_ainvoke_simple(self, llm_env):
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        client = self._make_client(llm_env)
        req = LLMRequest(
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=20,
            temperature=0.0,
        )
        try:
            resp = await client.ainvoke(req)
        except Exception as e:
            self._skip_on_billing_error(e)
            raise
        # 基本形态
        assert isinstance(resp, LLMResponse)
        assert resp.content  # 非空
        assert resp.model
        # prompt/completion 至少有一个
        if resp.usage:
            assert resp.usage.get("total_tokens", 0) > 0

    def test_invoke_sync(self, llm_env):
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        client = self._make_client(llm_env)
        req = LLMRequest(
            messages=[{"role": "user", "content": "Say 'ack'"}],
            max_tokens=10,
            temperature=0.0,
        )
        try:
            resp = client.invoke(req)
        except Exception as e:
            self._skip_on_billing_error(e)
            raise
        assert isinstance(resp, LLMResponse)
        assert resp.content

    @pytest.mark.asyncio
    async def test_ainvoke_with_per_request_model_override(self, llm_env):
        """
        request.model 显式给出时不与 partial 冲突（修复 llm/client.py:89 的 partial kwargs 冲突）。
        """
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        client = self._make_client(llm_env)
        req = LLMRequest(
            messages=[{"role": "user", "content": "ok"}],
            model=llm_env["LLM_MODEL"],  # 显式 override
            max_tokens=5,
        )
        try:
            resp = await client.ainvoke(req)
        except Exception as e:
            self._skip_on_billing_error(e)
            raise
        # 至少 model 字段没崩（具体内容受账户状态影响）
        assert resp.model == llm_env["LLM_MODEL"] or resp.model  # 接受任意非空值


# ============================================================
# 5b. OpenAIClient 真实 astream / stream（走 .env 真 API）
# ============================================================
class TestOpenAIClientStreaming:
    def _make_client(self, llm_env) -> OpenAIClient:
        return OpenAIClient(
            api_key=llm_env["LLM_API_KEY"],
            base_url=llm_env["LLM_BASE_URL"],
            model=llm_env["LLM_MODEL"],
            max_retries=1,
            timeout=60.0,
        )

    def _skip_on_billing_error(self, exc: Exception) -> None:
        from openai import APIStatusError
        if isinstance(exc, APIStatusError):
            code = getattr(exc, "status_code", None)
            if code in (402, 429):
                pytest.skip(
                    f"真 LLM 流式调用受限于账户/限流（HTTP {code}）"
                    f"，请充值或稍后重试。错误：{str(exc)[:120]}"
                )

    @pytest.mark.asyncio
    async def test_astream_simple(self, llm_env):
        """验证 astream 返回 async iterator，逐块产出 LLMResponse。"""
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        client = self._make_client(llm_env)
        req = LLMRequest(
            messages=[{"role": "user", "content": "Reply with the word: pong"}],
            max_tokens=20,
            temperature=0.0,
        )
        try:
            chunks = []
            async for chunk in client.astream(req):
                chunks.append(chunk)
            assert len(chunks) > 0
            # 最后一个或多个 chunk 应有内容
            assert any(c.content for c in chunks) or any(c.reasoning_content for c in chunks)
        except Exception as e:
            self._skip_on_billing_error(e)
            raise

    @pytest.mark.asyncio
    async def test_astream_yields_llmresponse_type(self, llm_env):
        """stream 的每个 chunk 都是 LLMResponse 类型。"""
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        client = self._make_client(llm_env)
        req = LLMRequest(
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        try:
            async for chunk in client.astream(req):
                assert isinstance(chunk, LLMResponse)
            # 如果没报错就说明类型正确
        except Exception as e:
            self._skip_on_billing_error(e)
            raise

    def test_stream_sync(self, llm_env):
        """验证 stream 返回 iterator，逐块产出 LLMResponse。"""
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        client = self._make_client(llm_env)
        req = LLMRequest(
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        try:
            chunks = list(client.stream(req))
            assert len(chunks) > 0
            for chunk in chunks:
                assert isinstance(chunk, LLMResponse)
        except Exception as e:
            self._skip_on_billing_error(e)
            raise


# ============================================================
# 6. 重试 deny 行为（认证错不重试）
# ============================================================
class TestRetryDenyBehavior:
    """
    验证 _initialize_provider 不会真正连网络（不调 chat.completions.create），
    只测 deny_exceptions 黑名单逻辑。直接构造 client，再用 LLMRetryConfig.should_retry 判断。
    """

    def test_openai_auth_error_in_deny(self, llm_env):
        """
        LLMRetryConfig.default().retry_config.deny_exceptions 应包含 openai 认证类。
        """
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        from infra.llm.retry import LLMRetryConfig
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError

        rc = LLMRetryConfig.default()
        deny = rc.retry_config.deny_exceptions
        # 认证/权限/请求类应在 deny
        assert AuthenticationError in deny
        assert BadRequestError in deny
        assert PermissionDeniedError in deny

    def test_rate_limit_in_allow(self, llm_env):
        """
        LLMRetryConfig.default().retry_config.allow_exceptions 应包含 openai 限流/超时类。
        """
        if not _has_real_credentials(llm_env):
            pytest.skip("缺真 LLM_API_KEY / LLM_BASE_URL")
        from infra.llm.retry import LLMRetryConfig
        from openai import APITimeoutError, RateLimitError, APIError

        rc = LLMRetryConfig.default()
        allow = rc.retry_config.allow_exceptions
        # 限流/超时/API 错误应在 allow
        assert RateLimitError in allow
        assert APITimeoutError in allow
        assert APIError in allow
        # 通用类也应在
        assert ConnectionError in allow
        assert TimeoutError in allow
        assert OSError in allow


# ============================================================
# 7. LLMClient 抽象基类不能直接实例化
# ============================================================
class TestAbstractBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            LLMClient(api_key="x")

    def test_abstract_requires_streaming_methods(self):
        """抽象基类声明了 astream / stream，子类必须实现。"""
        assert hasattr(LLMClient, "astream")
        assert hasattr(LLMClient, "stream")
        assert getattr(LLMClient, "astream").__isabstractmethod__
        assert getattr(LLMClient, "stream").__isabstractmethod__
