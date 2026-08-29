"""
tests/test_lark.py — infra.utils.lark 单元测试

覆盖：
1. _build_message_data：text / markdown / interactive / is_raw 四种形态
2. create_field：纯函数 + is_short 默认值
3. _send_request：成功 / 200+code!=0 / 502 重试 / 504 重试 / TimeoutError / ClientError / 未知异常
4. send_to_lark：webhook 未配置短路 / max_retries 退避 / 业务错不重试 / 异常后仍按重试退避
5. send_text / send_markdown / send_card 便捷函数
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import infra.utils.lark as lark_module
import infra.config.loader as loader_module
from infra.config import init_config
from infra.config.loader import ConfigLoader


# ============================================================
# Fixture：隔离 lark 模块全局 + 注入临时配置
# ============================================================
@pytest.fixture(autouse=True)
def _reset_lark_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    每个测试前后：
    1. 重置 lark 模块全局（_webhook / _retry_config）
    2. 重置 ConfigLoader 单例
    3. 备份 MACRO_MONITOR_ENV / ENVIRONMENT
    4. monkeypatch infra.utils.lark.asyncio.sleep → 立刻返回（避免指数退避真等）
    """
    # 备份 env
    saved_env = {}
    for k in ("MACRO_MONITOR_ENV", "ENVIRONMENT"):
        saved_env[k] = os.environ.pop(k, None)

    # 重置 lark 模块全局
    lark_module._webhook = None
    lark_module._retry_config = None

    # 重置 config loader
    loader_module._config = None
    loader_module._config_loaded = False
    loader_module._loader = None
    ConfigLoader._instance = None

    # sleep 永不真等
    async def _fast_sleep(_seconds: float) -> None:
        return None
    monkeypatch.setattr(lark_module.asyncio, "sleep", _fast_sleep)

    yield

    # TEARDOWN
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    lark_module._webhook = None
    lark_module._retry_config = None
    loader_module._config = None
    loader_module._config_loaded = False
    loader_module._loader = None
    ConfigLoader._instance = None


@pytest.fixture
def lark_config_dir(tmp_path: Path) -> Path:
    """写一份带 lark webhook + retry 的 config.yaml 并 init。"""
    cfg = {
        "lark": {
            "webhook": "https://open.feishu.cn/hook_test_xyz",
            "retry": {
                "max_retries": 2,
                "base_delay": 0.01,
                "max_delay": 0.05,
                "timeout": 5,
            },
        }
    }
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8"
    )
    init_config(tmp_path)
    return tmp_path


# ============================================================
# 1. _build_message_data —— 纯函数
# ============================================================
class TestBuildMessageData:
    def test_text_default(self):
        data = lark_module._build_message_data("hi")
        assert data == {"msg_type": "text", "content": {"text": "hi"}}

    def test_text_explicit(self):
        data = lark_module._build_message_data("hi", msg_type="text")
        assert data["msg_type"] == "text"
        assert data["content"]["text"] == "hi"

    def test_markdown_with_title(self):
        data = lark_module._build_message_data(
            "**bold**", msg_type="markdown", title="T", template="red"
        )
        assert data["msg_type"] == "interactive"
        assert data["card"]["header"]["title"]["content"] == "T"
        assert data["card"]["header"]["template"] == "red"
        assert data["card"]["elements"][0]["tag"] == "markdown"
        assert data["card"]["elements"][0]["content"] == "**bold**"

    def test_interactive_with_string_fields(self):
        data = lark_module._build_message_data(
            "body",
            msg_type="interactive",
            title="Title",
            fields=["a", "b"],
        )
        elems = data["card"]["elements"]
        assert elems[0]["tag"] == "div"
        # 第二个 div 是 fields
        assert elems[1]["tag"] == "div"
        assert "fields" in elems[1]
        # str 字段默认 is_short=False
        assert elems[1]["fields"][0]["is_short"] is False
        assert elems[1]["fields"][1]["is_short"] is False

    def test_interactive_with_tuple_fields(self):
        data = lark_module._build_message_data(
            "body",
            msg_type="interactive",
            fields=[("a", True), ("b", False)],
        )
        fields = data["card"]["elements"][1]["fields"]
        assert fields[0]["is_short"] is True
        assert fields[1]["is_short"] is False

    def test_is_raw_passthrough(self):
        raw = {"msg_type": "text", "content": {"text": "raw"}}
        data = lark_module._build_message_data(raw, is_raw=True)
        assert data is raw

    def test_unknown_msg_type_falls_through_to_text(self):
        data = lark_module._build_message_data("hi", msg_type="unknown_xxx")
        assert data["msg_type"] == "text"


# ============================================================
# 2. create_field —— 纯函数
# ============================================================
class TestCreateField:
    def test_default_is_short(self):
        f = lark_module.create_field("Label", "value")
        assert f["is_short"] is True
        assert f["text"]["tag"] == "lark_md"
        assert "Label" in f["text"]["content"]
        assert "value" in f["text"]["content"]

    def test_explicit_is_short(self):
        f = lark_module.create_field("L", "V", is_short=False)
        assert f["is_short"] is False

    def test_markdown_in_value_preserved(self):
        f = lark_module.create_field("Price", "$50,000 **+5%**")
        # 构造规则是 "**{label}**\n{value}"
        assert f["text"]["content"].startswith("**Price**\n")
        assert "$50,000 **+5%**" in f["text"]["content"]


# ============================================================
# 3. _send_request —— HTTP 单次请求
# ============================================================
def _make_aiohttp_response(
    status: int,
    body: Any = None,
    text: Optional[str] = None,
    json_raises: bool = False,
):
    """
    构造一个可被 aiohttp 风格 `async with response as r` 使用的 fake response。
    注意：response.text / response.json 是 **awaitable** 方法（lark 用 await 调它们），
    所以这里用 AsyncMock 而不是设属性。
    """
    resp = MagicMock()
    resp.status = status
    body_text = text if text is not None else (json.dumps(body) if body is not None else "")

    async def _text():
        return body_text

    async def _json():
        if json_raises:
            raise json.JSONDecodeError("x", "x", 0)
        return body

    resp.text = _text
    resp.json = _json
    return resp


def _wrap_client_session(session: MagicMock) -> MagicMock:
    """
    把内层 fake session 包装成 `async with aiohttp.ClientSession() as session:` 的外层 ctx。
    lark 源码是双层 async with：
        async with aiohttp.ClientSession() as session:
            async with session.post(...) as response:
    所以 patch 时，ClientSession(...) 返回的对象必须支持 __aenter__/__aexit__。
    """
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _make_session(response: MagicMock) -> MagicMock:
    """构造一个 fake aiohttp.ClientSession：post 返回 response（用 async with 包）。"""
    session = MagicMock()

    # session.post(...) 本身是普通函数，返回一个 _ResponseContextManager-like 对象
    # 但 _send_request 写的是 `async with session.post(...) as response:`
    # 所以 session.post(...) 返回的对象必须支持 __aenter__/__aexit__。
    # 用 AsyncMock 设置这俩方法——它本身就是 awaitable 协议的对象。
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(return_value=cm)
    return session


class TestSendRequest:
    @pytest.mark.asyncio
    async def test_success(self):
        resp = _make_aiohttp_response(200, {"code": 0, "msg": "ok"})
        session = _make_session(resp)
        # lark 用 `async with aiohttp.ClientSession() as session:`，所以
        # patch 的 ClientSession 返回的也必须是个 async ctx manager。
        ctx = _wrap_client_session(session)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=ctx):
            result = await lark_module._send_request(
                {"msg_type": "text", "content": {"text": "hi"}},
                "https://x",
                5,
            )
        assert result == {"status": "success", "message": "Lark发送成功"}

    @pytest.mark.asyncio
    async def test_business_error_no_retry(self):
        resp = _make_aiohttp_response(200, {"code": "230xxx", "msg": "invalid token"})
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is False
        assert "invalid token" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_json_no_retry(self):
        resp = _make_aiohttp_response(200, body=None, text="<html>oops</html>", json_raises=True)
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is False

    @pytest.mark.parametrize("status", [502, 503, 504])
    @pytest.mark.asyncio
    async def test_http_error_should_retry(self, status):
        resp = _make_aiohttp_response(status, text="server error")
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is True
        assert f"HTTP {status}" in result["error"]

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
    @pytest.mark.asyncio
    async def test_http_error_should_not_retry(self, status):
        resp = _make_aiohttp_response(status, text="nope")
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is False

    @pytest.mark.asyncio
    async def test_timeout(self):
        session = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is True
        assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_client_error(self):
        import aiohttp as _aio
        session = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=_aio.ClientError("conn refused"))
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is True
        assert "conn refused" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_exception_no_retry(self):
        session = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            result = await lark_module._send_request({}, "https://x", 5)
        assert result["status"] == "error"
        assert result["should_retry"] is False


# ============================================================
# 4. send_to_lark —— 带重试的发送
# ============================================================
class TestSendToLark:
    @pytest.mark.asyncio
    async def test_no_webhook_short_circuit(self, monkeypatch):
        # 不 init_config，避免 loader RuntimeError。直接 patch _get_webhook 返空。
        monkeypatch.setattr(lark_module, "_get_webhook", lambda: "")
        result = await lark_module.send_to_lark("hello")
        assert result == {"status": "error", "error": "webhook not configured"}
    @pytest.mark.asyncio
    async def test_success_first_try(self, lark_config_dir):
        resp = _make_aiohttp_response(200, {"code": 0})
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)) as cls:
            result = await lark_module.send_to_lark("hello", msg_type="text")
        assert result["status"] == "success"
        # 首次成功 → 只调一次
        assert cls.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_502_then_success(self, lark_config_dir):
        # 第一次 502（重试），第二次 200+code=0（成功）
        resp_502 = _make_aiohttp_response(502, text="bad gateway")
        resp_ok = _make_aiohttp_response(200, {"code": 0})
        sessions = [_make_session(resp_502), _make_session(resp_ok)]
        idx = {"i": 0}

        def _factory(*a, **kw):
            i = idx["i"]
            idx["i"] += 1
            return sessions[i]

        with patch.object(
            lark_module.aiohttp,
            "ClientSession",
            side_effect=lambda *a, **kw: _wrap_client_session(_factory()),
        ) as cls:
            result = await lark_module.send_to_lark("hi", max_retries=2, base_delay=0.01)
        assert result["status"] == "success"
        assert cls.call_count == 2

    @pytest.mark.asyncio
    async def test_business_error_no_retry(self, lark_config_dir):
        resp = _make_aiohttp_response(200, {"code": 999, "msg": "biz err"})
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)) as cls:
            result = await lark_module.send_to_lark("hi", max_retries=3)
        assert result["status"] == "error"
        # 业务错不重试
        assert cls.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_returns_error(self, lark_config_dir):
        resp = _make_aiohttp_response(503, text="unavail")
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)) as cls:
            result = await lark_module.send_to_lark("hi", max_retries=2, base_delay=0.01)
        assert result["status"] == "error"
        # 503 应该一直重试直到 max_retries 用完
        # max_retries=2 → 尝试 1+2=3 次
        assert cls.call_count == 3

    @pytest.mark.asyncio
    async def test_kwargs_override_config(self, lark_config_dir):
        resp = _make_aiohttp_response(200, {"code": 0})
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)) as cls:
            # 显式覆盖 max_retries=0 → 0 次重试，1 次尝试
            result = await lark_module.send_to_lark("hi", max_retries=0)
        assert result["status"] == "success"
        assert cls.call_count == 1


# ============================================================
# 5. 便捷函数
# ============================================================
class TestConvenienceFunctions:
    @pytest.mark.asyncio
    async def test_send_text(self, lark_config_dir):
        resp = _make_aiohttp_response(200, {"code": 0})
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)) as cls:
            result = await lark_module.send_text("plain")
        assert result["status"] == "success"
        assert cls.call_count == 1

    @pytest.mark.asyncio
    async def test_send_markdown(self, lark_config_dir):
        resp = _make_aiohttp_response(200, {"code": 0})
        session = _make_session(resp)
        with patch.object(
            lark_module.aiohttp,
            "ClientSession",
            return_value=_wrap_client_session(session),
        ) as cls:
            result = await lark_module.send_markdown("**md**", title="T")
        assert result["status"] == "success"
        # 验证发出的 payload 是 interactive
        ctx = cls.return_value
        inner_session = ctx.__aenter__.return_value
        post_kwargs = inner_session.post.call_args.kwargs
        body = json.loads(post_kwargs["data"])
        assert body["msg_type"] == "interactive"
        assert body["card"]["header"]["title"]["content"] == "T"

    @pytest.mark.asyncio
    async def test_send_card(self, lark_config_dir):
        resp = _make_aiohttp_response(200, {"code": 0})
        session = _make_session(resp)
        with patch.object(lark_module.aiohttp, "ClientSession", return_value=_wrap_client_session(session)):
            fields = [lark_module.create_field("K", "V")]
            result = await lark_module.send_card("body", "Title", template="red", fields=fields)
        assert result["status"] == "success"


def sessions_post_kwargs(cls_mock) -> Optional[Dict[str, Any]]:
    """
    辅助：拿到被 fake ClientSession 拦截后，session.post 实际被调用的 kwargs。
    ClientSession(...) 被 wrap 成 ctx（async ctx manager），ctx.__aenter__ 返 inner session。
    """
    try:
        ctx = cls_mock.return_value
        inner_session = ctx.__aenter__.return_value
        return inner_session.post.call_args.kwargs
    except Exception:
        return None
