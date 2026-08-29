"""
tests/test_wechat.py — infra.utils.wechat 单元测试

覆盖：
1. _send_wechat_sync：空消息 / 成功 / 失败 returncode / 超时 / 找不到命令 / 异常
2. _extract_cooldown：含 cooldown 的字符串 → 返回数；不含 → None
3. send_to_wechat：
   - 200 + status=ok → True
   - 200 + status=error → False
   - 200 + 非 JSON → True（兜底）
   - 500 + cooldown 信息 → raise Exception（含冷却时间）
   - 500 + 非 JSON → raise Exception
   - 非 200/500 → raise Exception
   - asyncio.TimeoutError → raise
   - aiohttp.ClientError → raise
4. 重试装饰器：默认 allow=(ConnectionError/TimeoutError/OSError)，所以业务 Exception 不重试——锁住现状行为。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import aiohttp

import infra.utils.wechat as wechat_module


# ============================================================
# Fake 工厂：aiohttp double async with
# ============================================================
def _wrap_client_session(session: MagicMock) -> MagicMock:
    """包装成 `async with aiohttp.ClientSession() as session:` 的外层 ctx。"""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _make_session_for_post(response: MagicMock) -> MagicMock:
    """构造一个 inner session：post() 返回支持 __aenter__/__aexit__ 的 ctx。"""
    session = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session.post = MagicMock(return_value=cm)
    return session


def _make_response(status: int, body: str = "") -> MagicMock:
    """构造一个 aiohttp response：status 属性 + text() awaitable。"""
    resp = MagicMock()
    resp.status = status

    async def _text():
        return body

    resp.text = _text
    return resp


# ============================================================
# 1. _send_wechat_sync
# ============================================================
class TestSendWechatSync:
    def test_empty_message_returns_false(self):
        assert wechat_module._send_wechat_sync("") is False
        assert wechat_module._send_wechat_sync("   \n  ") is False

    def test_success(self):
        fake = MagicMock(returncode=0, stderr="", stdout="")
        with patch.object(wechat_module.subprocess, "run", return_value=fake) as run_mock:
            ok = wechat_module._send_wechat_sync("hello")
        assert ok is True
        # 验证 hermes send --to weixin <msg> 形式
        args = run_mock.call_args.args[0]
        assert args[0] == "hermes"
        assert args[1] == "send"
        assert args[2:4] == ["--to", "weixin"]
        assert args[4] == "hello"

    def test_failure_returncode(self):
        fake = MagicMock(returncode=1, stderr="oops", stdout="")
        with patch.object(wechat_module.subprocess, "run", return_value=fake):
            ok = wechat_module._send_wechat_sync("hello")
        assert ok is False

    def test_timeout(self):
        with patch.object(
            wechat_module.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="hermes", timeout=10),
        ):
            ok = wechat_module._send_wechat_sync("hello", timeout=10)
        assert ok is False

    def test_file_not_found(self):
        with patch.object(
            wechat_module.subprocess, "run", side_effect=FileNotFoundError("no hermes"),
        ):
            ok = wechat_module._send_wechat_sync("hello")
        assert ok is False

    def test_unknown_exception(self):
        with patch.object(
            wechat_module.subprocess, "run", side_effect=RuntimeError("boom"),
        ):
            ok = wechat_module._send_wechat_sync("hello")
        assert ok is False

    def test_long_message_truncated(self):
        long_msg = "x" * 5000
        fake = MagicMock(returncode=0)
        with patch.object(wechat_module.subprocess, "run", return_value=fake) as run_mock:
            wechat_module._send_wechat_sync(long_msg, max_length=100)
        sent_msg = run_mock.call_args.args[0][4]
        # 原 5000 字符被截到 100 + "... [消息过长已截断]" 尾
        assert len(sent_msg) < 200
        assert "消息过长已截断" in sent_msg

    def test_stderr_used_when_nonempty(self):
        fake = MagicMock(returncode=2, stderr="real err", stdout="ignored")
        with patch.object(wechat_module.subprocess, "run", return_value=fake):
            # 不抛异常即可——走 failure 路径
            assert wechat_module._send_wechat_sync("hi") is False


# ============================================================
# 2. _extract_cooldown
# ============================================================
class TestExtractCooldown:
    def test_found(self):
        # random.uniform mock 掉，避免抖动
        with patch.object(wechat_module.random, "uniform", return_value=5.0):
            cd = wechat_module._extract_cooldown("cooldown active for 12.5s, pls wait")
        assert cd == 12.5 + 5.0

    def test_not_found_returns_none(self):
        assert wechat_module._extract_cooldown("some other error") is None

    def test_partial_match_returns_none(self):
        # 缺单位/格式不对
        assert wechat_module._extract_cooldown("cooldown active for 12.5") is None
        assert wechat_module._extract_cooldown("cooldown 12.5s") is None


# ============================================================
# 3. send_to_wechat
# ============================================================
class TestSendToWechat:
    @pytest.mark.asyncio
    async def test_200_ok(self, monkeypatch):
        # fast sleep 防卡
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        resp = _make_response(200, json.dumps({"status": "ok"}))
        session = _make_session_for_post(resp)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            ok = await wechat_module.send_to_wechat("hi")
        assert ok is True

    @pytest.mark.asyncio
    async def test_200_business_error(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        resp = _make_response(200, json.dumps({"status": "error", "msg": "biz fail"}))
        session = _make_session_for_post(resp)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            ok = await wechat_module.send_to_wechat("hi")
        assert ok is False

    @pytest.mark.asyncio
    async def test_200_invalid_json_returns_true(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        resp = _make_response(200, "not json at all")
        session = _make_session_for_post(resp)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            ok = await wechat_module.send_to_wechat("hi")
        # 200 + 无法解析 JSON → 当成功兜底
        assert ok is True

    @pytest.mark.asyncio
    async def test_500_with_cooldown_raises(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)
        # cooldown 提取需要 random.uniform
        with patch.object(wechat_module.random, "uniform", return_value=0.0):
            resp = _make_response(500, json.dumps({"msg": "rate limit, cooldown active for 30s"}))
            session = _make_session_for_post(resp)
            ctx = _wrap_client_session(session)
            with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
                with pytest.raises(Exception, match="Server error"):
                    await wechat_module.send_to_wechat("hi")

    @pytest.mark.asyncio
    async def test_500_with_invalid_json_raises(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        resp = _make_response(500, "<html>bad</html>")
        session = _make_session_for_post(resp)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            with pytest.raises(Exception, match="Server error"):
                await wechat_module.send_to_wechat("hi")

    @pytest.mark.asyncio
    async def test_404_raises(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        resp = _make_response(404, "not found")
        session = _make_session_for_post(resp)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            with pytest.raises(Exception, match="HTTP 404"):
                await wechat_module.send_to_wechat("hi")

    @pytest.mark.asyncio
    async def test_timeout_raises(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        # session.post 抛 TimeoutError
        session = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            with pytest.raises(Exception, match="Request timeout"):
                await wechat_module.send_to_wechat("hi")

    @pytest.mark.asyncio
    async def test_client_error_raises(self, monkeypatch):
        async def _fast_sleep(_s): return None
        monkeypatch.setattr(wechat_module.asyncio, "sleep", _fast_sleep)

        session = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("conn refused"))
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx):
            with pytest.raises(Exception, match="Client error"):
                await wechat_module.send_to_wechat("hi")


# ============================================================
# 4. 重试装饰器现状锁
# ============================================================
class TestRetryBehavior:
    """
    锁住现状：默认 allow=(ConnectionError, TimeoutError, OSError)，
    send_to_wechat 内部 raise Exception → 装饰器捕获后不在 allow 列表 → 不重试，原样透传。
    """

    @pytest.mark.asyncio
    async def test_business_exception_not_retried(self, monkeypatch):
        """
        业务错（raise Exception）默认不会被装饰器重试，原因是 allow 列表里没 Exception。
        """
        sleep_calls = []

        async def _track_sleep(s):
            sleep_calls.append(s)

        monkeypatch.setattr(wechat_module.asyncio, "sleep", _track_sleep)

        resp = _make_response(500, "server down")  # 触发 send_to_wechat raise Exception
        session = _make_session_for_post(resp)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx) as cls:
            with pytest.raises(Exception, match="Server error"):
                await wechat_module.send_to_wechat("hi")

        # 仅调用 1 次（没重试）
        assert cls.call_count == 1
        # 500 路径里 send_to_wechat 内部已经 sleep(3) 一次，装饰器没再额外 sleep
        assert len(sleep_calls) == 1

    @pytest.mark.asyncio
    async def test_network_error_triggers_retry(self, monkeypatch):
        """
        网络错（OSError）→ 在 allow 列表 → 装饰器会重试 5 次（max_retries=5），全失败后透传。
        注意：实测重试 6 次（1 + 5 retries）。
        """
        sleep_calls = []

        async def _track_sleep(s):
            sleep_calls.append(s)

        monkeypatch.setattr(wechat_module.asyncio, "sleep", _track_sleep)

        # session.post 抛 OSError——wechat 的 except 链没单独处理 OSError，落到
        # `except Exception: raise`（原异常透传），OSError 在 retry allow 列表中 → 重试 6 次。
        session = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(side_effect=OSError("conn reset"))
        cm.__aexit__ = AsyncMock(return_value=None)
        session.post = MagicMock(return_value=cm)
        ctx = _wrap_client_session(session)
        with patch.object(wechat_module.aiohttp, "ClientSession", return_value=ctx) as cls:
            with pytest.raises(OSError, match="conn reset"):
                await wechat_module.send_to_wechat("hi")

        # OSError 在 allow 列表 → 装饰器重试 5 次（共 6 次）
        assert cls.call_count == 6
