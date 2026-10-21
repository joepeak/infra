#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm 第二轮深度审查的回归测试（20261021 晚间）

覆盖本轮实测复现并修复的问题：
1. probe_capabilities 抛 UnboundLocalError（P0）——BadRequestError 在 except
   块内局部 import，同函数其他 try 块引用时名字未绑定
2. record_usage 遇 None 抛 TypeError（P1）——provider 可能返回 None token
3. poll_video 超时按 interval 累加而非真实耗时（P1）——max_wait=1s 实跑 6.6s
4. response_format 类型无法表达 json_schema（P1）——Dict[str, str] 装不下嵌套 dict
5. Media 侧 use_fallback 是死参数（P1）——传了静默无效
6. capability_probe 配置项从未被读取（P2）——配置形同虚设
7. media 成本维度与 token 维度混淆（P2）——按次/按张/按时长计费塞进 token 维度
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from infra.llm.capabilities import get_registry, reset_registry
from infra.llm.client import OpenAIClient, OpenAIMediaClient
from infra.llm.models import LLMRequest
from infra.llm.usage import (
    get_media_usage_summary,
    get_usage_summary,
    record_media_billable,
    record_media_task,
    record_usage,
    reset_media_usage_summary,
    reset_usage_summary,
)


# ============================================================
# 公共 helper
# ============================================================
def make_bad_request(message: str = "invalid response_format") -> Exception:
    """构造一个 openai BadRequestError（400）。"""
    from openai import BadRequestError

    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {}
    resp.request = MagicMock()
    body = MagicMock()
    body.message = message
    return BadRequestError(message, response=resp, body=body)


def make_ok_response(content: str = '{"ok": true}') -> MagicMock:
    """构造一个最小可用的 chat completion 响应。

    注意 mock 形态必须对齐真实 openai SDK——``model`` / ``finish_reason``
    必须是**字符串**（真实响应里就是 str），用 MagicMock 会导致
    pydantic 校验失败。
    """
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.model = "m"
    resp.usage = None
    return resp


class FakeMediaRequest:
    """最小 MediaRequest 替身（避免依赖真实 pydantic 模型的字段约束）。"""

    def __init__(
        self,
        model: Any = None,
        prompt: str = "a prompt",
        n: int = 1,
        size: Any = None,
        quality: Any = None,
        response_format: Any = None,
        reference_images: Any = None,
        seed: Any = None,
        extra_body: Any = None,
        retry_config: Any = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.n = n
        self.size = size
        self.quality = quality
        self.response_format = response_format
        self.reference_images = reference_images
        self.seed = seed
        self.extra_body = extra_body
        self.retry_config = retry_config


@pytest.fixture(autouse=True)
def _clean_registries():
    """每个用例前后清理注册表与聚合器，避免相互污染。"""
    reset_registry()
    reset_usage_summary()
    reset_media_usage_summary()
    yield
    reset_registry()
    reset_usage_summary()
    reset_media_usage_summary()


# ============================================================
# 1. probe_capabilities 的 UnboundLocalError
# ============================================================
class TestProbeCapabilitiesUnboundLocal:
    """BadRequestError 必须在模块级导入。

    旧代码在 except 块内 `from openai import BadRequestError`——名字只在
    except 执行后才绑定。probe 的第 2/3/4 步在**其他** try 块里引用它，
    于是「首次探测成功、后续探测 400」时抛 UnboundLocalError，
    整个能力探测崩溃。
    """

    async def test_probe_survives_first_ok_then_400(self):
        """第一次探测成功、第二次(json_schema)抛 400 —— 不能崩。"""
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=False
        )
        calls = {"n": 0}

        async def create(**kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                return make_ok_response()
            raise make_bad_request("invalid json_schema parameter")

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        changes = await client.probe_capabilities("m")

        # 不抛异常即为修复；json_schema 应被判定为不支持
        assert changes.get("json_schema") is False

    async def test_probe_survives_first_400_then_ok(self):
        """第一次探测就 400 —— 后续步骤仍能正常运行。"""
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=False
        )
        calls = {"n": 0}

        async def create(**kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise make_bad_request("thinking is not supported")
            return make_ok_response()

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        changes = await client.probe_capabilities("m")
        assert changes.get("thinking_param_rejected") is True

    def test_bad_request_error_imported_at_module_level(self):
        """回归锁：BadRequestError 必须是模块级名字。"""
        import infra.llm.client as client_mod

        assert hasattr(client_mod, "BadRequestError")


# ============================================================
# 2. record_usage 遇 None
# ============================================================
class TestRecordUsageNoneSafe:
    """record_usage 是公开 API——外部传 None 不该让统计崩掉。"""

    def test_none_total_tokens_does_not_raise(self):
        record_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": None}
        )
        summary = get_usage_summary()["unknown"]
        assert summary["prompt_tokens"] == 10
        assert summary["completion_tokens"] == 5
        assert summary["total_tokens"] == 0

    def test_missing_keys_default_to_zero(self):
        record_usage({"prompt_tokens": 10})
        summary = get_usage_summary()["unknown"]
        assert summary["completion_tokens"] == 0
        assert summary["total_tokens"] == 0
        assert summary["calls"] == 1

    def test_non_numeric_value_does_not_raise(self):
        """非数值输入按 0 处理——统计逻辑不该影响主流程。"""
        record_usage({"prompt_tokens": "abc", "total_tokens": None})
        summary = get_usage_summary()["unknown"]
        assert summary["prompt_tokens"] == 0

    def test_call_count_still_increments(self):
        record_usage({"total_tokens": None})
        record_usage({"total_tokens": None})
        assert get_usage_summary()["unknown"]["calls"] == 2


# ============================================================
# 3. poll_video 超时按真实耗时
# ============================================================
class TestPollVideoRealTimeout:
    """poll_video 必须按**真实经过时间**判定超时。

    旧实现 `elapsed += interval` 累加的是配置间隔而非实际耗时——
    单次轮询请求慢时，实际耗时会远超 max_wait。
    """

    async def test_timeout_uses_real_elapsed_time(self):
        client = OpenAIMediaClient(api_key="k", base_url="http://x", video_model="vm")

        async def retrieve(task_id: str) -> Any:
            await asyncio.sleep(0.3)
            resp = MagicMock()
            resp.status = "running"
            resp.completed = False
            resp.failed = False
            return resp

        client._async_client = MagicMock()
        client._async_client.video.tasks.retrieve = retrieve

        t0 = time.monotonic()
        result = await client.poll_video("t", interval=0.1, max_wait=1.0)
        real = time.monotonic() - t0

        assert result.videos[0].status == "running"
        # 真实耗时应接近 max_wait，而不是 interval * 轮询次数
        # 旧实现：每次请求 0.3s、interval 0.1s → 约 6s 才触发超时
        assert real < 2.5, f"实际耗时 {real:.2f}s 远超 max_wait=1.0s"

    async def test_timeout_still_returns_running_status(self):
        client = OpenAIMediaClient(api_key="k", base_url="http://x", video_model="vm")

        async def retrieve(task_id: str) -> Any:
            resp = MagicMock()
            resp.status = "running"
            resp.completed = False
            resp.failed = False
            return resp

        client._async_client = MagicMock()
        client._async_client.video.tasks.retrieve = retrieve

        result = await client.poll_video("t", interval=0.01, max_wait=0.05)
        assert result.videos[0].status == "running"
        assert "timed out" in (result.videos[0].error or "")


# ============================================================
# 4. response_format 支持 json_schema
# ============================================================
class TestResponseFormatNestedSchema:
    """response_format 是透传字段——必须能装下 json_schema 的嵌套 dict。"""

    def test_json_schema_response_format_accepted(self):
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "s",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                },
                "strict": True,
            },
        }
        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            json_mode="json_schema",
            response_format=schema,
        )
        assert req.response_format == schema

    def test_json_schema_survives_to_openai_kwargs(self):
        """嵌套结构不能被 model_dump 破坏。"""
        schema = {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": {"type": "object"}},
        }
        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            response_format=schema,
        )
        assert req.to_openai_kwargs()["response_format"] == schema

    def test_plain_json_object_still_works(self):
        """向后兼容：原来的 {'type': 'json_object'} 仍可用。"""
        req = LLMRequest(
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )
        assert req.response_format == {"type": "json_object"}


# ============================================================
# 5. Media 侧 use_fallback 真正生效
# ============================================================
class TestMediaFallback:
    """use_fallback 之前是死参数——传了也静默走主端点。"""

    def test_distinct_fallback_creates_separate_client(self):
        client = OpenAIMediaClient(
            api_key="k1", base_url="http://main", fallback_base_url="http://fb"
        )
        assert client._async_client is not client._fallback_async_client
        assert str(client._fallback_async_client.base_url).startswith("http://fb")

    def test_same_endpoint_reuses_main_client(self):
        """同端点同 key —— 复用主实例，零额外开销。"""
        client = OpenAIMediaClient(
            api_key="k1", base_url="http://main", fallback_base_url="http://main"
        )
        assert client._async_client is client._fallback_async_client

    def test_distinct_api_key_creates_separate_client(self):
        client = OpenAIMediaClient(
            api_key="k1", base_url="http://main", fallback_api_key="k2"
        )
        assert client._async_client is not client._fallback_async_client

    def test_media_client_for_selects_correct_client(self):
        client = OpenAIMediaClient(
            api_key="k1", base_url="http://main", fallback_base_url="http://fb"
        )
        assert client._media_client_for(False) is client._async_client
        assert client._media_client_for(True) is client._fallback_async_client

    async def test_generate_image_uses_fallback_client(self):
        """generate_image(use_fallback=True) 必须走 fallback 客户端。"""
        client = OpenAIMediaClient(
            api_key="k1", base_url="http://main", fallback_base_url="http://fb"
        )
        used: List[str] = []

        async def gen(**kwargs: Any) -> Any:
            used.append("fallback")
            resp = MagicMock()
            resp.data = []
            return resp

        client._fallback_async_client = MagicMock()
        client._fallback_async_client.images.generate = gen

        await client.generate_image(FakeMediaRequest(), use_fallback=True)
        assert used == ["fallback"]

    async def test_generate_image_main_client_by_default(self):
        client = OpenAIMediaClient(
            api_key="k1", base_url="http://main", fallback_base_url="http://fb"
        )
        used: List[str] = []

        async def gen(**kwargs: Any) -> Any:
            used.append("main")
            resp = MagicMock()
            resp.data = []
            return resp

        client._async_client = MagicMock()
        client._async_client.images.generate = gen

        await client.generate_image(FakeMediaRequest(), use_fallback=False)
        assert used == ["main"]


# ============================================================
# 6. capability_probe 配置真正生效
# ============================================================
class TestCapabilityProbeLazy:
    """capability_probe 配置项之前从未被读取——现在做懒探测。"""

    async def test_probe_runs_before_first_call(self):
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=True
        )
        probe_calls = {"n": 0}

        async def create(**kwargs: Any) -> Any:
            # 探测请求的特征：带 tools + tool_choice
            if "tools" in kwargs:
                probe_calls["n"] += 1
                return make_ok_response()
            resp = make_ok_response("hello")
            resp.usage = None
            return resp

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        await client.ainvoke(req)

        assert probe_calls["n"] > 0, "首次调用前应触发能力探测"

    async def test_probe_only_runs_once(self):
        """同一 (端点, 模型) 只探一次——避免每次调用都发 4 次探测请求。"""
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=True
        )
        probe_calls = {"n": 0}

        async def create(**kwargs: Any) -> Any:
            if "tools" in kwargs:
                probe_calls["n"] += 1
                return make_ok_response()
            return make_ok_response("hello")

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        await client.ainvoke(req)
        first = probe_calls["n"]
        await client.ainvoke(req)
        await client.ainvoke(req)

        assert probe_calls["n"] == first, "重复调用不应重复探测"

    async def test_probe_disabled_skips_probe(self):
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=False
        )
        probe_calls = {"n": 0}

        async def create(**kwargs: Any) -> Any:
            if "tools" in kwargs:
                probe_calls["n"] += 1
            return make_ok_response("hello")

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        await client.ainvoke(req)
        assert probe_calls["n"] == 0

    async def test_probe_failure_does_not_block_call(self):
        """探测失败不该让正常调用失败。"""
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=True
        )

        async def create(**kwargs: Any) -> Any:
            if "tools" in kwargs:
                raise RuntimeError("probe exploded")
            return make_ok_response("hello")

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await client.ainvoke(req)
        assert result.content == "hello"

    async def test_probe_result_written_to_registry(self):
        """探测结论应写入注册表——后续调用才能自适应。"""
        client = OpenAIClient(
            api_key="k", base_url="http://x", model="m", capability_probe=True
        )

        async def create(**kwargs: Any) -> Any:
            if "tools" in kwargs:
                raise make_bad_request("thinking is not supported")
            return make_ok_response("hello")

        client._async_client = MagicMock()
        client._async_client.chat.completions.create = create

        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        await client.ainvoke(req)

        caps = get_registry().get("http://x", "m")
        assert caps.thinking_param_rejected is True
        # source 可能是 "probe" 或后续学习覆盖的 "learn"——
        # 关键结论（thinking 被拒收）必须落库
        assert caps.source in ("probe", "learn")


# ============================================================
# 7. media 成本维度与 token 维度解耦
# ============================================================
class TestMediaCostSeparation:
    """media 按次/按张/按时长计费——不该塞进 token 维度。"""

    def test_media_call_does_not_pollute_token_summary(self):
        record_media_task("image", units={"images": 3})
        # token 聚合器不应出现 media_* 条目
        assert get_usage_summary() == {}

    def test_media_summary_has_calls_and_units(self):
        record_media_task("image", units={"images": 3})
        record_media_task("image", units={"images": 2})
        summary = get_media_usage_summary()["media_image"]
        assert summary.calls == 2
        assert summary.units["images"] == 5

    def test_video_unit_is_seconds(self):
        record_media_task("video", units={"seconds": 12})
        assert get_media_usage_summary()["media_video"].units["seconds"] == 12

    def test_audio_unit_is_chars(self):
        record_media_task("audio", units={"chars": 100})
        assert get_media_usage_summary()["media_audio"].units["chars"] == 100

    def test_multiple_billable_dimensions_coexist(self):
        """一次调用可能产生多个计费维度——必须并存而非互相覆盖。"""
        record_media_task(
            "image", model="dall-e-3", units={"images": 2, "hd_renders": 1}
        )
        units = get_media_usage_summary()["media_image"].units
        assert units["images"] == 2
        assert units["hd_renders"] == 1

    def test_cost_accumulated_per_currency(self):
        record_media_task("video", units={"seconds": 5}, cost=0.5, currency="USD")
        record_media_task("video", units={"seconds": 5}, cost=0.25, currency="USD")
        summary = get_media_usage_summary()["media_video"]
        assert summary.cost_by_currency["USD"] == pytest.approx(0.75)

    def test_different_currencies_are_not_mixed(self):
        """USD 与 CNY 不能加在一起——必须分桶。"""
        record_media_task("image", cost=1.0, currency="USD")
        record_media_task("image", cost=10.0, currency="CNY")
        costs = get_media_usage_summary()["media_image"].cost_by_currency
        assert costs["USD"] == pytest.approx(1.0)
        assert costs["CNY"] == pytest.approx(10.0)

    def test_cost_without_currency_goes_to_unknown_bucket(self):
        record_media_task("image", cost=2.0)
        costs = get_media_usage_summary()["media_image"].cost_by_currency
        assert costs["unknown"] == pytest.approx(2.0)

    def test_billable_does_not_increment_calls(self):
        """视频「提交 + 轮询」是同一次生成——补记不重复计任务数。"""
        record_media_task("video", units={"seconds": 0})
        record_media_billable("video", units={"seconds": 5})
        summary = get_media_usage_summary()["media_video"]
        assert summary.calls == 1
        assert summary.units["seconds"] == 5

    def test_billable_creates_entry_if_absent(self):
        """补记时若还没有条目，应能独立建条目（不崩）。"""
        record_media_billable("audio", units={"chars": 10})
        summary = get_media_usage_summary()["media_audio"]
        assert summary.calls == 0
        assert summary.units["chars"] == 10

    def test_model_tracked(self):
        record_media_task("image", model="dall-e-3")
        assert get_media_usage_summary()["media_image"].model == "dall-e-3"

    def test_snapshot_is_deep_copy(self):
        """快照必须是深拷贝——改它不能影响聚合器内部。"""
        record_media_task("image", units={"images": 1})
        snapshot = get_media_usage_summary()
        snapshot["media_image"].units["images"] = 999
        assert get_media_usage_summary()["media_image"].units["images"] == 1

    async def test_video_submit_and_poll_counted_once(self):
        """端到端：一次视频生成在聚合器里只算 1 次任务。"""
        client = OpenAIMediaClient(api_key="k", base_url="http://x", video_model="vm")

        async def gen(**kwargs: Any) -> Any:
            resp = MagicMock()
            resp.id = "task-1"
            return resp

        client._async_client = MagicMock()
        client._async_client.video.generate = gen

        await client.generate_video(FakeMediaRequest())

        async def retrieve(task_id: str) -> Any:
            resp = MagicMock()
            resp.status = "completed"
            resp.completed = True
            resp.failed = False
            resp.video_url = "http://x/v.mp4"
            resp.duration_seconds = 5.0
            resp.cost = 0.5
            resp.currency = "USD"
            return resp

        client._async_client.video.tasks.retrieve = retrieve
        await client.poll_video("task-1", interval=0.01, max_wait=2.0)

        summary = get_media_usage_summary()["media_video"]
        assert summary.calls == 1, "一次视频生成不应计 2 次"
        assert summary.units["seconds"] == 5, "应累计视频时长"
        assert summary.cost_by_currency["USD"] == pytest.approx(0.5)

    async def test_image_units_counted(self):
        client = OpenAIMediaClient(api_key="k", base_url="http://x")

        async def gen(**kwargs: Any) -> Any:
            resp = MagicMock()
            item = MagicMock()
            item.url = "http://x/i.png"
            item.b64_json = None
            item.revised_prompt = None
            resp.data = [item, item]
            return resp

        client._async_client = MagicMock()
        client._async_client.images.generate = gen

        await client.generate_image(FakeMediaRequest(n=2))

        summary = get_media_usage_summary()["media_image"]
        assert summary.calls == 1
        assert summary.units["images"] == 2, "应按张数计费"
