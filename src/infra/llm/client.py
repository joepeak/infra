"""infra.llm.client：LLM 客户端抽象基类 + Provider 实现。

设计：
- LLMClient 抽象基类——子类实现具体 provider
- 集成 infra.utils.retry 做重试
- 集成 infra.llm.usage 做 token 聚合
- 自动记录 token 用量
- 能力注册表（infra.llm.capabilities）驱动思考参数自适应 + fallback provider 支持
"""
import os
import asyncio
import inspect
import time as _time
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, Callable, Dict, Union, Tuple

from infra.utils.retry import RetryConfig, retry_async, retry_sync
from infra.llm.models import (
    LLMRequest,
    LLMResponse,
    MediaRequest,
    MediaResponse,
    Modality,
    ImageResult,
    VideoResult,
    AudioResult,
    UsageCallback,
)
from infra.llm.usage import record_usage
from infra.llm.retry import LLMRetryConfig
from infra.llm.capabilities import get_registry, ProviderCapabilities, reset_registry
from infra.logger import get_logger

# 用 infra 的 LazyLogger（而非原生 logging.getLogger）——
# 原生 logger 没有 handler，LLM 层的日志不会落进 infra 的日志文件。
logger = get_logger(__name__)


def _invoke_usage_callback(
    callback: Optional[UsageCallback],
    info: Dict[str, Any],
) -> None:
    """Safely invoke a usage_callback. Exceptions are swallowed and logged."""
    if callback is None:
        return
    try:
        callback(info)
    except Exception:
        logger.warning("usage_callback 执行失败（忽略）", exc_info=True)


def _safe_extract(value: Any, expected_type: Union[type, Tuple[type, ...]]) -> Any:
    """Return value only if it matches the expected type, else None.

    Guards against mock objects (auto-created attrs) leaking into pydantic
    validation — provider responses in tests are often plain MagicMocks.
    """
    if isinstance(value, expected_type):
        return value
    return None


def _dispose_awaitable(result: Any) -> None:
    """处置一个「可能还没被 await 的协程」——不泄漏、不报 never-awaited 警告。

    用于 GC 兜底路径。优先级很重要：

      1. 有运行中的 loop → ``create_task`` 让协程**真正执行**（这才释放连接）
      2. 没有 loop → 只能 ``close()`` 掉协程对象，避免 never-awaited 警告
         （此时连接只能靠 SDK 自身的回收逻辑）

    Args:
        result: 协程对象（已确认是 awaitable）
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        closer = getattr(result, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        return
    try:
        loop.create_task(result)
    except Exception:
        pass


class _StreamUsage:
    """流式调用的 token 用量累加器。"""

    __slots__ = ("prompt", "completion", "total")

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.total = 0

    def add(self, usage: Any) -> None:  # noqa: ANN401
        """累加一个 chunk 的 usage（usage 为 None 时跳过）。"""
        if not usage:
            return
        self.prompt += getattr(usage, "prompt_tokens", 0) or 0
        self.completion += getattr(usage, "completion_tokens", 0) or 0
        self.total += getattr(usage, "total_tokens", 0) or 0

    def has_any(self) -> bool:
        """是否记录到过任何 token。

        不能只看 ``total``——部分 provider 返回 ``total_tokens=0``
        但 prompt/completion 正常（免费端点常见），只看 total 会漏记成本。
        """
        return bool(self.prompt or self.completion or self.total)

    def as_usage(self) -> Dict[str, int]:
        """转成 usage dict；``total_tokens`` 缺失时回退到 prompt + completion。"""
        total = self.total or (self.prompt + self.completion)
        return {
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": total,
        }


class AsyncStreamIterator:
    """流式响应的异步迭代器（可显式关闭）。

    为什么不用裸 async generator：generator 的 ``finally`` 块只在生成器被
    **关闭或 GC** 时才执行。消费方写 ``gen = client.astream(...); async for
    chunk in gen: break`` 时，生成器引用仍被持有，``finally`` 不会执行——
    底层 HTTP 连接不会释放，久了会耗尽连接池。

    本类直接持有底层流并实现 ``__anext__``，提供三条确定性释放路径：
      1. ``async with client.astream(req) as stream:`` —— **推荐**，退出即释放
      2. ``await stream.aclose()`` —— 显式释放
      3. 对象被 GC 时由 ``__del__`` 兜底释放

    注意：裸 ``async for ... break`` **不会**立即关闭——CPython 的 async for
    在 break 时不通知迭代器，底层流要等对象释放才关闭。这是语言层面的限制，
    不是本实现能绕过的；需要确定性释放就用 ``async with``。
    """

    def __init__(
        self,
        stream_factory: Any,  # noqa: ANN401
        on_close: Any = None,  # noqa: ANN401
        on_chunk: Any = None,  # noqa: ANN401
        transform: Any = None,  # noqa: ANN401
    ) -> None:
        self._stream_factory = stream_factory
        self._on_close = on_close
        self._on_chunk = on_chunk
        self._transform = transform
        self._stream: Any = None  # noqa: ANN401
        self._iterator: Any = None  # noqa: ANN401
        self._closed = False
        self._settled = False

    def __aiter__(self) -> "AsyncStreamIterator":
        return self

    async def __anext__(self) -> Any:  # noqa: ANN401
        if self._closed:
            raise StopAsyncIteration
        if self._iterator is None:
            # 惰性创建底层流（真正发请求发生在第一次迭代时）
            self._stream = self._stream_factory()
            if inspect.isawaitable(self._stream):
                # 异步工厂（openai SDK 的 create 是协程）
                self._stream = await self._stream
            self._iterator = self._stream.__aiter__()
        try:
            raw = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException:
            # 流中途出错（网络异常 / provider 错误）——必须关闭底层流，
            # 否则连接泄漏（异常路径最容易漏）
            await self.aclose()
            raise
        if self._on_chunk is not None:
            self._on_chunk(raw)
        if self._transform is not None:
            return self._transform(raw)
        return raw

    async def aclose(self) -> None:
        """关闭流（幂等）——立即释放底层 HTTP 连接。

        兼容三种底层流形态（真实 openai SDK 只提供其中一种，混用会漏关）：
          - ``aclose()`` 协程——部分 HTTP 客户端（如 httpx 风格的自定义流）
          - ``close()`` 协程——**openai ``AsyncStream`` 的真实形态**（3.x）
          - ``close()`` 同步方法——httpx.Response 等

        注意：openai 的 ``AsyncStream.close`` 是 ``async def``，所以必须 await；
        若误用同步调用会得到「coroutine was never awaited」并且连接**不会**释放。
        """
        if self._closed:
            await self._settle()
            return
        self._closed = True
        await self._close_underlying()
        self._stream = None
        self._iterator = None
        await self._settle()

    async def _close_underlying(self) -> None:
        """按优先级关闭底层流（幂等、异常不外泄）。"""
        stream = self._stream
        if stream is None:
            return
        # 1) aclose() 协程
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            try:
                result = aclose()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("流关闭时发生异常（忽略）", exc_info=True)
            return
        # 2) close()——可能是协程（openai AsyncStream），也可能是同步方法（httpx）
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug("流关闭时发生异常（忽略）", exc_info=True)

    async def _settle(self) -> None:
        """结算 usage + 跑关闭回调（幂等）。"""
        if self._settled:
            return
        self._settled = True
        if self._on_close is not None:
            try:
                await self._on_close()
            except Exception:
                logger.debug("流关闭回调异常（忽略）", exc_info=True)

    async def __aenter__(self) -> "AsyncStreamIterator":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """GC 兜底——对象被回收时关闭底层流。

        ``aclose()`` 是协程，``__del__`` 里无法 await。这里尽力而为：
          - ``close()`` 是**同步**方法 → 直接调
          - ``close()`` 是**协程**（openai ``AsyncStream`` 的真实形态）→
            尝试在当前运行中的 loop 里调度
          - 只有 ``aclose()`` → 同样尝试调度
        没有运行中的 loop 时放弃（连接最终由 SDK 自身回收）。
        """
        stream = getattr(self, "_stream", None)
        if stream is None:
            return

        for attr in ("aclose", "close"):
            fn = getattr(stream, attr, None)
            if not callable(fn):
                continue
            try:
                result = fn()
            except Exception:
                return
            if inspect.isawaitable(result):
                # 协程——必须调度到 loop 上 await，否则连接不会释放
                _dispose_awaitable(result)
            return


class SyncStreamIterator:
    """流式响应的同步迭代器（可显式关闭）。

    与 :class:`AsyncStreamIterator` 同理——``with`` / ``close()`` 提供
    确定性释放，``__del__`` 做 GC 兜底。
    """

    def __init__(
        self,
        stream_factory: Any,  # noqa: ANN401
        on_close: Any = None,  # noqa: ANN401
        on_chunk: Any = None,  # noqa: ANN401
        transform: Any = None,  # noqa: ANN401
    ) -> None:
        self._stream_factory = stream_factory
        self._on_close = on_close
        self._on_chunk = on_chunk
        self._transform = transform
        self._stream: Any = None  # noqa: ANN401
        self._iterator: Any = None  # noqa: ANN401
        self._closed = False
        self._settled = False

    def __iter__(self) -> "SyncStreamIterator":
        return self

    def __next__(self) -> Any:  # noqa: ANN401
        if self._closed:
            raise StopIteration
        if self._iterator is None:
            self._stream = self._stream_factory()
            self._iterator = iter(self._stream)
        try:
            raw = next(self._iterator)
        except StopIteration:
            self.close()
            raise
        except BaseException:
            # 流中途出错——必须关闭底层流，否则连接泄漏
            self.close()
            raise
        if self._on_chunk is not None:
            self._on_chunk(raw)
        if self._transform is not None:
            return self._transform(raw)
        return raw

    def close(self) -> None:
        """关闭流（幂等）——立即释放底层 HTTP 连接。

        同步路径下底层流通常是 httpx 风格的**同步** ``close()``；
        但也要兼容返回协程的 ``close()``（部分异步实现），此时把协程
        丢到当前 loop 上执行，避免连接不释放。
        """
        if self._closed:
            self._settle()
            return
        self._closed = True
        if self._stream is not None:
            close = getattr(self._stream, "close", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        _dispose_awaitable(result)
                except Exception:
                    logger.debug("流关闭时发生异常（忽略）", exc_info=True)
        self._stream = None
        self._iterator = None
        self._settle()

    def _settle(self) -> None:
        """结算 usage + 跑关闭回调（幂等）。"""
        if self._settled:
            return
        self._settled = True
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception:
                logger.debug("流关闭回调异常（忽略）", exc_info=True)

    def __enter__(self) -> "SyncStreamIterator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        """GC 兜底——对象被回收时关闭底层流。

        同步流的 ``close()`` 通常是同步方法；若返回协程则调度到当前
        loop（无 loop 时丢弃，避免 never-awaited 警告）。
        """
        stream = getattr(self, "_stream", None)
        if stream is None:
            return
        close = getattr(stream, "close", None)
        if not callable(close):
            return
        try:
            result = close()
        except Exception:
            return
        if inspect.isawaitable(result):
            _dispose_awaitable(result)


class LLMClient(ABC):
    """LLM 客户端抽象基类——子类实现 ainvoke/invoke。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: float = 120.0,
        retry_config: Optional[RetryConfig] = None,
        fallback_model: Optional[str] = None,
        fallback_base_url: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        capability_probe: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.capability_probe = capability_probe
        # fallback 的 base_url 缺省回落到主 base_url（同端点不同 key 或同 key 不同模型）
        self.fallback_base_url = fallback_base_url or base_url
        # fallback 的 API key 缺省使用主 key（同端点不同 key 或同 key 不同模型）
        self.fallback_api_key = fallback_api_key or api_key
        if retry_config is None:
            retry_config = LLMRetryConfig.default().retry_config.copy(max_retries=max_retries)
        self.retry_config = retry_config
        self._initialize_provider()
        logger.info(
            f"Initializing LLMClient | provider={self.provider}, "
            f"model={self.default_model}, max_retries={max_retries}, "
            f"fallback_model={fallback_model}, "
            f"fallback_base_url={self.fallback_base_url}"
        )

    @abstractmethod
    def _initialize_provider(self) -> None:
        """子类初始化 provider 客户端（如 AsyncOpenAI）——必须实现。"""

    @abstractmethod
    def _initialize_fallback_provider(self) -> None:
        """子类初始化 fallback provider 客户端——必须实现。"""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider 名（'openai' / 'anthropic' / 'deepseek' / 'local'）。"""

    def _client_for(self, use_fallback: bool) -> Any:
        """按主/备选择底层客户端。"""
        return self._fallback_async_client if use_fallback else self._async_client

    def provider_base_url(self, use_fallback: bool = False) -> Optional[str]:
        """返回某侧（主/备）的 base_url，供能力注册表做缓存键。"""
        return self.fallback_base_url if use_fallback else self.base_url

    def provider_model(self, model: Optional[str], use_fallback: bool = False) -> str:
        """解析某侧实际使用的模型名（回落到默认模型）。"""
        if model:
            return model
        if use_fallback and self.fallback_model:
            return self.fallback_model
        return self.default_model

    async def probe_capabilities(self, model: str, use_fallback: bool = False) -> dict:
        """主动探测某模型的能力边界，返回能力变更字典（供注册表 update）。

        用最小请求（dummy tool + max_tokens=5）试探：
          1. 带 tools + tool_choice（强制）+ response_format 发一次
             - 若 400 且提到 tool_choice/thinking → thinking 与 tools 冲突
             - 若 400 且提到 response_format/json_object → 不支持 response_format
             - 若 400 且提到 tools → 不支持 tools
             - 不报错 → 能力齐备
          2. 探测 json_schema 支持（若 response_format 已知支持，再试 json_schema）
          3. 探测 reasoning_effort 参数接受度（GLM 需要，deepseek 不接受）
          4. 探测中文 JSON 后缀敏感度（deepseek-flash 曾返回空）

        探测失败（网络/超时/鉴权等）绝不抛出：返回空字典，由调用方回退到"学习式"降级。
        """
        changes: dict = {}
        dummy_tool = {
            "type": "function",
            "function": {
                "name": "ping",
                "description": "Just a capability probe.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        client = self._client_for(use_fallback)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": '只回复一个 json：{"ok": true}'}],
                    tools=[dummy_tool],
                    tool_choice={"type": "function", "function": {"name": "ping"}},
                    response_format={"type": "json_object"},
                    max_tokens=5,
                ),
                timeout=30.0,
            )
        except Exception as e:
            from openai import BadRequestError
            if isinstance(e, BadRequestError):
                msg = str(e).lower()
                if "tool_choice" in msg and "thinking" in msg:
                    changes["thinking_conflicts_with_tools"] = True
                    changes["supports_tools"] = True
                elif "thinking" in msg:
                    changes["thinking_param_rejected"] = True
                if ("response_format" in msg or "json_object" in msg) and "must contain the word" not in msg:
                    changes["supports_response_format"] = False
                if "tools" in msg and "thinking" not in msg and "tool_choice" not in msg:
                    changes["supports_tools"] = False
                changes["last_error"] = str(e)[:200]
                logger.info(f"[probe] {model} 探测命中 400，学到能力变更: {changes}")
            else:
                logger.warning(f"[probe] {model} 能力探测跳过（{type(e).__name__}: {e}）")
            return changes

        # 请求成功：能力齐备
        changes["supports_tools"] = True
        changes["supports_response_format"] = True

        # ---- 探测 json_schema 支持 ----
        try:
            await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": '只回复一个 json：{"ok": true}'}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "probe_schema",
                            "schema": {
                                "type": "object",
                                "properties": {"ok": {"type": "boolean"}},
                                "required": ["ok"],
                            },
                            "strict": True,
                        },
                    },
                    max_tokens=5,
                ),
                timeout=15.0,
            )
            changes["json_schema"] = True
        except Exception as e:
            if isinstance(e, BadRequestError) and "json_schema" in str(e).lower():
                changes["json_schema"] = False
            else:
                logger.warning(f"[probe] {model} json_schema 探测跳过（{type(e).__name__}: {e}）")

        # ---- 探测 reasoning_effort 参数接受度 ----
        try:
            await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=5,
                    extra_body={"reasoning_effort": "low"},
                ),
                timeout=15.0,
            )
            changes["reasoning_effort_supported"] = True
        except Exception as e:
            if isinstance(e, BadRequestError) and "reasoning_effort" in str(e).lower():
                changes["reasoning_effort_supported"] = False
            else:
                logger.warning(f"[probe] {model} reasoning_effort 探测跳过（{type(e).__name__}: {e}）")

        # ---- 探测中文 JSON 后缀敏感度 ----
        # deepseek-flash 在收到中文 JSON 后缀时可能返回空字符串
        # 通过在 prompt 中加入中文 JSON 标记，观察是否正常返回
        try:
            test_response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": '只回复一个 json：{"ok": true} 这个 JSON 后面没有其他乱七八糟的内容'}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=10,
                ),
                timeout=15.0,
            )
            content = test_response.choices[0].message.content
            if not content or content.strip() == "":
                changes["rejects_cn_suffix"] = True
            else:
                changes["rejects_cn_suffix"] = False
        except Exception as e:
            logger.warning(f"[probe] {model} cn_suffix 探测跳过（{type(e).__name__}: {e}）")

        logger.info(f"[probe] {model} 能力探测完成: {changes}")
        return changes

    @abstractmethod
    async def ainvoke(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> LLMResponse:
        """异步调用——子类必须实现。

        usage_callback：每次成功响应后以 usage dict 回调（可选）。
        """

    @abstractmethod
    def invoke(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> LLMResponse:
        """同步调用——子类必须实现。

        usage_callback 同 ainvoke。
        """

    @abstractmethod
    def astream(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> Any:
        """异步流式调用——返回流式响应迭代器。子类必须实现。

        返回值是异步可迭代对象（支持 ``async for``），**调用方不需 await**。
        额外提供 ``aclose()`` / ``async with`` 供提前退出时释放连接。

        usage_callback 在流结束时以聚合 usage 回调一次（可选）。
        """

    @abstractmethod
    def stream(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> Any:
        """同步流式调用——返回流式响应迭代器。子类必须实现。

        usage_callback 在流结束时以聚合 usage 回调一次（可选）。
        """

    def _record_usage_from_response(
        self,
        response: LLMResponse,
        usage_callback: Optional[UsageCallback] = None,
    ) -> None:
        """记录 token 用量——子类 ainvoke/invoke 后调用。

        若提供 usage_callback，则在记录完本地聚合后以 usage dict 回调一次。
        """
        if response.usage:
            record_usage(response.usage)
            if usage_callback is not None:
                _invoke_usage_callback(usage_callback, {
                    "modality": "text",
                    "provider": self.provider,
                    "model": response.model or "",
                    "prompt_tokens": response.usage.get("prompt_tokens", 0),
                    "completion_tokens": response.usage.get("completion_tokens", 0),
                    "total_tokens": response.usage.get("total_tokens", 0),
                })


# ===========================================================================
# OpenAI Provider（第一期实现）
# ===========================================================================

class OpenAIClient(LLMClient):
    """OpenAI / DeepSeek 等 OpenAI 兼容 API 客户端。

    多 Provider 支持（方案A）：主模型与 fallback 模型可以来自不同厂商，
    即不同的 ``base_url`` / ``api_key``。此时为 fallback 单独建一个 AsyncOpenAI 实例；
    若 fallback 与主模型同端点，则复用主实例，零额外开销。
    """

    def _initialize_provider(self) -> None:
        from openai import OpenAI, AsyncOpenAI
        kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._sync_client = OpenAI(**kwargs)  # type: ignore[arg-type]
        self._async_client = AsyncOpenAI(**kwargs)  # type: ignore[arg-type]

        # fallback 是否跨端点：base_url 不同，或显式提供了不同的 api_key
        _distinct_fallback = (
            bool(self.fallback_base_url)
            and self.fallback_base_url.rstrip("/") != (self.base_url or "").rstrip("/")
        )
        _distinct_key = bool(self.fallback_api_key) and self.fallback_api_key != self.api_key
        if _distinct_fallback or _distinct_key:
            self._fallback_sync_client = OpenAI(
                api_key=self.fallback_api_key or self.api_key,  # type: ignore[arg-type]
                base_url=self.fallback_base_url,
                timeout=self.timeout,
            )
            self._fallback_async_client = AsyncOpenAI(
                api_key=self.fallback_api_key or self.api_key,  # type: ignore[arg-type]
                base_url=self.fallback_base_url,
                timeout=self.timeout,
            )
            logger.info(f"OpenAIClient 使用独立 fallback provider：base_url={self.fallback_base_url}")
        else:
            # 同端点：复用主实例
            self._fallback_sync_client = self._sync_client
            self._fallback_async_client = self._async_client

    def _initialize_fallback_provider(self) -> None:
        # 初始化在 _initialize_provider 中完成
        pass

    @property
    def provider(self) -> str:
        return "openai"

    @staticmethod
    def _adapt_capability_params(
        request: LLMRequest,
        kwargs: Dict[str, Any],
        provider_base: Optional[str],
        effective_model: str,
    ) -> Dict[str, Any]:
        """按能力注册表适配请求参数（thinking 剔除 + json_mode 翻译）。

        同步 / 异步 / 流式四条调用路径共用这一份逻辑——避免各自复制后分叉。

        两件事：
        1. **thinking 参数端点适配**——``extra_body.thinking`` 是 DeepSeek 系专有
           参数，其他厂商可能不认识而报 400。若注册表已确认该端点拒收
           thinking，则自动剔除。
        2. **json_mode → response_format 翻译**——按端点已知能力选择
           json_object / json_schema / 纯 prompt 策略；``json_mode=None`` 时，
           若已知端点不支持 response_format 则自动剔除。
        """
        _caps = get_registry().get(provider_base, effective_model)

        # 1. thinking 参数端点适配
        _extra = kwargs.get("extra_body")
        if isinstance(_extra, dict) and "thinking" in _extra:
            if _caps.thinking_param_rejected:
                _extra = {k: v for k, v in _extra.items() if k != "thinking"}
                kwargs["extra_body"] = _extra or None
                logger.info(
                    f"[OpenAIClient] {effective_model} 已确认拒收 thinking 参数，本次调用自动剔除。"
                )

        # 2. json_mode → response_format 翻译
        _json_mode = getattr(request, "json_mode", None)
        if _json_mode:
            if _json_mode == "json_object":
                if _caps.supports_response_format is not False:
                    kwargs["response_format"] = {"type": "json_object"}
            elif _json_mode == "json_schema":
                if _caps.json_schema:
                    pass  # API 支持 json_schema 格式，response_format 已在 request 中携带 schema
                elif _caps.supports_response_format is not False:
                    kwargs["response_format"] = {"type": "json_object"}
            elif _json_mode == "prompt_only":
                kwargs.pop("response_format", None)
        else:
            # json_mode=None 时，若能力注册表已知不支持 response_format，自动剔除
            if _caps.supports_response_format is False:
                kwargs.pop("response_format", None)

        return kwargs

    async def ainvoke(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> LLMResponse:
        """Asynchronous invocation.

        use_fallback=True 时走 fallback provider（可用独立的 base_url/api_key）。

        能力注册表驱动思考参数自适应：
        - 若 extra_body.thinking 存在且该端点已知拒收 thinking → 自动剔除
        - 命中 BadRequestError 时学习能力回填注册表，供下次调用参考
        """
        cfg = request.retry_config or self.retry_config
        kwargs = request.to_openai_kwargs()

        client = self._client_for(use_fallback)
        effective_model = request.model or (self.fallback_model if use_fallback else None) or self.default_model
        provider_base = self.provider_base_url(use_fallback)

        kwargs = self._adapt_capability_params(request, kwargs, provider_base, effective_model)

        create_func = partial(
            client.chat.completions.create,
            model=effective_model,
            **kwargs,
        )
        t0 = _time.monotonic()
        try:
            response = await retry_async(create_func, config=cfg)
        except Exception as e:
            from openai import BadRequestError
            if isinstance(e, BadRequestError):
                # 学习：把端点能力回填注册表，后续调用与各 Agent 即可自适应
                get_registry().learn_from_error(provider_base, effective_model, str(e))
                logger.error(
                    f"[OpenAIClient] ainvoke 400（耗时 {_time.monotonic()-t0:.0f}s, "
                    f"model={effective_model}, use_fallback={use_fallback}）: {e}"
                )
            else:
                logger.error(
                    f"[OpenAIClient] ainvoke 失败（耗时 {_time.monotonic()-t0:.0f}s, "
                    f"use_fallback={use_fallback}）: {type(e).__name__}: {e}"
                )
            raise
        logger.info(
            f"[OpenAIClient] ainvoke 返回（耗时 {_time.monotonic()-t0:.0f}s, "
            f"model={effective_model}, use_fallback={use_fallback}）"
        )
        result = LLMResponse.from_openai_response(response, post_processor=request.post_processor)
        result.model = result.model or effective_model
        self._record_usage_from_response(result, usage_callback)
        if result.content:
            get_registry().learn_from_successful_response(
                provider_base, effective_model, result.content, result.reasoning_content
            )
        return result

    def invoke(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> LLMResponse:
        """Synchronous invocation.

        use_fallback=True 时走 fallback provider。
        """
        from infra.utils.retry import retry_sync
        cfg = request.retry_config or self.retry_config
        kwargs = request.to_openai_kwargs()

        client = self._sync_client if not use_fallback else self._fallback_sync_client
        effective_model = request.model or (self.fallback_model if use_fallback else None) or self.default_model
        provider_base = self.provider_base_url(use_fallback)

        kwargs = self._adapt_capability_params(request, kwargs, provider_base, effective_model)

        create_func = partial(
            client.chat.completions.create,
            model=effective_model,
            **kwargs,
        )
        t0 = _time.monotonic()
        try:
            response = retry_sync(create_func, config=cfg)
        except Exception as e:
            from openai import BadRequestError
            if isinstance(e, BadRequestError):
                get_registry().learn_from_error(provider_base, effective_model, str(e))
                logger.error(
                    f"[OpenAIClient] invoke 400（耗时 {_time.monotonic()-t0:.0f}s, "
                    f"model={effective_model}, use_fallback={use_fallback}）: {e}"
                )
            else:
                logger.error(
                    f"[OpenAIClient] invoke 失败（耗时 {_time.monotonic()-t0:.0f}s）: {type(e).__name__}: {e}"
                )
            raise
        result = LLMResponse.from_openai_response(
            response, post_processor=request.post_processor
        )
        result.model = result.model or effective_model
        self._record_usage_from_response(result, usage_callback)
        if result.content:
            get_registry().learn_from_successful_response(
                provider_base, effective_model, result.content, result.reasoning_content
            )
        return result

    def astream(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> Any:
        """Asynchronous streaming invocation.

        Returns an :class:`AsyncStreamIterator` yielding LLMResponse objects for
        each incremental chunk from the provider. Token usage is accumulated and
        a final usage callback fires at the end of the stream.

        调用方式与普通 async generator 一致（**不需要 await**）::

            async for chunk in client.astream(req):
                ...

        返回的迭代器额外支持 ``aclose()`` / ``async with``——提前退出循环时
        用它们**确定性**释放底层 HTTP 连接::

            async with client.astream(req) as stream:
                async for chunk in stream:
                    if done(chunk):
                        break            # aclose 在退出 async with 时执行

        注意：裸 ``async for ... break`` 不会立即关闭——CPython 的 async for
        在 break 时不通知迭代器，底层流要等迭代器对象被 GC 才关闭。
        需要确定性释放就用 ``async with``（或显式 ``await stream.aclose()``）。

        use_fallback=True 时走 fallback provider。
        """
        kwargs = request.to_openai_kwargs()

        client = self._client_for(use_fallback)
        effective_model = request.model or (self.fallback_model if use_fallback else None) or self.default_model
        provider_base = self.provider_base_url(use_fallback)

        kwargs = self._adapt_capability_params(request, kwargs, provider_base, effective_model)
        kwargs["stream"] = True

        t0 = _time.monotonic()
        usage = _StreamUsage()

        async def _make_stream() -> Any:  # noqa: ANN401
            """创建底层流（惰性——第一次迭代时才真正发请求）。"""
            return await client.chat.completions.create(
                model=effective_model,
                **kwargs,
            )

        async def _on_close() -> None:
            """流结束（或提前关闭）时结算 usage。"""
            if usage.has_any():
                _resp = LLMResponse(  # type: ignore[call-arg]
                    usage=usage.as_usage(),
                    model=effective_model,
                )
                self._record_usage_from_response(_resp, usage_callback)
            logger.info(
                f"[OpenAIClient] astream 返回（耗时 {_time.monotonic()-t0:.0f}s, "
                f"model={effective_model}, use_fallback={use_fallback}）"
            )

        return AsyncStreamIterator(
            _make_stream,
            on_close=_on_close,
            on_chunk=lambda chunk: usage.add(getattr(chunk, "usage", None)),
            transform=lambda chunk: LLMResponse.from_openai_chunk(
                chunk, post_processor=request.post_processor
            ),
        )

    def stream(
        self,
        request: LLMRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> Any:
        """Synchronous streaming invocation.

        Returns a :class:`SyncStreamIterator` yielding LLMResponse objects for
        each incremental chunk.

        The returned iterator supports ``close()`` and ``with`` — use them when
        you break out of the loop early for a **deterministic** release of the
        underlying HTTP connection::

            with client.stream(req) as stream:
                for chunk in stream:
                    if done(chunk):
                        break            # close 在退出 with 时执行

        注意：裸 ``for ... break`` 不会立即关闭——要确定性释放就用 ``with``
        （或显式 ``stream.close()``）。

        use_fallback=True 时走 fallback provider。
        """
        kwargs = request.to_openai_kwargs()

        client = self._sync_client if not use_fallback else self._fallback_sync_client
        effective_model = request.model or (self.fallback_model if use_fallback else None) or self.default_model
        provider_base = self.provider_base_url(use_fallback)

        kwargs = self._adapt_capability_params(request, kwargs, provider_base, effective_model)
        kwargs["stream"] = True

        usage = _StreamUsage()

        def _make_stream() -> Any:  # noqa: ANN401
            """创建底层流（惰性——第一次迭代时才真正发请求）。"""
            return client.chat.completions.create(
                model=effective_model,
                **kwargs,
            )

        def _on_close() -> None:
            """流结束（或提前关闭）时结算 usage。"""
            if usage.has_any():
                _resp = LLMResponse(  # type: ignore[call-arg]
                    usage=usage.as_usage(),
                    model=effective_model,
                )
                self._record_usage_from_response(_resp, usage_callback)

        return SyncStreamIterator(
            _make_stream,
            on_close=_on_close,
            on_chunk=lambda chunk: usage.add(getattr(chunk, "usage", None)),
            transform=lambda chunk: LLMResponse.from_openai_chunk(
                chunk, post_processor=request.post_processor
            ),
        )


class LLMFactory:
    """LLM 工厂——按 config['provider'] 选 provider 客户端。"""

    @staticmethod
    def create(config: dict) -> LLMClient:
        provider = config.get("provider", "openai").lower()
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        model = config.get("default_model", "gpt-4o")
        max_retries = config.get("max_retries", 3)
        timeout = config.get("timeout", 120.0)
        fallback_model = config.get("fallback_model")
        fallback_base_url = config.get("fallback_base_url")
        fallback_api_key = config.get("fallback_api_key")
        capability_probe = config.get("capability_probe", True)

        # 所有 OpenAI 兼容协议的 provider（deepseek/zhipu/glm 等）都走 OpenAIClient，
        # 区别仅在 base_url + 能力注册表条目。不新增 Provider 类。
        if provider in ("openai", "deepseek", "zhipu", "glm", "moonshot", "lingyi", "qwen", "doubao"):
            if not api_key:
                raise ValueError(
                    "LLM config 缺 api_key——请在 config dict 配 api_key 或环境变量 LLM_API_KEY"
                )
            return OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                model=model,
                max_retries=max_retries,
                timeout=timeout,
                fallback_model=fallback_model,
                fallback_base_url=fallback_base_url,
                fallback_api_key=fallback_api_key,
                capability_probe=capability_probe,
            )
        raise ValueError(
            f"unsupported provider: {provider}（已支持：openai/deepseek/zhipu/glm/moonshot/lingyi/qwen/doubao，均走 OpenAI 兼容协议）"
        )


# ===========================================================================
# Media Client (image / video / audio generation)
# ===========================================================================

class MediaClient(ABC):
    """Abstract media-generation client.

    Extends the LLM text client with image, video and audio generation
    capabilities. Providers that only do text will raise ``NotImplementedError``
    on media methods — callers must check ``provider_media_capabilities``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        image_model: str = "dall-e-3",
        video_model: Optional[str] = None,
        audio_model: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        retry_config: Optional[RetryConfig] = None,
        provider: str = "openai",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.image_model = image_model
        self.video_model = video_model
        self.audio_model = audio_model
        self.timeout = timeout
        self._provider = provider
        if retry_config is None:
            retry_config = LLMRetryConfig.default().retry_config.copy(max_retries=max_retries)
        self.retry_config = retry_config
        self._initialize_media_provider()

    @abstractmethod
    def _initialize_media_provider(self) -> None:
        """子类初始化媒体生成 provider 客户端（如 AsyncOpenAI）。"""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider 名（'openai' / 'seedream' / 'tokenfree' 等）。"""

    @property
    def provider_media_capabilities(self) -> ProviderCapabilities:
        """返回该 provider 的媒体能力画像。"""
        return get_registry().get(self.base_url, None)

    @abstractmethod
    async def generate_image(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Generate an image from a text prompt.

        usage_callback：生成完成后以 media cost info 回调一次（可选）。
        """

    @abstractmethod
    def generate_image_sync(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Synchronous image generation."""

    async def generate_video(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Submit a video generation task (async, returns task_id).

        usage_callback：提交任务后以 video 状态回调一次（可选）。
        """
        raise NotImplementedError(
            f"{self.provider} does not support video generation"
        )

    async def poll_video(
        self,
        task_id: str,
        use_fallback: bool = False,
        interval: float = 10.0,
        max_wait: float = 600.0,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Poll a video task until completion or timeout.

        usage_callback：任务完成/失败/超时时以 video 状态回调一次（可选）。
        """
        raise NotImplementedError(
            f"{self.provider} does not support video polling"
        )

    async def synthesize_audio(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Synthesize speech/audio from text.

        usage_callback：音频生成完成后以 audio cost info 回调一次（可选）。
        """
        raise NotImplementedError(
            f"{self.provider} does not support audio synthesis"
        )

    def _record_media_usage(
        self,
        modality: str,
        cost_info: Optional[Dict[str, Any]] = None,
        usage_callback: Optional[UsageCallback] = None,
    ) -> None:
        """Record a media generation call for usage tracking.

        cost_info: modality + model + cost-determining parameters.
        usage_callback: optional external callback invoked with cost_info (errors swallowed).
        """
        record_usage({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, purpose=f"media_{modality}")
        if usage_callback is not None and cost_info is not None:
            info: Dict[str, Any] = {
                "modality": modality,
                "provider": self._provider,
            }
            info.update(cost_info)
            _invoke_usage_callback(usage_callback, info)


class OpenAIMediaClient(MediaClient):
    """OpenAI-compatible media client (DALL·E, Seedream, etc.).

    Uses the OpenAI SDK's ``images`` namespace for image generation and
    ``chat.completions`` for text-to-speech / music. Video endpoints are
    provider-specific and must be passed via ``extra_body`` or overridden.
    """

    def _initialize_media_provider(self) -> None:
        from openai import OpenAI, AsyncOpenAI

        kwargs: dict = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._sync_client = OpenAI(**kwargs)  # type: ignore[arg-type]
        self._async_client = AsyncOpenAI(**kwargs)  # type: ignore[arg-type]

    @property
    def provider(self) -> str:
        return self._provider

    async def _generate_image_async(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        cfg = request.retry_config or self.retry_config
        client = self._async_client
        model = request.model or self.image_model

        api_kwargs: dict = {"model": model, "prompt": request.prompt, "n": request.n}
        if request.size:
            api_kwargs["size"] = request.size
        if request.quality:
            api_kwargs["quality"] = request.quality
        if request.response_format:
            api_kwargs["response_format"] = request.response_format
        if request.reference_images:
            api_kwargs["extra_image_urls"] = request.reference_images
        if request.seed is not None:
            api_kwargs["seed"] = request.seed
        if request.extra_body:
            api_kwargs.update(request.extra_body)

        async def _do_call() -> Any:
            return await client.images.generate(**api_kwargs)

        t0 = _time.monotonic()
        try:
            response = await retry_async(_do_call, config=cfg)
        except Exception as e:
            logger.error(
                f"[OpenAIMediaClient] image generation failed (model={model}): "
                f"{type(e).__name__}: {e}"
            )
            raise
        elapsed = _time.monotonic() - t0

        images = []
        data_list = getattr(response, "data", None) or []
        for item in data_list:
            images.append(
                ImageResult(
                    url=getattr(item, "url", None),
                    b64_json=getattr(item, "b64_json", None),
                    revised_prompt=getattr(item, "revised_prompt", None),
                    seed=request.seed,
                )
            )

        cost_info = {
            "model": model,
            "image_count": len(images),
            "size": request.size or "",
            "quality": request.quality or "standard",
        }
        self._record_media_usage("image", cost_info, usage_callback)
        logger.info(
            f"[OpenAIMediaClient] generated {len(images)} image(s) in {elapsed:.1f}s "
            f"with model={model}"
        )
        return MediaResponse(
            modality="image",
            model=model,
            images=images,
            cost=_safe_extract(getattr(response, "cost", None), (int, float)),
            currency=_safe_extract(getattr(response, "currency", None), str),
            cost_metadata=_safe_extract(getattr(response, "cost_metadata", None), dict),
        )

    def _generate_image_sync(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        client = self._sync_client
        model = request.model or self.image_model

        api_kwargs: dict = {"model": model, "prompt": request.prompt, "n": request.n}
        if request.size:
            api_kwargs["size"] = request.size
        if request.quality:
            api_kwargs["quality"] = request.quality
        if request.response_format:
            api_kwargs["response_format"] = request.response_format
        if request.reference_images:
            api_kwargs["extra_image_urls"] = request.reference_images
        if request.seed is not None:
            api_kwargs["seed"] = request.seed
        if request.extra_body:
            api_kwargs.update(request.extra_body)

        cfg = request.retry_config or self.retry_config
        t0 = _time.monotonic()
        try:
            response = retry_sync(lambda: client.images.generate(**api_kwargs), config=cfg)
        except Exception as e:
            logger.error(
                f"[OpenAIMediaClient] image generation (sync) failed (model={model}): "
                f"{type(e).__name__}: {e}"
            )
            raise
        elapsed = _time.monotonic() - t0

        images = []
        data_list = getattr(response, "data", None) or []
        for item in data_list:
            images.append(
                ImageResult(
                    url=getattr(item, "url", None),
                    b64_json=getattr(item, "b64_json", None),
                    revised_prompt=getattr(item, "revised_prompt", None),
                    seed=request.seed,
                )
            )

        cost_info = {
            "model": model,
            "image_count": len(images),
            "size": request.size or "",
            "quality": request.quality or "standard",
        }
        self._record_media_usage("image", cost_info, usage_callback)
        logger.info(
            f"[OpenAIMediaClient] generated {len(images)} image(s) in {elapsed:.1f}s "
            f"with model={model}"
        )
        return MediaResponse(
            modality="image",
            model=model,
            images=images,
            cost=_safe_extract(getattr(response, "cost", None), (int, float)),
            currency=_safe_extract(getattr(response, "currency", None), str),
            cost_metadata=_safe_extract(getattr(response, "cost_metadata", None), dict),
        )

    async def generate_image(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        return await self._generate_image_async(request, use_fallback, usage_callback)

    def generate_image_sync(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        return self._generate_image_sync(request, use_fallback, usage_callback)

    async def generate_video(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Seedance/WanX-style video generation via OpenAI-compatible endpoint.

        Uses ``extra_body`` for provider-specific parameters (e.g., ``prompt'',
        ``image_url'', ``duration'' for Seedance). The first call submits a task
        and returns a ``task_id``; use ``poll_video`` to check completion.
        """
        model = request.model or self.video_model
        if not model:
            raise ValueError("video_model not configured for this MediaClient")

        api_kwargs: dict = {"model": model}
        if request.extra_body:
            api_kwargs.update(request.extra_body)

        client = self._async_client
        cfg = request.retry_config or self.retry_config

        t0 = _time.monotonic()
        try:
            response = await retry_async(
                lambda: client.video.generate(**api_kwargs), config=cfg
            )
        except AttributeError:
            # Provider may not have video.generate — check if it uses chat-based video
            response = await retry_async(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": request.prompt}],
                    max_tokens=100,
                    extra_body=api_kwargs.get("extra_body"),
                ),
                config=cfg,
            )

        elapsed = _time.monotonic() - t0
        task_id = getattr(response, "id", None) or getattr(response, "task_id", None)

        cost_info = {
            "model": model,
            "video_status": "pending",
            "video_duration_seconds": None,
            "task_id": task_id,
        }
        self._record_media_usage("video", cost_info, usage_callback)
        logger.info(
            f"[OpenAIMediaClient] submitted video task {task_id} in {elapsed:.1f}s "
            f"with model={model}"
        )
        return MediaResponse(
            modality="video",
            model=model,
            videos=[VideoResult(task_id=task_id, status="pending")],  # type: ignore[call-arg]
            cost=_safe_extract(getattr(response, "cost", None), (int, float)),
            currency=_safe_extract(getattr(response, "currency", None), str),
            cost_metadata=_safe_extract(getattr(response, "cost_metadata", None), dict),
        )

    async def poll_video(
        self,
        task_id: str,
        use_fallback: bool = False,
        interval: float = 10.0,
        max_wait: float = 600.0,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """Poll a video-task endpoint (provider-specific path via extra_body).

        usage_callback：任务完成/失败/超时时以 video cost info 回调一次。
        """
        client = self._async_client
        elapsed = 0.0

        while elapsed < max_wait:
            primary_error: Optional[BaseException] = None
            try:
                # Try the dedicated video/tasks endpoint first
                response = await client.video.tasks.retrieve(task_id)  # type: ignore[union-attr]
            except Exception as e:
                primary_error = e
                # Fall back to a chat-based polling
                try:
                    response = await client.chat.completions.create(  # type: ignore[union-attr]
                        model=self.video_model or "video-poll",
                        messages=[{"role": "user", "content": "poll"}],
                        extra_body={"task_id": task_id},
                    )
                except Exception as fallback_error:
                    # 两条路都失败——把主端点的真实原因一起带出去，
                    # 否则排查时只能看到 fallback 的错误，主因（如余额不足）丢失。
                    raise RuntimeError(
                        f"视频任务轮询失败：主端点 {type(primary_error).__name__}: {primary_error}；"
                        f"chat fallback {type(fallback_error).__name__}: {fallback_error}"
                    ) from fallback_error

            status = getattr(response, "status", None)
            if status == "completed" or getattr(response, "completed", False):
                video_url = getattr(response, "video_url", None) or getattr(
                    response, "result_url", None
                )
                duration = _safe_extract(getattr(response, "duration_seconds", None), (int, float)) or _safe_extract(getattr(response, "duration", None), (int, float))
                cost_info = {
                    "model": self.video_model or "",
                    "video_status": "completed",
                    "video_duration_seconds": duration,
                    "task_id": task_id,
                    "cost": _safe_extract(getattr(response, "cost", None), (int, float)),
                    "currency": _safe_extract(getattr(response, "currency", None), str),
                }
                self._record_media_usage("video", cost_info, usage_callback)
                return MediaResponse(
                    modality="video",
                    model=self.video_model,
                    videos=[VideoResult(  # type: ignore[call-arg]
                        task_id=task_id,
                        status="completed",
                        video_url=video_url,
                        duration_seconds=duration,
                    )],
                    cost=_safe_extract(getattr(response, "cost", None), (int, float)),
                    currency=_safe_extract(getattr(response, "currency", None), str),
                    cost_metadata=_safe_extract(getattr(response, "cost_metadata", None), dict),
                )
            elif status == "failed" or getattr(response, "failed", False):
                cost_info = {
                    "model": self.video_model or "",
                    "video_status": "failed",
                    "video_duration_seconds": None,
                    "task_id": task_id,
                }
                self._record_media_usage("video", cost_info, usage_callback)
                return MediaResponse(
                    modality="video",
                    model=self.video_model,
                    videos=[VideoResult(  # type: ignore[call-arg]
                        task_id=task_id,
                        status="failed",
                        error=str(getattr(response, "error", "Unknown video generation error")),
                    )],
                )

            await asyncio.sleep(interval)
            elapsed += interval

        # Timeout — return last known state
        cost_info = {
            "model": self.video_model or "",
            "video_status": "running",
            "video_duration_seconds": None,
            "task_id": task_id,
        }
        self._record_media_usage("video", cost_info, usage_callback)
        return MediaResponse(
            modality="video",
            model=self.video_model,
            videos=[VideoResult(  # type: ignore[call-arg]
                task_id=task_id,
                status="running",
                error=f"Polling timed out after {max_wait}s",
            )],
        )

    async def synthesize_audio(
        self,
        request: MediaRequest,
        use_fallback: bool = False,
        usage_callback: Optional[UsageCallback] = None,
    ) -> MediaResponse:
        """TTS via OpenAI-compatible ``/v1/audio/speech`` endpoint."""
        model = request.model or self.audio_model
        if not model:
            raise ValueError("audio_model not configured for this MediaClient")

        api_kwargs: dict = {"model": model, "input": request.prompt, "voice": request.response_format or "nova"}
        if request.size:
            api_kwargs["speed"] = 1.0

        client = self._async_client
        cfg = request.retry_config or self.retry_config

        async def _do_call() -> Any:
            return await client.audio.speech.create(**api_kwargs)

        response = await retry_async(_do_call, config=cfg)

        # Response may be a binary stream (audio bytes) or an object with .url
        audio_url = getattr(response, "url", None)
        b64_data = None
        duration = _safe_extract(getattr(response, "duration_seconds", None), (int, float)) or _safe_extract(getattr(response, "duration", None), (int, float))
        if audio_url is None:
            # Binary response — read stream
            try:
                content = response.read() if hasattr(response, "read") else response
                if isinstance(content, bytes):
                    import base64
                    b64_data = base64.b64encode(content).decode("ascii")
                    audio_url = None
            except Exception:
                pass

        cost_info = {
            "model": model,
            "input_chars": len(request.prompt),
            "duration_seconds": duration,
        }
        self._record_media_usage("audio", cost_info, usage_callback)
        logger.info(f"[OpenAIMediaClient] synthesized audio with model={model}")
        return MediaResponse(
            modality="audio",
            model=model,
            audios=[AudioResult(  # type: ignore[call-arg]
                url=audio_url,
                b64_json=b64_data,
                duration_seconds=duration,
                voice=api_kwargs.get("voice"),
            )],
            cost=_safe_extract(getattr(response, "cost", None), (int, float)),
            currency=_safe_extract(getattr(response, "currency", None), str),
            cost_metadata=_safe_extract(getattr(response, "cost_metadata", None), dict),
        )


# ===========================================================================
# Media Factory + Global Singleton
# ===========================================================================

class MediaFactory:
    """Media client factory — mirrors LLMFactory for image/video/audio."""

    @staticmethod
    def create(config: dict) -> MediaClient:
        provider = config.get("provider", "openai").lower()
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        image_model = config.get("image_model", "dall-e-3")
        video_model = config.get("video_model")
        audio_model = config.get("audio_model")
        max_retries = config.get("max_retries", 3)
        timeout = config.get("timeout", 120.0)

        if not api_key:
            raise ValueError(
                "Media config 缺 api_key——请在 config dict 配 api_key 或环境变量 MEDIA_API_KEY"
            )

        if provider == "openai":
            return OpenAIMediaClient(
                api_key=api_key,
                base_url=base_url,
                image_model=image_model,
                video_model=video_model,
                audio_model=audio_model,
                max_retries=max_retries,
                timeout=timeout,
                provider=provider,
            )
        # Seedream / tokenfree are OpenAI-compatible — treat as openai with a provider hint.
        if provider in ("seedream", "tokenfree"):
            logger.info(f"[MediaFactory] provider '{provider}' treated as OpenAI-compatible")
            return OpenAIMediaClient(
                api_key=api_key,
                base_url=base_url,
                image_model=image_model,
                video_model=video_model,
                audio_model=audio_model,
                max_retries=max_retries,
                timeout=timeout,
                provider=provider,
            )
        raise ValueError(
            f"unsupported media provider: {provider}（已实现：openai/seedream/tokenfree）"
        )


_media_client: Optional[MediaClient] = None


async def init_media_client(config: dict) -> Optional[MediaClient]:
    """初始化全局媒体生成客户端（按配置创建）。

    配置来源：
      1. yaml media.* 节 (api_key, base_url, image_model, video_model, audio_model)
      2. 环境变量 MEDIA_API_KEY / MEDIA_BASE_URL / IMAGE_MODEL / VIDEO_MODEL / AUDIO_MODEL

    若未配置 api_key —— 返回 None（不抛错，让上层检查）。
    """
    global _media_client
    if _media_client is not None:
        return _media_client

    media_config = config.get("media", {}) or {}

    def _pick(yaml_val: Any, env_key: str) -> Any:
        placeholder = {"", "YOUR_API_KEY_HERE", None}
        if yaml_val and str(yaml_val) not in placeholder:
            return yaml_val
        return os.getenv(env_key)

    api_key = _pick(media_config.get("api_key"), "MEDIA_API_KEY")
    if not api_key:
        logger.warning("Media API key 未配置（yaml media.api_key 与环境变量 MEDIA_API_KEY 均为空），Media client 不可用。")
        return None

    base_url = _pick(media_config.get("base_url"), "MEDIA_BASE_URL")
    image_model = _pick(media_config.get("image_model"), "IMAGE_MODEL")
    video_model = _pick(media_config.get("video_model"), "VIDEO_MODEL")
    audio_model = _pick(media_config.get("audio_model"), "AUDIO_MODEL")

    _media_client = MediaFactory.create({
        "provider": media_config.get("provider", "openai"),
        "api_key": str(api_key),
        "base_url": str(base_url) if base_url else None,
        "image_model": str(image_model) if image_model else "dall-e-3",
        "video_model": str(video_model) if video_model else None,
        "audio_model": str(audio_model) if audio_model else None,
        "max_retries": media_config.get("max_retries", 3),
        "timeout": media_config.get("timeout", 300.0),
    })
    logger.info(
        f"Media global client initialized. provider={_media_client.provider}, "
        f"base_url={getattr(getattr(_media_client, '_async_client', None), 'base_url', None)}, "
        f"image_model={_media_client.image_model}"
    )
    return _media_client


def get_media_client() -> Optional[MediaClient]:
    """获取全局媒体客户端实例。"""
    if _media_client is None:
        logger.warning("Media client accessed before initialization or was not configured.")
    return _media_client


def _reset_media_client_for_test() -> None:
    """测试用——重置全局媒体单例。"""
    global _media_client
    _media_client = None
    reset_registry()


# ===========================================================================
# 全局单例 + init_llm_client / get_llm_client
# ===========================================================================

_llm_client: Optional[LLMClient] = None


def _pick(yaml_val: Any, env_key: str) -> Any:
    """优先级：yaml 有效值（非空且非占位符）→ 环境变量。

    占位符集合防止 'YOUR_API_KEY_HERE' 这样的字面值被误用。
    """
    placeholder = {"", "YOUR_API_KEY_HERE", None}
    if yaml_val and str(yaml_val) not in placeholder:
        return yaml_val
    return os.getenv(env_key)


async def init_llm_client(config: dict) -> Optional[LLMClient]:
    """初始化全局 LLM 客户端（按配置创建）。

    配置优先级（高→低）：
      1. yaml llm.* 节（且非占位符/非空）
      2. 环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL（.env 由 bootstrap 加载）

    若两边都未配置 api_key——返 None（不抛错，让上层检查）。

    备用模型/Provider（方案A）：
      LLM_MODEL_FALLBACK / LLM_FALLBACK_BASE_URL / LLM_FALLBACK_API_KEY
      控制主模型连续失败时切换的强模型/端点。
      LLM_CAPABILITY_PROBE —— 控制是否主动探测 provider 能力（默认开启）。
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    llm_config = config.get("llm", {}) or {}

    api_key = _pick(llm_config.get("api_key"), "LLM_API_KEY")
    if not api_key:
        logger.error(
            "LLM API key 未配置（yaml llm.api_key 与环境变量 LLM_API_KEY 均为空），LLM client 不可用。"
        )
        return None

    base_url = _pick(llm_config.get("base_url"), "LLM_BASE_URL")
    model = _pick(llm_config.get("default_model"), "LLM_MODEL")
    fallback_model = _pick(llm_config.get("fallback_model"), "LLM_MODEL_FALLBACK")
    fallback_base_url = _pick(llm_config.get("fallback_base_url"), "LLM_FALLBACK_BASE_URL")
    fallback_api_key = _pick(llm_config.get("fallback_api_key"), "LLM_FALLBACK_API_KEY")
    _probe_raw = _pick(llm_config.get("capability_probe"), "LLM_CAPABILITY_PROBE")
    capability_probe = (
        str(_probe_raw).strip().lower() not in ("false", "0", "no", "off")
        if _probe_raw is not None
        else True
    )

    _llm_client = LLMFactory.create({
        "provider": llm_config.get("provider", "openai"),
        "api_key": str(api_key),
        "base_url": str(base_url) if base_url else None,
        "default_model": str(model) if model else "gpt-4o",
        "max_retries": llm_config.get("max_retries", 5),
        "timeout": llm_config.get("timeout", 120.0),
        "fallback_model": str(fallback_model) if fallback_model else None,
        "fallback_base_url": str(fallback_base_url) if fallback_base_url else None,
        "fallback_api_key": str(fallback_api_key) if fallback_api_key else None,
        "capability_probe": capability_probe,
    })
    _init_base_url = getattr(getattr(_llm_client, "_async_client", None), "base_url", None)
    logger.info(
        f"LLM global client initialized. base_url={_init_base_url}, "
        f"model={_llm_client.default_model}"
    )
    return _llm_client


def get_llm_client() -> Optional[LLMClient]:
    """获取全局 LLM 客户端实例。"""
    if _llm_client is None:
        logger.warning("LLM client accessed before initialization or was not configured.")
    return _llm_client


def _reset_for_test() -> None:
    """测试用——重置全局 LLM 单例。"""
    global _llm_client
    _llm_client = None
    reset_registry()
