"""infra.llm.models：LLM 通用数据模型（pydantic）。

设计：
- 跨 provider 通用（OpenAI、Anthropic、DeepSeek、local...）
- 请求/响应/选项严格类型化
- from_openai_response 适配器：OpenAI 响应 → LLMResponse
- from_anthropic_response：待实现（如果未来要支持）
"""
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from infra.utils.retry import RetryConfig

UsageCallback = Callable[[Dict[str, Any]], None]

Modality = Literal["text", "image", "video", "audio"]


class LLMRequest(BaseModel):
    """LLM 请求参数（跨 provider 通用）。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[Dict[str, Any]] = Field(..., description="消息列表（role + content）")
    model: Optional[str] = Field(None, description="模型名（覆盖默认）")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    top_p: float = Field(default=1.0, ge=0, le=1)
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具定义")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None)
    response_format: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "response_format（透传给 provider 的原始 dict）——"
            "如 {'type': 'json_object'} 或 "
            "{'type': 'json_schema', 'json_schema': {...}}（嵌套结构）"
        ),
    )
    timeout: Optional[float] = Field(None, description="单次调用超时（秒）")
    extra_body: Optional[Dict[str, Any]] = Field(default=None, description="Extra parameters passed to the LLM API (e.g. thinking control for DeepSeek)")
    retry_config: Optional[RetryConfig] = Field(default=None, description="Override retry config for this request (defaults to LLMRetryConfig)")
    post_processor: Optional[Callable[[str, Optional[str]], str]] = Field(
        default=None,
        description="内容后处理器：接收 (content, reasoning_content)，返回清洗后的字符串，用于过滤 CoT 等",
    )
    modality: Modality = Field(default="text", description="请求模态：text|image|video|audio")
    json_mode: Optional[str] = Field(
        None,
        description="JSON 输出策略：'json_object' | 'json_schema' | 'prompt_only' | None（自动）",
    )

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """转为 OpenAI 风格 kwargs（messages + model 在外层处理，避免重复传参）。"""
        kwargs = self.model_dump(exclude_none=True, exclude={"retry_config", "model", "post_processor", "modality", "json_mode"})
        msgs = kwargs.pop("messages", None)
        if msgs is not None:
            kwargs["messages"] = msgs
        return kwargs


class LLMResponse(BaseModel):
    """LLM 响应（统一格式）。"""
    content: str = Field(default="", description="文本内容")
    reasoning_content: Optional[str] = Field(default=None, description="OpenAI reasoning 字段内容")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="工具调用列表")
    usage: Dict[str, int] = Field(default_factory=dict, description="token 用量")
    model: Optional[str] = Field(None, description="实际使用的模型名")
    finish_reason: Optional[str] = Field(None)

    @property
    def total_tokens(self) -> int:
        """总 token 数。

        优先用 provider 返回的 ``total_tokens``；缺失时回退到
        ``prompt_tokens + completion_tokens``。
        """
        total = self.usage.get("total_tokens")
        if total:
            return int(total)
        return int(
            self.usage.get("prompt_tokens", 0)
            + self.usage.get("completion_tokens", 0)
        )

    @property
    def prompt_tokens(self) -> int:
        """输入 token 数（缺失时 0）。"""
        return int(self.usage.get("prompt_tokens", 0))

    @property
    def completion_tokens(self) -> int:
        """输出 token 数（缺失时 0）。"""
        return int(self.usage.get("completion_tokens", 0))

    @property
    def is_truncated(self) -> bool:
        """输出是否因长度上限被截断（finish_reason == "length"）。"""
        return self.finish_reason == "length"

    @property
    def is_success(self) -> bool:
        """是否正常结束（finish_reason 为 stop / tool_calls / None）。"""
        return self.finish_reason in (None, "", "stop", "tool_calls")

    @staticmethod
    def _get_value(source: Any, name: str) -> Any:
        if isinstance(source, dict):
            return source.get(name)
        return getattr(source, name, None)

    @classmethod
    def _extract_reasoning_content(cls, message: Any) -> Optional[str]:
        reasoning = cls._get_value(message, "reasoning_content") or cls._get_value(message, "reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

        reasoning_details = cls._get_value(message, "reasoning_details") or []
        if not reasoning_details:
            return None

        parts = []
        for detail in reasoning_details:
            text = cls._get_value(detail, "text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return "\n".join(parts) if parts else None


    @classmethod
    def from_openai_response(
        cls, response: Any, post_processor: Optional[Callable[[str, Optional[str]], str]] = None
    ) -> "LLMResponse":
        """从 OpenAI 风格响应构造 LLMResponse（完整响应，非流式）。"""
        message = response.choices[0].message
        reasoning_content = cls._extract_reasoning_content(message)
        content = cls._get_value(message, "content") or ""
        if not content and reasoning_content:
            content = reasoning_content

        if post_processor is not None:
            content = post_processor(content, reasoning_content)

        result: Dict[str, Any] = {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": [],
            "model": getattr(response, "model", None),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        }

        # token 用量
        # 注意：provider 可能不返回 total_tokens（部分端点返回 0），
        # 此时回退到 prompt + completion，避免调用方直接读 usage["total_tokens"] 拿到 0。
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if not total_tokens:
                total_tokens = prompt_tokens + completion_tokens
            result["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        # 工具调用
        if hasattr(message, "tool_calls") and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return cls(**result)


    @classmethod
    def from_openai_chunk(
        cls,
        chunk: Any,
        post_processor: Optional[Callable[[str, Optional[str]], str]] = None,
    ) -> "LLMResponse":
        """从 OpenAI 流式响应的单个 chunk 构造 LLMResponse。

        流式 chunk 结构不同于完整响应：
        - choices[0].delta（而非 .message）
        - delta 可能只有部分字段（content、role、tool_calls、reasoning_content 等）
        - usage 通常在最后一个 chunk 中
        """
        choice = chunk.choices[0] if chunk.choices else None
        delta = getattr(choice, "delta", None) if choice else None

        content = ""
        reasoning_content = None
        tool_calls = []

        if delta is not None:
            content = cls._get_value(delta, "content") or ""
            reasoning_content = cls._extract_reasoning_content(delta)
            if not content and reasoning_content:
                content = reasoning_content

            # 流式 tool_calls 增量累加
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    tool_call_data = {
                        "index": getattr(tc, "index", 0),
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", "") or "",
                            "arguments": getattr(getattr(tc, "function", None), "arguments", "") or "",
                        },
                    }
                    tool_calls.append(tool_call_data)

        if post_processor is not None:
            content = post_processor(content, reasoning_content)

        result: Dict[str, Any] = {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "model": getattr(chunk, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
        }

        # token 用量（通常在最后一个 chunk）
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            if not total_tokens:
                total_tokens = prompt_tokens + completion_tokens
            result["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        return cls(**result)


class MediaRequest(BaseModel):
    """Multimodal generation request (image / video / audio).

    Mirrors OpenAI-compatible media endpoints (DALL·E, Seedream, etc.).
    Only ``prompt`` is universally required; provider-specific fields (size,
    quality, n, reference images, seed) are passed through ``extra_body``.
    """
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    if TYPE_CHECKING:
        from infra.utils.retry import RetryConfig

    prompt: str = Field(..., description="Text prompt describing the desired output")
    model: Optional[str] = Field(None, description="Model name override")
    modality: Modality = Field("text", description="Generation modality")
    size: Optional[str] = Field(None, description="Output size, e.g. '1024x768'")
    quality: Optional[str] = Field(None, description="Quality hint (e.g. 'standard' / 'hd')")
    n: int = Field(default=1, description="Number of images to generate (image modality)")
    reference_images: Optional[List[str]] = Field(default=None, description="URL list of reference images")
    seed: Optional[int] = Field(None, description="Random seed (if supported by provider)")
    response_format: Optional[str] = Field(default="url", description="b64_json | url")
    extra_body: Optional[Dict[str, Any]] = Field(default=None, description="Provider-specific extras")
    timeout: Optional[float] = Field(None, description="Per-call timeout (seconds)")
    retry_config: Optional[RetryConfig] = Field(default=None, description="Override retry config (defaults to MediaClient.retry_config)")


class MediaResponse(BaseModel):
    """Unified media generation response."""
    # existing_data_ok 是 pydantic 运行时支持的合法配置项，
    # 但 pydantic 2.13.5 的 ConfigDict TypedDict 尚未收录它 -> 类型检查报
    # typeddict-unknown-key。这里保留配置（运行时有效），仅屏蔽类型告警。
    model_config = ConfigDict(existing_data_ok=True)  # type: ignore[typeddict-unknown-key]

    modality: Modality = Field(default="text", description="The modality that was generated")
    model: Optional[str] = Field(None, description="Actual model name returned by provider")
    images: List["ImageResult"] = Field(default_factory=list, description="Generated image results (image modality)")
    videos: List["VideoResult"] = Field(default_factory=list, description="Video task results (video modality)")
    audios: List["AudioResult"] = Field(default_factory=list, description="Audio results (audio modality)")
    cost: Optional[float] = Field(None, description="Provider-returned cost (if available)")
    currency: Optional[str] = Field(None, description="Cost currency code (e.g., 'USD', 'CNY')")
    cost_metadata: Optional[Dict[str, Any]] = Field(None, description="Provider-specific cost info from API response")


class ImageResult(BaseModel):
    """A single generated image."""
    # existing_data_ok 是 pydantic 运行时支持的合法配置项，
    # 但 pydantic 2.13.5 的 ConfigDict TypedDict 尚未收录它 -> 类型检查报
    # typeddict-unknown-key。这里保留配置（运行时有效），仅屏蔽类型告警。
    model_config = ConfigDict(existing_data_ok=True)  # type: ignore[typeddict-unknown-key]

    url: Optional[str] = Field(None, description="Publicly reachable URL to the image")
    b64_json: Optional[str] = Field(None, description="Base64-encoded image data (when response_format=b64_json)")
    revised_prompt: Optional[str] = Field(None, description="Provider-refined prompt (DALL·E)")
    seed: Optional[int] = Field(None, description="Seed used for generation (for reproducibility)")


class VideoResult(BaseModel):
    """A video generation task (async, 4-10 minutes)."""

    task_id: Optional[str] = Field(None, description="Task ID for polling status")
    status: str = Field("pending", description="pending | running | completed | failed | cancelled")
    video_url: Optional[str] = Field(None, description="Final video URL once completed")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail URL")
    progress: Optional[float] = Field(None, description="Progress 0.0–1.0 (if supported)")
    error: Optional[str] = Field(None, description="Error message if status=failed")
    estimated_remaining_seconds: Optional[int] = Field(None, description="ETA when running")
    duration_seconds: Optional[float] = Field(None, description="Duration of the generated video in seconds")


class AudioResult(BaseModel):
    """A single generated audio output."""
    # existing_data_ok 是 pydantic 运行时支持的合法配置项，
    # 但 pydantic 2.13.5 的 ConfigDict TypedDict 尚未收录它 -> 类型检查报
    # typeddict-unknown-key。这里保留配置（运行时有效），仅屏蔽类型告警。
    model_config = ConfigDict(existing_data_ok=True)  # type: ignore[typeddict-unknown-key]

    url: Optional[str] = Field(None, description="Publicly reachable URL to the audio")
    b64_json: Optional[str] = Field(None, description="Base64-encoded audio data (when response_format=b64_json)")
    duration_seconds: Optional[float] = Field(None, description="Duration of the audio in seconds")
    voice: Optional[str] = Field(None, description="Voice identifier used")
    seed: Optional[int] = Field(None, description="Seed used for generation (for reproducibility)")


MediaResponse.model_rebuild()
