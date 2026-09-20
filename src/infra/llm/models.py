"""infra.llm.models：LLM 通用数据模型（pydantic）。

设计：
- 跨 provider 通用（OpenAI、Anthropic、DeepSeek、local...）
- 请求/响应/选项严格类型化
- from_openai_response 适配器：OpenAI 响应 → LLMResponse
- from_anthropic_response：待实现（如果未来要支持）
"""
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from infra.utils.retry import RetryConfig


class LLMRequest(BaseModel):
    """LLM 请求参数（跨 provider 通用）。"""
    messages: List[Dict[str, Any]] = Field(..., description="消息列表（role + content）")
    model: Optional[str] = Field(None, description="模型名（覆盖默认）")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    top_p: float = Field(default=1.0, ge=0, le=1)
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具定义")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None)
    response_format: Optional[Dict[str, str]] = Field(None, description="response_format（如 OpenAI 的 json_object）")
    timeout: Optional[float] = Field(None, description="单次调用超时（秒）")
    extra_body: Optional[Dict[str, Any]] = Field(default=None, description="Extra parameters passed to the LLM API (e.g. thinking control for DeepSeek)")
    retry_config: Optional[Any] = Field(default=None, description="Override retry config for this request (defaults to LLMRetryConfig)")

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """转为 OpenAI 风格 kwargs（messages + model 在外层处理，避免重复传参）。"""
        kwargs = self.model_dump(exclude_none=True, exclude={"retry_config", "model"})
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
    def from_openai_response(cls, response: Any) -> "LLMResponse":
        """从 OpenAI 风格响应构造 LLMResponse。"""
        message = response.choices[0].message
        reasoning_content = cls._extract_reasoning_content(message)
        content = cls._get_value(message, "content") or ""
        if not content and reasoning_content:
            content = reasoning_content

        result: Dict[str, Any] = {
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": [],
            "model": getattr(response, "model", None),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
        }

        # token 用量
        usage = getattr(response, "usage", None)
        if usage is not None:
            result["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
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
