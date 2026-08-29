"""infra.llm.models：LLM 通用数据模型（pydantic）。

设计：
- 跨 provider 通用（OpenAI、Anthropic、DeepSeek、local...）
- 请求/响应/选项严格类型化
- from_openai_response 适配器：OpenAI 响应 → LLMResponse
- from_anthropic_response：待实现（如果未来要支持）
"""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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

    def to_openai_kwargs(self) -> Dict[str, Any]:
        """转为 OpenAI 风格 kwargs（messages 必须在最前）。"""
        kwargs = self.model_dump(exclude_none=True)
        # 显式提取 messages 到最前
        msgs = kwargs.pop("messages", None)
        if msgs is not None:
            kwargs["messages"] = msgs
        return kwargs


class LLMResponse(BaseModel):
    """LLM 响应（统一格式）。"""
    content: str = Field(default="", description="文本内容")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="工具调用列表")
    usage: Dict[str, int] = Field(default_factory=dict, description="token 用量")
    model: Optional[str] = Field(None, description="实际使用的模型名")
    finish_reason: Optional[str] = Field(None)

    @classmethod
    def from_openai_response(cls, response: Any) -> "LLMResponse":
        """从 OpenAI 风格响应构造 LLMResponse。"""
        message = response.choices[0].message
        result = {
            "content": message.content or "",
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
