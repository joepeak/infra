"""LLM Provider 能力注册表。

背景：不同厂商/端点对 OpenAI 兼容协议的实现度不同——
  - 部分端点不接受 ``tools``（函数调用）
  - 部分端点不接受 ``response_format={"type":"json_object"}``
  - 思考型模型（DeepSeek thinking / Qwen enable_thinking 等）在开启思考时
    往往不允许强制 ``tool_choice``，二者组合会直接 400

历史实现是"每次调用都靠试错"：撞到 ``BadRequestError`` 才降级，且降级标志只挂在
Agent 实例上——换个 Agent、换个节点又要重撞一遍，白白浪费请求配额与延迟。

本模块把"某 (base_url, model) 组合支持什么"集中缓存，两种来源：
  - ``learn()``：从真实调用的错误里学习（零额外请求，鲁棒）
  - ``set()``  ：录入主动探测结果（可选，见 ``OpenAIClient.probe_capabilities``）
调用方通过 ``get()`` 读取已知能力；未知（``None``）时按乐观默认处理，命中真实错误
再 ``learn()`` 回填，下一次调用即免于试错。

设计约束：
  - 纯数据 + 进程内缓存，无外部依赖，不 import ``client``（避免循环导入）
  - 探测/学习失败绝不影响主流程：最坏退化为"未知 → 乐观默认 → 试错"
  - 缓存键不含 api_key，避免密钥泄漏进日志/快照
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class ProviderCapabilities:
    """单个 (base_url, model) 组合的能力画像。

    字段用 ``Optional[bool]``：``None`` 表示"尚未探明"，True/False 为已确认结论。
    """
    #: 是否接受 tools（函数调用）。None=未知
    supports_tools: Optional[bool] = None
    #: 是否接受 response_format={"type":"json_object"}。None=未知
    supports_response_format: Optional[bool] = None
    #: 思考模式是否与强制 tool_choice 冲突（DeepSeek 系典型）。None=未知
    thinking_conflicts_with_tools: Optional[bool] = None
    #: thinking 参数本身是否被端点拒收（需要彻底去掉 extra_body.thinking）。None=未知
    thinking_param_rejected: Optional[bool] = None
    #: 是否不接受强制 tool_choice（与 thinking 无关的端点限制）。None=未知
    tool_choice_unsupported: Optional[bool] = None
    #: 最近一次学习到的错误摘要（仅供排查用，不参与判定）
    last_error: Optional[str] = None
    #: 数据来源标记：'probe' | 'learn' | None
    source: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[object]]:
        return asdict(self)

    #: 参与"是否有新结论"判定的字段（排除 last_error/source 这类噪声）
    DECISION_FIELDS = (
        "supports_tools",
        "supports_response_format",
        "thinking_conflicts_with_tools",
        "thinking_param_rejected",
        "tool_choice_unsupported",
    )


class ProviderCapabilityRegistry:
    """进程内 (base_url, model) → 能力 的缓存注册表（线程安全）。"""

    def __init__(self) -> None:
        self._caps: Dict[str, ProviderCapabilities] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(base_url: Optional[str], model: Optional[str]) -> str:
        # 尾部斜杠归一化，避免 "http://x/v1" 与 "http://x/v1/" 记成两条
        base = (base_url or "").rstrip("/")
        return f"{base}::{model or ''}"

    def get(self, base_url: Optional[str], model: Optional[str]) -> ProviderCapabilities:
        """读取能力画像；未知组合返回全 None 的空画像（不写入缓存）。"""
        with self._lock:
            caps = self._caps.get(self.key(base_url, model))
            if caps is None:
                return ProviderCapabilities()
            # 返回副本，防止调用方误改缓存
            return ProviderCapabilities(**caps.as_dict())

    def update(self, base_url: Optional[str], model: Optional[str], *, source: str, **changes) -> ProviderCapabilities:
        """合并式回填能力。仅覆盖显式传入且非 None 的字段。"""
        with self._lock:
            k = self.key(base_url, model)
            caps = self._caps.get(k) or ProviderCapabilities()
            for field_name, value in changes.items():
                if value is not None and hasattr(caps, field_name):
                    setattr(caps, field_name, value)
            if source:
                caps.source = source
            self._caps[k] = caps
            return ProviderCapabilities(**caps.as_dict())

    def learn_from_error(self, base_url: Optional[str], model: Optional[str], error_message: str) -> ProviderCapabilities:
        """从真实调用的 BadRequestError 文本里学习能力，回填缓存（状态感知）。

        按"降级阶梯"推进——重复命中同一错误时会推动到下一级，保证每次学习
        都有新结论（否则调用方会空转重试）：
          1. thinking + tool_choice 冲突 → thinking_conflicts_with_tools=True
             （若已为 True 仍报同一错误 → thinking_param_rejected=True，彻底移除）
          2. thinking 被拒收             → thinking_param_rejected=True
          3. tool_choice 单独被拒收      → tool_choice_unsupported=True
          4. response_format/json_object → supports_response_format=False
          5. tools 单独被拒收            → supports_tools=False
        无法归类时只记录 last_error，不改变结论。
        """
        msg = (error_message or "").lower()
        with self._lock:
            k = self.key(base_url, model)
            caps = self._caps.get(k) or ProviderCapabilities()

            if "tool_choice" in msg and "thinking" in msg:
                if caps.thinking_conflicts_with_tools:
                    # 已禁用 thinking 仍冲突：说明 thinking 参数本身不被接受
                    caps.thinking_param_rejected = True
                else:
                    caps.thinking_conflicts_with_tools = True
            elif "thinking" in msg:
                caps.thinking_param_rejected = True
            elif "tool_choice" in msg:
                caps.tool_choice_unsupported = True

            # 注意排除"请求姿势"类错误：如 DeepSeek 的
            # "Prompt must contain the word 'json' ... to use 'response_format'"
            # 那是 prompt 写法问题，不代表端点不支持 response_format。
            if ("response_format" in msg or "json_object" in msg) and "must contain the word" not in msg:
                caps.supports_response_format = False
            if "tools" in msg and "thinking" not in msg and "tool_choice" not in msg:
                caps.supports_tools = False

            caps.last_error = (error_message or "")[:200]
            caps.source = "learn"
            self._caps[k] = caps
            return ProviderCapabilities(**caps.as_dict())

    def snapshot(self) -> Dict[str, Dict[str, Optional[object]]]:
        """导出全部已学习能力（供日志/诊断）。"""
        with self._lock:
            return {k: v.as_dict() for k, v in self._caps.items()}

    def clear(self) -> None:
        """清空缓存（测试用）。"""
        with self._lock:
            self._caps.clear()


# 进程内单例
_registry = ProviderCapabilityRegistry()


def get_registry() -> "ProviderCapabilityRegistry":
    """获取全局能力注册表。"""
    return _registry


def reset_registry() -> None:
    """重置全局注册表（测试用）。"""
    _registry.clear()
