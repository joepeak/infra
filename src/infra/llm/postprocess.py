"""infra.llm.postprocess：LLM 响应后处理工具。

提供**启发式**的 chain-of-thought（CoT）检测与清洗。

⚠️ 这些工具不能 100% 可靠 —— 识别 "CoT 还是最终答案" 本质是语义判断。
它们仅作为**应用层钩子**供调用方选择使用：

    from infra.llm.postprocess import strip_cot_after_answer, make_cot_aware_processor

    client.invoke(
        LLMRequest(
            messages=[...],
            post_processor=strip_cot_after_answer,
        )
    )

设计原则
--------
- **保守**：误判 "正常回答" 为 CoT 的代价 > 漏掉真正的 CoT
- **可选**：默认不开启；调用方显式传入 post_processor 才生效
- **无副作用**：原样返回 content，不修改 response 对象
"""
from __future__ import annotations

import re
from typing import Callable, Optional


# ===========================================================================
# CoT 检测
# ===========================================================================

#: 强 CoT 起始模式 —— 出现即判定为 CoT，無須額外條件
_COT_START_PATTERNS_STRONG: list[re.Pattern] = [
    re.compile(r"(?i)^\s*(let me think|let'?s think|i need to|we need to|first,?\s+i'll|to solve\s+this|to answer\s+this)"),
    re.compile(r"(?i)^\s*step\s*\d+\b"),
]

#: 弱 CoT 起始模式 —— 容易與正常回答混淆，需結合長度判斷
_COT_START_PATTERNS_WEAK: list[re.Pattern] = [
    re.compile(r"(?i)^\s*(first,?\s|next,?\s|then,?\s|so,?\s|well,?\s)"),
]

#: CoT 段落中的常见连接词 / 推理标记
_COT_MARKERS: list[str] = [
    "first,", "second,", "third,", "finally,", "in conclusion",
    "to summarize", "summary:", "therefore,", "thus,",
    "so the answer is", "the answer is", "in short,",
]

#: 当 content 超过这个长度且不以常见最终答案特征开头，可能是 CoT
_COT_LENGTH_THRESHOLD = 400


def is_likely_cot(content: str, reasoning_content: Optional[str] = None) -> bool:
    """启发式判断 content 是否可能是 chain-of-thought 泄露。

    返回 True 仅当**多个**信号同时命中，降低误判率。

    判定依据（任一即可）：
    1. content 以**强** CoT 起始模式开头（Let me think / We need to / Step 1 等）。
    2. content 很长 (>400 字符) 且**没有** reasoning_content 且包含 CoT 标记。
    3. content 以**弱** CoT 起始模式开头 **且** 长度 > 400。
    """
    if not content or not content.strip():
        return False

    stripped = content.strip()

    # 信号 1：以强 CoT 起始模式开头 → 直接判定
    for pat in _COT_START_PATTERNS_STRONG:
        if pat.search(stripped):
            return True

    # 信号 2：长 content + 无 reasoning channel + CoT 标记
    if len(stripped) > _COT_LENGTH_THRESHOLD and not reasoning_content:
        has_cot_marker = any(m.lower() in stripped.lower() for m in _COT_MARKERS)
        if has_cot_marker:
            return True

    # 信号 3：以弱 CoT 起始模式开头 + 足够长
    if len(stripped) > _COT_LENGTH_THRESHOLD:
        for pat in _COT_START_PATTERNS_WEAK:
            if pat.search(stripped):
                return True

    return False


def _has_final_answer_marker(content: str) -> bool:
    """检查内容是否包含'最终答案'标志。"""
    lower = content.strip().lower()
    final_markers = [
        "final answer", "the answer is", "so the answer is",
        "answer:", "final:", "conclusion:",
    ]
    return any(m in lower for m in final_markers)


# ===========================================================================
# CoT 清洗
# ===========================================================================

#: 匹配 "final answer:" / "answer:" 等标记后的内容（不强制换行前）
_FINAL_ANSWER_PATTERN = re.compile(
    r"(?i)\b(?:final\s+)?answer\s*[:：]\s*",
)


def _extract_after_final_answer(content: str) -> Optional[str]:
    """从 CoT 内容中提取 'final answer:' 之后的部分。

    返回提取到的内容（去除首尾空白），若找不到则返回 None。
    """
    matches = list(_FINAL_ANSWER_PATTERN.finditer(content))
    if not matches:
        return None
    last_match = matches[-1]
    extracted = content[last_match.end():].strip()
    return extracted if extracted else None


def strip_cot_after_answer(content: str, reasoning_content: Optional[str] = None) -> str:
    """后处理器：如果 content 包含 CoT 并以 'final answer:' 标记结尾，则只保留标记后的内容。

    - **推理型模型**（content 为空）：原样返回 reasoning_content → content（infra.llm 已处理）。
    - **CoT 泄露模型**（content 非空但含 CoT）：尝试提取 final answer 标记后的内容。
    - **正常回答**（content 清晰）：原样返回。
    - **无法确定**：保守返回原 content（避免误伤）。
    """
    if not content:
        return reasoning_content or ""
    if not is_likely_cot(content, reasoning_content):
        return content

    # 尝试提取 final answer 后的内容
    extracted = _extract_after_final_answer(content)
    if extracted:
        return extracted

    # 如果 content 以 CoT 开头但没有 final answer 标记，保守不处理
    # （返回原文 —— 呼叫方的 _is_valid_content 会检测到并触发降级）
    return content


def strip_cot_with_fallback(
    content: str, reasoning_content: Optional[str] = None
) -> str:
    """更激进的 CoT 清洗：如果检测到 CoT 但没有 final answer 标记，则返回 reasoning_content。

    适用于**确信模型会输出 CoT** 但又没有清晰 final answer 标记的场景。
    """
    if not is_likely_cot(content, reasoning_content):
        return content
    extracted = _extract_after_final_answer(content)
    if extracted:
        return extracted
    if reasoning_content:
        return reasoning_content.strip()
    return content


def make_cot_aware_processor(
    *,
    aggressive: bool = False,
    min_length: int = _COT_LENGTH_THRESHOLD,
) -> Callable[[str, Optional[str]], str]:
    """创建一個基於長度閥值的 CoT 清洗后處理器。

    - ``aggressive=True``  : 如検測到 CoT 但無 final answer 標記，嘗試使用 reasoning_content。
    - ``min_length``       : 觸發 CoT 判定所需的最短 content 長度。
    """
    def _processor(content: str, reasoning_content: Optional[str] = None) -> str:
        if not content:
            return reasoning_content or ""
        if not reasoning_content and len(content) < min_length:
            return content
        if not is_likely_cot(content, reasoning_content):
            return content
        extracted = _extract_after_final_answer(content)
        if extracted:
            return extracted
        if aggressive and reasoning_content:
            return reasoning_content.strip()
        return content

    return _processor


# ===========================================================================
# 便捷组合
# ===========================================================================

#: 推薦的預設後處理器 — 保守策略，只提取 final answer
default_cot_processor: Callable[[str, Optional[str]], str] = strip_cot_after_answer

#: 激進策略，會回退到 reasoning_content
aggressive_cot_processor: Callable[[str, Optional[str]], str] = strip_cot_with_fallback
