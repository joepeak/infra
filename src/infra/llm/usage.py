"""infra.llm.usage：进程内 Token 用量聚合。

设计：
- ContextVar 标记当前上下文用途（按用途分别累计）
- threading.Lock 保证多线程安全
- defaultdict 累加——按用途返回快照
- 跨项目通用——任何 LLM 项目都可观测成本

注意：本聚合是进程内——多进程/多机器需额外方案（DB/Redis 持久化）。
"""
import logging
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import Dict, Optional

logger = logging.getLogger(__name__)


_usage_lock = threading.Lock()
_usage_totals: Dict[str, Dict[str, int]] = defaultdict(
    lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
)
_current_purpose: ContextVar = ContextVar("_llm_purpose", default="unknown")


def set_llm_purpose(purpose: str):
    """设置当前上下文的 LLM 用途标签（如 'write_scene' / 'evaluate_node'）。

    用法（with 风格）：
        token = set_llm_purpose("writing")
        try:
            ...  # LLM 调用
        finally:
            reset_llm_purpose(token)
    """
    return _current_purpose.set(purpose)


def reset_llm_purpose(token):
    """恢复用途标签上下文。"""
    if token is not None:
        _current_purpose.reset(token)


def record_usage(usage: Dict[str, int], purpose: Optional[str] = None) -> None:
    """记录一次调用的 token 用量到进程内聚合器。

    Args:
        usage: token 用量 dict（至少含 'prompt_tokens' / 'completion_tokens' / 'total_tokens'）
        purpose: 覆盖当前上下文的用途标签（None=用当前 ContextVar）
    """
    p = purpose or _current_purpose.get()
    with _usage_lock:
        agg = _usage_totals[p]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            agg[k] += usage.get(k, 0)
        agg["calls"] += 1


def get_usage_summary() -> Dict[str, Dict[str, int]]:
    """返回按用途汇总的 token 用量快照（深拷贝）。"""
    with _usage_lock:
        return {k: dict(v) for k, v in _usage_totals.items()}


def reset_usage_summary() -> None:
    """清空所有累计（测试用）。"""
    with _usage_lock:
        _usage_totals.clear()


def log_usage_summary() -> None:
    """输出当前累计用量的 INFO 日志（供工作流结束或定时触发）。"""
    summary = get_usage_summary()
    if not summary:
        return
    total = sum(v["total_tokens"] for v in summary.values())
    lines = [
        f"  - {k}: calls={v['calls']}, prompt={v['prompt_tokens']}, "
        f"completion={v['completion_tokens']}, total={v['total_tokens']}"
        for k, v in sorted(summary.items())
    ]
    logger.info(
        f"💰 LLM Token 用量汇总（进程内累计，总计 {total} tokens）:\n"
        + "\n".join(lines)
    )
