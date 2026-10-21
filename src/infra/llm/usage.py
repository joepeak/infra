"""infra.llm.usage：进程内 Token 用量聚合。

设计：
- ContextVar 标记当前上下文用途（按用途分别累计）
- threading.Lock 保证多线程安全
- defaultdict 累加——按用途返回快照
- 跨项目通用——任何 LLM 项目都可观测成本

注意：本聚合是进程内——多进程/多机器需额外方案（DB/Redis 持久化）。
"""
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import Any, Dict, Optional

from infra.logger import get_logger

# 用 infra 的 LazyLogger（而非原生 logging.getLogger）——
# 原生 logger 没有 handler，LLM 层的日志不会落进 infra 的日志文件。
logger = get_logger(__name__)


_usage_lock = threading.Lock()
_usage_totals: Dict[str, Dict[str, int]] = defaultdict(
    lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
)
_current_purpose: ContextVar = ContextVar("_llm_purpose", default="unknown")


def _to_int(value: Any) -> int:  # noqa: ANN401
    """把用量数值安全转成 ``int``。

    provider 可能返回 ``None``（部分端点不返回 total_tokens），
    直接参与 ``+=`` 会抛 TypeError——让整次调用的用量统计丢失。
    非数值输入按 0 处理，不让统计逻辑影响主流程。
    """
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def set_llm_purpose(purpose: str) -> Any:  # noqa: ANN401  # 返 ContextVar Token
    """设置当前上下文的 LLM 用途标签（如 'write_scene' / 'evaluate_node'）。

    用法（with 风格）：
        token = set_llm_purpose("writing")
        try:
            ...  # LLM 调用
        finally:
            reset_llm_purpose(token)
    """
    return _current_purpose.set(purpose)


def reset_llm_purpose(token: Any) -> None:
    """恢复用途标签上下文。"""
    if token is not None:
        _current_purpose.reset(token)


def record_usage(usage: Dict[str, int], purpose: Optional[str] = None) -> None:
    """记录一次调用的 token 用量到进程内聚合器。

    Args:
        usage: token 用量 dict（至少含 'prompt_tokens' / 'completion_tokens' / 'total_tokens'）
        purpose: 覆盖当前上下文的用途标签（None=用当前 ContextVar）

    Note:
        数值会强制转 ``int``——provider 可能返回 ``None``（部分端点不返回
        total_tokens），直接累加会抛 TypeError 让整次调用的用量统计丢失。
    """
    p = purpose or _current_purpose.get()
    with _usage_lock:
        agg = _usage_totals[p]
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            agg[k] += _to_int(usage.get(k))
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


# ===========================================================================
# Media 用量聚合（按次 / 按张 / 按时长计费——与 token 是不同的计费维度）
# ===========================================================================
#
# 为什么与 token 聚合分开：
#   图像 / 视频 / 音频**不按 token 计费**——图像按张、视频按时长、音频按字符。
#   把它们塞进 token 维度（prompt/completion/total_tokens）会导致：
#     1. token 恒为 0，语义上无意义；
#     2. 「调用次数」混进 token 字典，调用方无法区分
#        「1 次媒体调用」和「1 次零 token 的文本调用」；
#     3. 真正决定成本的字段（张数 / 时长 / 金额）全部丢失。
#   因此这里用**独立的聚合器**，只累计计费维度，不碰 token 聚合器。
#
# 两个关键设计：
#   1. **金额按币种分桶**——不同 provider 可能返回 USD / CNY，
#      累加成单一数字没有意义（1 USD + 1 CNY ≠ 2）。用 Dict[币种, 金额]。
#   2. **计费单位多维并存**——一次调用可能同时产生多个计费维度
#      （如「2 张图 + 1 次高清渲染」），用 Dict[单位名, 数量] 而非单一数值。

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MediaUsage:
    """单个模态的媒体用量累计快照。"""

    #: 生成任务数——视频按「提交任务」计一次，不按轮询次数计
    calls: int = 0
    #: 各计费维度的累计数量——键是单位名（"images" / "seconds" / "chars"）
    units: Dict[str, float] = field(default_factory=dict)
    #: provider 返回的真实金额——**按币种分桶**，不同币种不混算
    cost_by_currency: Dict[str, float] = field(default_factory=dict)
    #: 最近一次使用到的模型（便于排查成本构成）
    model: str = ""

    def copy(self) -> "MediaUsage":
        """深拷贝快照——防止调用方误改聚合器内部状态。"""
        return MediaUsage(
            calls=self.calls,
            units=dict(self.units),
            cost_by_currency=dict(self.cost_by_currency),
            model=self.model,
        )


#: 各模态的默认计费单位名
_MEDIA_UNIT_NAMES: Dict[str, str] = {
    "image": "images",
    "video": "seconds",
    "audio": "chars",
}

_media_lock = threading.Lock()
_media_totals: Dict[str, MediaUsage] = {}


def record_media_task(
    modality: str,
    model: str = "",
    units: Optional[Dict[str, float]] = None,
    cost: Optional[float] = None,
    currency: Optional[str] = None,
) -> None:
    """记录**一次媒体生成任务**。

    用于「任务开始时」调用——视频的提交、图像的生成请求等。

    Args:
        modality: 模态（"image" / "video" / "audio"）
        model: 使用的模型名（用于成本构成分析）
        units: 计费维度数量，如 ``{"images": 2}`` / ``{"seconds": 5}``。
            省略时用该模态的默认单位，数量为 0。
        cost: provider 返回的真实金额（没有则传 None）
        currency: 金额币种（provider 返回的为准，如 "USD" / "CNY"）
    """
    key = f"media_{modality}"
    with _media_lock:
        agg = _media_totals.get(key)
        if agg is None:
            agg = MediaUsage()
            _media_totals[key] = agg
        agg.calls += 1
        if model:
            agg.model = model
        _add_units(agg, units, modality)
        _add_cost(agg, cost, currency)


def record_media_billable(
    modality: str,
    units: Optional[Dict[str, float]] = None,
    cost: Optional[float] = None,
    currency: Optional[str] = None,
) -> None:
    """补记**已计费任务**的计费维度与金额——**不重复计任务数**。

    用于「任务完成后」补记实际产生的计费量。典型场景：视频任务提交时
    不知道最终时长，轮询完成后才知道——这时补记秒数与真实金额，
    但任务数已经在提交时计过，不能再计一次。

    Args:
        modality: 模态（"image" / "video" / "audio"）
        units: 补记的计费维度数量，如 ``{"seconds": 5}``
        cost: provider 返回的真实金额
        currency: 金额币种
    """
    key = f"media_{modality}"
    with _media_lock:
        agg = _media_totals.get(key)
        if agg is None:
            agg = MediaUsage()
            _media_totals[key] = agg
        _add_units(agg, units, modality)
        _add_cost(agg, cost, currency)


def _add_units(
    agg: MediaUsage,
    units: Optional[Dict[str, float]],
    modality: str,
) -> None:
    """把计费维度累加进聚合项。"""
    if not units:
        return
    for name, value in units.items():
        try:
            agg.units[name] = agg.units.get(name, 0.0) + float(value)
        except (TypeError, ValueError):
            continue


def _add_cost(
    agg: MediaUsage,
    cost: Optional[float],
    currency: Optional[str],
) -> None:
    """把金额按币种累加——**不同币种不混算**。"""
    if cost is None:
        return
    try:
        amount = float(cost)
    except (TypeError, ValueError):
        return
    if not currency:
        # 没有币种信息时归到 "unknown"——不与已知币种混算
        currency = "unknown"
    agg.cost_by_currency[currency] = agg.cost_by_currency.get(currency, 0.0) + amount


def get_media_usage_summary() -> Dict[str, MediaUsage]:
    """返回按模态汇总的媒体用量快照（深拷贝）。"""
    with _media_lock:
        return {k: v.copy() for k, v in _media_totals.items()}


def reset_media_usage_summary() -> None:
    """清空媒体用量累计（测试用）。"""
    with _media_lock:
        _media_totals.clear()


def log_media_usage_summary() -> None:
    """输出媒体用量累计的 INFO 日志。"""
    summary = get_media_usage_summary()
    if not summary:
        return
    lines: List[str] = []
    for key, usage in sorted(summary.items()):
        units = ", ".join(f"{n}={v:g}" for n, v in sorted(usage.units.items()))
        costs = ", ".join(
            f"{v:g} {c}" for c, v in sorted(usage.cost_by_currency.items())
        )
        parts = [f"calls={usage.calls}"]
        if units:
            parts.append(units)
        if costs:
            parts.append(f"cost={costs}")
        if usage.model:
            parts.append(f"model={usage.model}")
        lines.append(f"  - {key}: " + ", ".join(parts))
    logger.info(
        "🎨 LLM 媒体用量汇总（进程内累计）:\n" + "\n".join(lines)
    )
