"""infra.time_util：@deprecated——20260829 改函数式+移 utils。

保留此文件仅为向后兼容：
- 旧代码 `from infra.time_util import TimeUtil` 仍可用
- 新代码应 `from infra.utils.time_util import to_utc_event_time`

未来某版本可删此文件——主项目全部 import 改完即可。
"""
import warnings
from typing import Any, Optional

# Re-export 新位置符号
from infra.utils.time_util import to_utc_event_time  # noqa: F401


class TimeUtil:
    """@deprecated：保留为兼容——请用 `to_utc_event_time()` 函数。"""

    @staticmethod
    def to_utc_event_time(val: Any, source_tz: str = "UTC") -> "datetime":  # type: ignore[name-defined]
        """@deprecated：函数式版本更 Pythonic——用 `infra.utils.time_util.to_utc_event_time`。"""
        warnings.warn(
            "TimeUtil.to_utc_event_time is deprecated; use infra.utils.time_util.to_utc_event_time instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return to_utc_event_time(val, source_tz)


__all__ = ["to_utc_event_time", "TimeUtil"]
