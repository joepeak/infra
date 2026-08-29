"""infra.utils.time_util：时区转换工具（函数式）。

20260829 refactor：从 infra.time_util.TimeUtil 类（静态方法）改为函数式 API。
- 更 Pythonic（PEP 8 推荐小工具用函数）
- 命名空间更干净（无需 `TimeUtil.` 前缀）
- 行为完全兼容

用法：
    from infra.utils.time_util import to_utc_event_time
    utc_dt = to_utc_event_time("2026-08-29", "Asia/Shanghai")
"""
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo


def to_utc_event_time(val: str | datetime | date, source_tz: str = "UTC") -> datetime:
    """
    统一将各种输入格式（字符串、datetime、date）和各种源头时区转为标准 UTC datetime。

    Args:
        val: 输入值（字符串 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"，或 datetime/date 对象）
        source_tz: 源头时区——"UTC" / "Asia/Shanghai" / 其他时区名（用 zoneinfo）

    Returns:
        带 UTC 时区的 datetime 对象
    """
    if isinstance(val, str):
        # 处理仅有年月日的情况（如 FRED 宏观数据的 "2026-08-01"）
        if len(val) == 10:
            dt = datetime.strptime(val, "%Y-%m-%d")
        else:
            dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
    elif isinstance(val, date) and not isinstance(val, datetime):
        # date 对象转为 datetime
        dt = datetime.combine(val, datetime.min.time())
    else:
        dt = val

    # 如果没有时区信息，加上源头时区信息
    if dt.tzinfo is None:
        if source_tz == "UTC":
            dt = dt.replace(tzinfo=timezone.utc)
        elif source_tz == "Asia/Shanghai":  # A股/中国宏观
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        else:  # 美股等 (America/New_York)
            dt = dt.replace(tzinfo=ZoneInfo(source_tz))

    # 统一转换为 UTC 时区
    return dt.astimezone(timezone.utc)
