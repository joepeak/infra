"""infra.time_util：统一时区转换工具。

从 core.db.database_model.py 抽离（20260829 拆包）——独立成类库后 TimeUtil 不应依赖 ORM 类。
"""
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo


class TimeUtil:
    @staticmethod
    def to_utc_event_time(val: str | datetime | date, source_tz: str = "UTC") -> datetime:
        """
        统一将各种输入格式（字符串、datetime、date）和各种源头时区转为标准 UTC datetime
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
