#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.utils.time_util.to_utc_event_time 测试

覆盖：
1. str 格式：YYYY-MM-DD / YYYY-MM-DD HH:MM:SS
2. datetime 已带 tzinfo
3. datetime 无 tzinfo + source_tz 各种取值
4. date 对象
5. ISO 8601 带 T 不被支持（边界）
6. Asia/Shanghai 固定 +8
"""

from __future__ import annotations

from datetime import datetime, date, timezone, timedelta

import pytest

from infra.utils.time_util import to_utc_event_time


# ============================================================
# 1. 字符串输入
# ============================================================
class TestStringInput:
    def test_date_only_string_utc(self):
        dt = to_utc_event_time("2026-08-29")
        assert dt == datetime(2026, 8, 29, 0, 0, 0, tzinfo=timezone.utc)

    def test_datetime_string_utc(self):
        dt = to_utc_event_time("2026-08-29 12:34:56")
        assert dt == datetime(2026, 8, 29, 12, 34, 56, tzinfo=timezone.utc)

    def test_date_string_with_shanghai_tz(self):
        """上海 00:00 → UTC 16:00（前一天）。"""
        dt = to_utc_event_time("2026-08-29", source_tz="Asia/Shanghai")
        assert dt == datetime(2026, 8, 28, 16, 0, 0, tzinfo=timezone.utc)

    def test_datetime_string_with_shanghai_tz(self):
        """上海 12:00 → UTC 04:00（同日）。"""
        dt = to_utc_event_time("2026-08-29 12:00:00", source_tz="Asia/Shanghai")
        assert dt == datetime(2026, 8, 29, 4, 0, 0, tzinfo=timezone.utc)

    def test_date_string_with_america_new_york(self):
        """纽约 00:00 EDT（夏令时 UTC-4）→ UTC 04:00。

        需要 `tzdata` 包（pip install tzdata），否则 ZoneInfo 找不到 America/New_York。
        Python 3.9+ 的 zoneinfo 只自带 UTC，其它时区要 tzdata。
        """
        pytest.importorskip("tzdata")
        # 8 月在夏令时（EDT, UTC-4）
        dt = to_utc_event_time("2026-08-29", source_tz="America/New_York")
        assert dt == datetime(2026, 8, 29, 4, 0, 0, tzinfo=timezone.utc)

    def test_iso_with_t_not_supported(self):
        """ISO 8601 带 T 不被支持 → ValueError。"""
        with pytest.raises(ValueError):
            to_utc_event_time("2026-08-29T10:00:00")

    def test_garbage_string_raises(self):
        with pytest.raises(ValueError):
            to_utc_event_time("not a date")


# ============================================================
# 2. datetime 输入
# ============================================================
class TestDatetimeInput:
    def test_naive_datetime_assumed_utc_by_default(self):
        dt = to_utc_event_time(datetime(2026, 8, 29, 10, 0, 0))
        # 默认 source_tz=UTC，结果与输入等价（带 tzinfo）
        assert dt == datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
        assert dt.tzinfo == timezone.utc

    def test_naive_datetime_with_shanghai(self):
        dt = to_utc_event_time(datetime(2026, 8, 29, 10, 0, 0), source_tz="Asia/Shanghai")
        # 上海 10:00 → UTC 02:00
        assert dt == datetime(2026, 8, 29, 2, 0, 0, tzinfo=timezone.utc)

    def test_aware_datetime_utc_input(self):
        dt_in = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
        dt = to_utc_event_time(dt_in)
        assert dt == dt_in

    def test_aware_datetime_other_tz_converted(self):
        """已带 tzinfo 的 datetime 会被转到 UTC，与 source_tz 无关。"""
        tz_sh = timezone(timedelta(hours=8))
        dt_in = datetime(2026, 8, 29, 10, 0, 0, tzinfo=tz_sh)
        dt = to_utc_event_time(dt_in, source_tz="UTC")  # 即便传 UTC，也按已有 tz 转换
        assert dt == datetime(2026, 8, 29, 2, 0, 0, tzinfo=timezone.utc)


# ============================================================
# 3. date 输入
# ============================================================
class TestDateInput:
    def test_date_utc(self):
        dt = to_utc_event_time(date(2026, 8, 29))
        # date 转 datetime.min.time() → 00:00:00，再按 UTC 加 tzinfo
        assert dt == datetime(2026, 8, 29, 0, 0, 0, tzinfo=timezone.utc)

    def test_date_with_shanghai(self):
        dt = to_utc_event_time(date(2026, 8, 29), source_tz="Asia/Shanghai")
        assert dt == datetime(2026, 8, 28, 16, 0, 0, tzinfo=timezone.utc)


# ============================================================
# 4. 返回值始终带 UTC tzinfo
# ============================================================
class TestReturnType:
    @pytest.mark.parametrize("val", [
        "2026-08-29",
        "2026-08-29 12:00:00",
        datetime(2026, 8, 29),
        date(2026, 8, 29),
    ])
    def test_return_is_utc_aware(self, val):
        dt = to_utc_event_time(val)
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)


# ============================================================
# 5. 无效时区
# ============================================================
class TestInvalidTimezone:
    def test_unknown_zoneinfo_raises(self):
        with pytest.raises(Exception):  # ZoneInfoNotFoundError
            to_utc_event_time("2026-08-29", source_tz="Mars/Olympus_Mons")


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
