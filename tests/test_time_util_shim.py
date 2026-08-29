#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.time_util 向后兼容 shim 测试

⚠️ 这个模块整体标 @deprecated：
  - infra.time_util.to_utc_event_time 应被理解为直接 re-export 新位置的符号
  - infra.time_util.TimeUtil 类是兼容 shim，调它会发 DeprecationWarning

覆盖：
1. to_utc_event_time 是 infra.utils.time_util 那个的同一函数
2. TimeUtil.to_utc_event_time 调时发 DeprecationWarning
3. TimeUtil.to_utc_event_time 仍然返回正确 UTC datetime
4. __all__ 内容
"""

from __future__ import annotations

import warnings

import pytest


# ============================================================
# 1. to_utc_event_time
# ============================================================
class TestToUtcEventTime:
    def test_is_reexport_from_new_location(self):
        """to_utc_event_time 应与新位置的同一函数对象。"""
        from infra.time_util import to_utc_event_time
        from infra.utils.time_util import to_utc_event_time as new_fn
        assert to_utc_event_time is new_fn

    def test_basic_usage(self):
        """基本调用没问题。"""
        from infra.time_util import to_utc_event_time
        from datetime import datetime, timezone
        result = to_utc_event_time("2026-08-29")
        assert result == datetime(2026, 8, 29, tzinfo=timezone.utc)


# ============================================================
# 2. TimeUtil 类
# ============================================================
class TestTimeUtil:
    def test_deprecation_warning(self):
        from infra.time_util import TimeUtil
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TimeUtil.to_utc_event_time("2026-08-29")
            # 至少一条 DeprecationWarning
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            # 消息提到 TimeUtil
            assert "TimeUtil" in str(dep_warnings[0].message)
            assert "deprecated" in str(dep_warnings[0].message).lower()

    def test_deprecated_recommends_new(self):
        from infra.time_util import TimeUtil
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TimeUtil.to_utc_event_time("2026-08-29")
            msg = str(w[0].message)
            # 推荐用新位置
            assert "infra.utils.time_util" in msg

    def test_still_returns_correct_result(self):
        """TimeUtil 调时仍返回正确结果（与函数式一致）。"""
        from infra.time_util import TimeUtil
        from datetime import datetime, timezone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = TimeUtil.to_utc_event_time("2026-08-29 12:00:00")
        assert result == datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)

    def test_source_tz_param(self):
        from infra.time_util import TimeUtil
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 上海 12:00 → UTC 04:00
            result = TimeUtil.to_utc_event_time("2026-08-29 12:00:00", source_tz="Asia/Shanghai")
        from datetime import datetime, timezone
        assert result == datetime(2026, 8, 29, 4, 0, 0, tzinfo=timezone.utc)

    def test_is_static_method(self):
        """TimeUtil.to_utc_event_time 是 @staticmethod——可不实例化调用。"""
        from infra.time_util import TimeUtil
        # 不传 self 也能调（说明是 staticmethod）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            TimeUtil.to_utc_event_time("2026-08-29")


# ============================================================
# 3. __all__
# ============================================================
class TestAllExports:
    def test_all_contains_expected(self):
        from infra.time_util import __all__
        assert "to_utc_event_time" in __all__
        assert "TimeUtil" in __all__


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
