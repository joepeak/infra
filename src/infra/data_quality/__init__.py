#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataQuality 枚举
描述单次数值数据的质量状态
"""

from enum import StrEnum


class DataQuality(StrEnum):
    """数据质量状态（8 状态）

    继承 StrEnum（Python 3.11+）让 str(成员) 自动返回成员值（如 "invalid"），
    而不是 "DataQuality.INVALID"。同时保留 str 行为：成员可与 str 直接 == 比较。
    """
    UNKNOWN = "unknown"         # 尚未检查
    GOOD = "good"               # 数据正常、有效、符合预期
    DELAYED = "delayed"         # 数据源应更新但尚未更新
    STALE = "stale"             # 数据有效，但已明显过期
    PARTIAL = "partial"         # 数据不完整，但已有数据可能正确
    SUSPICIOUS = "suspicious"   # 数据存在，但数值存在异常嫌疑
    INVALID = "invalid"         # 数据存在，但格式/数值异常
    MISSING = "missing"         # 当前没有有效数据

    # 合并优先级：INVALID > MISSING > PARTIAL > SUSPICIOUS > STALE > DELAYED > GOOD > UNKNOWN
    @classmethod
    def merge(cls, qualities: list["DataQuality"]) -> "DataQuality":
        """取最高严重级别"""
        priority = [
            cls.INVALID, cls.MISSING, cls.PARTIAL, cls.SUSPICIOUS,
            cls.STALE, cls.DELAYED, cls.GOOD, cls.UNKNOWN,
        ]
        for p in priority:
            if p in qualities:
                return p
        return cls.UNKNOWN