#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataQuality 枚举 + merge 优先级测试

覆盖：
1. 8 个枚举成员值正确
2. str 继承带来的 == / str() 行为
3. merge 的固定优先级（INVALID > MISSING > PARTIAL > SUSPICIOUS > STALE > DELAYED > GOOD > UNKNOWN）
4. merge 边界：空列表、单元素、列表顺序影响结果
"""

from __future__ import annotations

import pytest

from infra.data_quality import DataQuality


# ============================================================
# 1. 枚举成员值
# ============================================================
class TestDataQualityMembers:
    def test_all_eight_members_exist(self):
        expected = {
            "UNKNOWN", "GOOD", "DELAYED", "STALE",
            "PARTIAL", "SUSPICIOUS", "INVALID", "MISSING",
        }
        actual = {m.name for m in DataQuality}
        assert actual == expected

    def test_member_string_values(self):
        # 锁定具体值（外部可能用这些字符串做序列化/落库）
        assert DataQuality.UNKNOWN.value == "unknown"
        assert DataQuality.GOOD.value == "good"
        assert DataQuality.DELAYED.value == "delayed"
        assert DataQuality.STALE.value == "stale"
        assert DataQuality.PARTIAL.value == "partial"
        assert DataQuality.SUSPICIOUS.value == "suspicious"
        assert DataQuality.INVALID.value == "invalid"
        assert DataQuality.MISSING.value == "missing"


# ============================================================
# 2. str 继承行为
# ============================================================
class TestDataQualityStrBehavior:
    def test_eq_with_plain_string(self):
        """继承自 str，可以与字符串直接比较。"""
        assert DataQuality.INVALID == "invalid"
        assert DataQuality.GOOD == "good"
        # 与非成员字符串不相等
        assert (DataQuality.GOOD == "bad") is False

    def test_str_returns_value(self):
        """str(成员) 返回 value（StrEnum 在 Python 3.11+ 保证这一行为）。"""
        assert str(DataQuality.INVALID) == "invalid"
        assert str(DataQuality.MISSING) == "missing"

    def test_can_be_used_in_set(self):
        """str 行为 → 可哈希、可入 set。"""
        s = {DataQuality.GOOD, DataQuality.BAD if False else DataQuality.GOOD}
        assert len(s) == 1


# ============================================================
# 3. merge 行为
# ============================================================
class TestDataQualityMerge:
    def test_merge_empty_returns_unknown(self):
        assert DataQuality.merge([]) == DataQuality.UNKNOWN

    def test_merge_single(self):
        assert DataQuality.merge([DataQuality.GOOD]) == DataQuality.GOOD
        assert DataQuality.merge([DataQuality.MISSING]) == DataQuality.MISSING

    def test_merge_priority_invalid_wins(self):
        """INVALID 优先级最高。"""
        # 顺序无所谓，INVALID 总在前面
        assert DataQuality.merge([DataQuality.GOOD, DataQuality.INVALID]) == DataQuality.INVALID
        assert DataQuality.merge([DataQuality.INVALID, DataQuality.GOOD]) == DataQuality.INVALID
        # 同时含 INVALID / MISSING → INVALID
        assert DataQuality.merge([DataQuality.MISSING, DataQuality.INVALID]) == DataQuality.INVALID

    def test_merge_priority_missing_beats_others(self):
        """除 INVALID 外，MISSING 优先级最高。"""
        assert DataQuality.merge([DataQuality.GOOD, DataQuality.MISSING]) == DataQuality.MISSING
        assert DataQuality.merge([DataQuality.MISSING, DataQuality.STALE, DataQuality.PARTIAL]) == DataQuality.MISSING

    def test_merge_priority_partial(self):
        """除 INVALID / MISSING 外，PARTIAL 优先级最高。"""
        assert DataQuality.merge([DataQuality.GOOD, DataQuality.PARTIAL]) == DataQuality.PARTIAL
        assert DataQuality.merge([DataQuality.SUSPICIOUS, DataQuality.PARTIAL]) == DataQuality.PARTIAL

    def test_merge_priority_suspicious(self):
        """SUSPICIOUS 高于 STALE/DELAYED/GOOD/UNKNOWN。"""
        assert DataQuality.merge([DataQuality.GOOD, DataQuality.SUSPICIOUS]) == DataQuality.SUSPICIOUS
        assert DataQuality.merge([DataQuality.STALE, DataQuality.SUSPICIOUS]) == DataQuality.SUSPICIOUS
        assert DataQuality.merge([DataQuality.DELAYED, DataQuality.SUSPICIOUS]) == DataQuality.SUSPICIOUS

    def test_merge_priority_stale(self):
        """STALE 高于 DELAYED/GOOD/UNKNOWN。"""
        assert DataQuality.merge([DataQuality.GOOD, DataQuality.STALE]) == DataQuality.STALE
        assert DataQuality.merge([DataQuality.DELAYED, DataQuality.STALE]) == DataQuality.STALE

    def test_merge_priority_delayed(self):
        """DELAYED 高于 GOOD/UNKNOWN。"""
        assert DataQuality.merge([DataQuality.GOOD, DataQuality.DELAYED]) == DataQuality.DELAYED
        assert DataQuality.merge([DataQuality.UNKNOWN, DataQuality.DELAYED]) == DataQuality.DELAYED

    def test_merge_priority_good_beats_unknown(self):
        """GOOD 高于 UNKNOWN。"""
        assert DataQuality.merge([DataQuality.UNKNOWN, DataQuality.GOOD]) == DataQuality.GOOD

    def test_merge_follows_first_occurrence_in_priority_list(self):
        """
        关键行为：merge 按**优先级列表**顺序找，不是按输入列表顺序。
        多个同优先级候选 → 取决于输入列表中哪个先被命中。
        """
        # 没有 INVALID/MISSING/PARTIAL/SUSPICIOUS/STALE/DELAYED 时
        # 仅 GOOD 与 UNKNOWN → GOOD
        assert DataQuality.merge([DataQuality.UNKNOWN, DataQuality.GOOD]) == DataQuality.GOOD
        # 单元素测试
        assert DataQuality.merge([DataQuality.UNKNOWN]) == DataQuality.UNKNOWN

    def test_merge_full_set(self):
        """所有成员一起 merge → INVALID（最高优先级）。"""
        all_qs = [
            DataQuality.UNKNOWN, DataQuality.GOOD, DataQuality.DELAYED,
            DataQuality.STALE, DataQuality.PARTIAL, DataQuality.SUSPICIOUS,
            DataQuality.MISSING,
        ]
        result = DataQuality.merge(all_qs)
        # 没有 INVALID
        assert result == DataQuality.MISSING
        # 加入 INVALID
        result = DataQuality.merge(all_qs + [DataQuality.INVALID])
        assert result == DataQuality.INVALID


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
