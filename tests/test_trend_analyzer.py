#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_trend 测试

覆盖：
1. 边界：空 list、长度 1
2. 上升趋势（trend=up, strength 分支）
3. 下降趋势（trend=down）
4. 平稳趋势（trend=sideways, slope≈0）
5. y 全相等（np.polyfit 可能 RankWarning，应能处理）
6. data 项缺 'value' → KeyError
7. window 大于数据长度：使用全部数据
8. 斜率正负反转
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from infra.utils.trend_analyzer import analyze_trend


def _series(values, window=10):
    """构造 [ {'value': v} for v in values ]。"""
    return [{"value": float(v)} for v in values]


# ============================================================
# 1. 边界
# ============================================================
class TestBoundaries:
    def test_empty_list(self):
        result = analyze_trend([])
        assert result["trend"] == "sideways"
        assert result["slope"] == 0
        assert result["strength"] == "weak"
        assert result["window"] == 0

    def test_single_point(self):
        result = analyze_trend(_series([42]))
        assert result["trend"] == "sideways"
        assert result["slope"] == 0
        assert result["strength"] == "weak"
        assert result["window"] == 0  # len < 2 → 0

    def test_missing_value_key_raises_key_error(self):
        with pytest.raises(KeyError):
            analyze_trend([{"no_value": 1}, {"no_value": 2}])


# ============================================================
# 2. 上升趋势
# ============================================================
@pytest.mark.slow
class TestUpTrend:
    def test_clear_uptrend(self):
        # 100 → 200，斜率 ≈ +10（11 个点，window=10 会取后 10 个）
        values = [100 + 10 * i for i in range(11)]
        result = analyze_trend(_series(values), window=10, threshold=0.01)
        assert result["trend"] == "up"
        assert result["slope"] > 0
        # 强上升：slope 10 > threshold*3 = 0.03
        assert result["strength"] == "strong"

    def test_moderate_uptrend(self):
        # 斜率 0.2，threshold=0.1 → slope > threshold 但 < threshold*3=0.3 → moderate
        values = [0.2 * i for i in range(20)]
        result = analyze_trend(_series(values), window=10, threshold=0.1)
        assert result["trend"] == "up"
        assert result["slope"] > 0.1
        assert result["slope"] < 0.3
        assert result["strength"] == "moderate"


# ============================================================
# 3. 下降趋势
# ============================================================
@pytest.mark.slow
class TestDownTrend:
    def test_clear_downtrend(self):
        values = [200 - 10 * i for i in range(11)]
        result = analyze_trend(_series(values), window=10, threshold=0.01)
        assert result["trend"] == "down"
        assert result["slope"] < 0
        assert result["strength"] == "strong"


# ============================================================
# 4. 平稳趋势
# ============================================================
@pytest.mark.slow
class TestSidewaysTrend:
    def test_flat_series_is_sideways(self):
        # 100 个相同值
        result = analyze_trend(_series([100] * 20), window=10, threshold=0.01)
        assert result["trend"] == "sideways"
        # 斜率为 0 或非常接近
        assert abs(result["slope"]) < 1e-6
        assert result["strength"] == "weak"

    def test_small_oscillation(self):
        # 100, 101, 100, 101, ... polyfit 斜率 ≈ +0.03
        # threshold=0.1 → sideways；threshold=0.01 → up
        values = [100 + (i % 2) for i in range(20)]
        # 较大的 threshold 把微斜率归为 sideways
        result = analyze_trend(_series(values), window=10, threshold=0.1)
        assert result["trend"] == "sideways"
        # 较小的 threshold 会把它判为 up（polyfit 拟合出 +0.03）
        result_strict = analyze_trend(_series(values), window=10, threshold=0.01)
        assert result_strict["trend"] == "up"


# ============================================================
# 5. window 行为
# ============================================================
@pytest.mark.slow
class TestWindow:
    def test_window_larger_than_data_uses_all(self):
        # 5 个点，window=20
        values = [1, 2, 3, 4, 5]
        result = analyze_trend(_series(values), window=20, threshold=0.01)
        assert result["trend"] == "up"
        # 实际用了 5 个点
        assert result["window"] == 5  # 实际窗口是 len(data) when < window

    def test_window_smaller_takes_recent(self):
        # 20 个点，window=5
        values = list(range(20))  # 0..19
        result = analyze_trend(_series(values), window=5, threshold=0.01)
        # 实际用的是 15..19 五个点（最后 5 个）
        assert result["window"] == 5
        assert result["trend"] == "up"
        # 斜率应接近 1（连续整数的 polyfit 斜率）
        assert abs(result["slope"] - 1.0) < 0.01


# ============================================================
# 6. 返回值类型
# ============================================================
class TestReturnType:
    def test_slope_is_float(self):
        result = analyze_trend(_series([1, 2, 3, 4, 5]))
        assert isinstance(result["slope"], float)

    def test_keys_present(self):
        result = analyze_trend(_series([1, 2, 3, 4, 5]))
        assert set(result.keys()) >= {"trend", "slope", "strength", "window"}


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
