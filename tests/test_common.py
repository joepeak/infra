#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.utils.common 工具函数测试

覆盖：
1. parse_json_string：合法 JSON / 非法 JSON / 非字符串
2. merge_results：空参 / 多个 dict / 第一个非 None 胜出 / 异常传播
3. parse_datetime_with_format：默认格式 / 自定义格式 / 非法格式
"""

from __future__ import annotations

from datetime import datetime

import pytest

from infra.utils.common import (
    parse_json_string,
    merge_results,
    parse_datetime_with_format,
)


# ============================================================
# 1. parse_json_string
# ============================================================
class TestParseJsonString:
    def test_valid_dict(self):
        assert parse_json_string('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}

    def test_valid_list(self):
        assert parse_json_string("[1, 2, 3]") == [1, 2, 3]

    def test_valid_primitive(self):
        assert parse_json_string("42") == 42
        assert parse_json_string("true") is True
        assert parse_json_string("null") is None
        assert parse_json_string('"hello"') == "hello"

    def test_invalid_json_returns_original(self):
        """解析失败时回退返回原字符串。"""
        s = "{not valid json"
        assert parse_json_string(s) == s

    def test_empty_string_returns_original(self):
        # 空字符串不是合法 JSON
        assert parse_json_string("") == ""

    def test_non_string_input_raises_type_error(self):
        """非 str/bytes 输入会抛 TypeError（已知行为，未捕获）。"""
        with pytest.raises(TypeError):
            parse_json_string({"already": "a dict"})

    def test_bytes_input_supported(self):
        # JSONDecodeError 也会捕获 bytes 失败的情况
        assert parse_json_string(b'{"x": 1}') == {"x": 1}
        assert parse_json_string(b"garbage") == b"garbage"


# ============================================================
# 2. merge_results
# ============================================================
class TestMergeResults:
    def test_no_args_returns_none(self):
        assert merge_results() is None

    def test_single_dict(self):
        assert merge_results(lambda: {"a": 1}) == {"a": 1}

    def test_multiple_dicts_merged(self):
        def a():
            return {"a": 1}

        def b():
            return {"b": 2}

        # 后面的覆盖前面的同名 key
        result = merge_results(a, b)
        assert result == {"a": 1, "b": 2}

    def test_none_results_are_skipped(self):
        def a():
            return None

        def b():
            return {"b": 2}

        def c():
            return None

        assert merge_results(a, b, c) == {"b": 2}

    def test_all_none_returns_none(self):
        assert merge_results(lambda: None, lambda: None) is None

    def test_exceptions_propagate(self):
        """merge_results 不捕获异常，调用方的异常会向上抛。"""

        def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            merge_results(boom)

    def test_callable_invoked_without_args(self):
        """callable 必须无参（merge_results 不传参）。"""
        counter = {"calls": 0}

        def fn():
            counter["calls"] += 1
            return {"k": counter["calls"]}

        merge_results(fn, fn)
        assert counter["calls"] == 2

    def test_evaluated_eagerly_left_to_right(self):
        """短路：左 None 不会让右侧被跳过——callable 总被调用。"""
        results = []

        def a():
            results.append("a")
            return None

        def b():
            results.append("b")
            return {"b": 1}

        merge_results(a, b)
        assert results == ["a", "b"]


# ============================================================
# 3. parse_datetime_with_format
# ============================================================
class TestParseDatetimeWithFormat:
    def test_default_format(self):
        dt = parse_datetime_with_format("2026.08.29 12:34:56")
        assert isinstance(dt, datetime)
        assert dt.year == 2026
        assert dt.month == 8
        assert dt.day == 29
        assert dt.hour == 12
        assert dt.minute == 34
        assert dt.second == 56

    def test_custom_format_iso(self):
        dt = parse_datetime_with_format("2026-08-29T10:00:00", fmt="%Y-%m-%dT%H:%M:%S")
        assert dt == datetime(2026, 8, 29, 10, 0, 0)

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_datetime_with_format("not a date")

    def test_format_mismatch_raises_value_error(self):
        # 用了错误分隔符
        with pytest.raises(ValueError):
            parse_datetime_with_format("2026-08-29 12:34:56", fmt="%Y.%m.%d %H:%M:%S")


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
