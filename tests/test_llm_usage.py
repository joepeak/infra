#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm.usage 测试

覆盖：
1. record_usage 累加 token
2. record_usage 累加 calls
3. set_llm_purpose + ContextVar 切换用途
4. record_usage 使用 explicit purpose 参数覆盖 ContextVar
5. get_usage_summary 返回深拷贝（修改 snapshot 不影响内部状态）
6. reset_usage_summary 清空
7. usage 缺字段时按 0 累加
8. log_usage_summary 不抛
9. record_usage 线程安全（多线程并发）

注意：_usage_totals 是模块级 defaultdict——每个测试必须 reset 避免污染。
"""

from __future__ import annotations

import threading

import pytest

import infra.llm.usage as usage_module
from infra.llm.usage import (
    set_llm_purpose,
    reset_llm_purpose,
    record_usage,
    get_usage_summary,
    reset_usage_summary,
    log_usage_summary,
)


# ============================================================
# Fixture
# ============================================================
@pytest.fixture(autouse=True)
def _reset_usage():
    """每个测试前清空 _usage_totals。"""
    reset_usage_summary()
    yield
    reset_usage_summary()


# ============================================================
# 1-2. record_usage 累加
# ============================================================
class TestRecordUsage:
    def test_basic_accumulation(self):
        record_usage({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        summary = get_usage_summary()
        # 默认用途 "unknown"
        assert "unknown" in summary
        assert summary["unknown"]["prompt_tokens"] == 10
        assert summary["unknown"]["completion_tokens"] == 20
        assert summary["unknown"]["total_tokens"] == 30
        assert summary["unknown"]["calls"] == 1

    def test_multiple_calls_accumulate(self):
        record_usage({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        record_usage({"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20})
        summary = get_usage_summary()
        assert summary["unknown"]["prompt_tokens"] == 15
        assert summary["unknown"]["completion_tokens"] == 35
        assert summary["unknown"]["total_tokens"] == 50
        assert summary["unknown"]["calls"] == 2

    def test_calls_counter_increments(self):
        for _ in range(5):
            record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        assert get_usage_summary()["unknown"]["calls"] == 5


# ============================================================
# 3. ContextVar 切换用途
# ============================================================
class TestPurposeContextVar:
    def test_set_purpose_changes_recorded_bucket(self):
        token = set_llm_purpose("writing")
        try:
            record_usage({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        finally:
            reset_llm_purpose(token)

        summary = get_usage_summary()
        assert "writing" in summary
        assert summary["writing"]["total_tokens"] == 30
        # 不会写入 "unknown"
        assert "unknown" not in summary

    def test_set_then_reset_restores_previous(self):
        # 默认是 "unknown"
        token1 = set_llm_purpose("writing")
        record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        reset_llm_purpose(token1)
        # 复位后应是 "unknown"
        record_usage({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        summary = get_usage_summary()
        assert summary["writing"]["total_tokens"] == 2
        assert summary["unknown"]["total_tokens"] == 7

    def test_nested_purposes(self):
        outer = set_llm_purpose("outer")
        try:
            inner = set_llm_purpose("inner")
            try:
                record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
            finally:
                reset_llm_purpose(inner)
            record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3})
        finally:
            reset_llm_purpose(outer)

        summary = get_usage_summary()
        assert summary["inner"]["total_tokens"] == 2
        assert summary["outer"]["total_tokens"] == 3

    def test_reset_purpose_with_none_noop(self):
        """reset_llm_purpose(None) 不抛。"""
        # 应直接 return（不调用 _current_purpose.reset）
        reset_llm_purpose(None)  # 不报错


# ============================================================
# 4. explicit purpose 参数覆盖 ContextVar
# ============================================================
class TestExplicitPurpose:
    def test_purpose_param_overrides_contextvar(self):
        token = set_llm_purpose("context_purpose")
        try:
            # explicit 覆盖
            record_usage(
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                purpose="explicit_purpose",
            )
        finally:
            reset_llm_purpose(token)

        summary = get_usage_summary()
        assert summary["explicit_purpose"]["total_tokens"] == 2
        # 上下文标签没收到
        assert "context_purpose" not in summary

    def test_purpose_none_uses_contextvar(self):
        token = set_llm_purpose("ctx")
        try:
            record_usage(
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                purpose=None,  # 用 ctx
            )
        finally:
            reset_llm_purpose(token)

        summary = get_usage_summary()
        assert summary["ctx"]["total_tokens"] == 2


# ============================================================
# 5. get_usage_summary 返回深拷贝
# ============================================================
class TestSummarySnapshot:
    def test_snapshot_is_copy(self):
        record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        snap = get_usage_summary()
        snap["unknown"]["prompt_tokens"] = 999
        # 内部状态未受影响
        assert get_usage_summary()["unknown"]["prompt_tokens"] == 1

    def test_snapshot_returns_empty_when_no_records(self):
        assert get_usage_summary() == {}


# ============================================================
# 6. reset_usage_summary
# ============================================================
class TestResetSummary:
    def test_reset_clears_all(self):
        record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        record_usage(
            {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            purpose="other",
        )
        assert len(get_usage_summary()) == 2

        reset_usage_summary()
        assert get_usage_summary() == {}


# ============================================================
# 7. 缺字段时按 0
# ============================================================
class TestMissingFields:
    def test_missing_keys_default_to_zero(self):
        record_usage({})  # 完全没有
        summary = get_usage_summary()
        assert summary["unknown"]["prompt_tokens"] == 0
        assert summary["unknown"]["completion_tokens"] == 0
        assert summary["unknown"]["total_tokens"] == 0
        assert summary["unknown"]["calls"] == 1  # 仍然计数

    def test_partial_keys(self):
        record_usage({"prompt_tokens": 5})  # 只给 prompt
        summary = get_usage_summary()
        assert summary["unknown"]["prompt_tokens"] == 5
        assert summary["unknown"]["completion_tokens"] == 0
        assert summary["unknown"]["total_tokens"] == 0
        assert summary["unknown"]["calls"] == 1


# ============================================================
# 8. log_usage_summary 不抛
# ============================================================
class TestLogUsageSummary:
    def test_empty_summary_no_log(self, caplog):
        """空 summary 时不写日志（直接 return）。"""
        with caplog.at_level("INFO", logger="infra.llm.usage"):
            log_usage_summary()
        # caplog 应为空（或只有初始 INFO）
        usage_records = [r for r in caplog.records if "Token 用量" in r.message]
        assert usage_records == []

    def test_non_empty_summary_logs(self, caplog):
        record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        with caplog.at_level("INFO", logger="infra.llm.usage"):
            log_usage_summary()
        usage_records = [r for r in caplog.records if "Token 用量" in r.message]
        assert len(usage_records) == 1
        # log 内容包含 "unknown" 用途 + 数字
        assert "unknown" in usage_records[0].message
        assert "calls=1" in usage_records[0].message

    def test_log_does_not_raise(self):
        record_usage({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        # 不应抛异常
        log_usage_summary()


# ============================================================
# 9. 线程安全
# ============================================================
class TestThreadSafety:
    def test_concurrent_record_usage(self):
        """多线程并发 record_usage：最终累加正确。"""
        n_threads = 10
        calls_per_thread = 50

        def worker():
            for _ in range(calls_per_thread):
                record_usage({
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                })

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = get_usage_summary()
        expected_calls = n_threads * calls_per_thread
        assert summary["unknown"]["calls"] == expected_calls
        assert summary["unknown"]["prompt_tokens"] == expected_calls
        assert summary["unknown"]["completion_tokens"] == expected_calls * 2
        assert summary["unknown"]["total_tokens"] == expected_calls * 3


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
