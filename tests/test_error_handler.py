#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.utils.error_handler 测试

覆盖：
1. handle_exceptions 装饰器：default_return / reraise / exception_types / 日志写入
2. async_handle_exceptions 装饰器同上但异步
3. ExceptionContext 上下文管理器：__enter__ / __exit__ 各种分支
4. AppException 的 error_code / details 出现在日志 extra 中
"""

from __future__ import annotations

import logging

import pytest

from infra.utils.error_handler import (
    handle_exceptions,
    async_handle_exceptions,
    ExceptionContext,
)
from infra.exceptions import AppException


# ============================================================
# 1. handle_exceptions 装饰器
# ============================================================
class TestHandleExceptionsDecorator:
    def test_no_exception_returns_value(self):
        @handle_exceptions()
        def fn(x):
            return x * 2

        assert fn(5) == 10

    def test_exception_returns_default(self):
        @handle_exceptions(default_return="fallback")
        def fn():
            raise ValueError("boom")

        assert fn() == "fallback"

    def test_exception_returns_default_none(self):
        @handle_exceptions()
        def fn():
            raise ValueError("boom")

        # 默认 default_return=None
        assert fn() is None

    def test_reraise_true(self):
        @handle_exceptions(reraise=True)
        def fn():
            raise ValueError("must raise")

        with pytest.raises(ValueError, match="must raise"):
            fn()

    def test_exception_types_filter(self):
        """不在 exception_types 中的异常：reraise=False → 返回 default_return。"""
        @handle_exceptions(
            exception_types=(KeyError,),
            default_return="not_in_filter_fallback",
            reraise=False,
        )
        def fn():
            raise ValueError("not in filter")

        # ValueError 不在 KeyError 列表中 → _should_capture=False
        # 走 reraise=False 分支 → 返回 default_return
        assert fn() == "not_in_filter_fallback"

    def test_exception_types_filter_reraise(self):
        """不在 exception_types 中 + reraise=True → 仍 raise 原异常。"""
        @handle_exceptions(
            exception_types=(KeyError,),
            reraise=True,
        )
        def fn():
            raise ValueError("not in filter, must raise")

        with pytest.raises(ValueError, match="not in filter"):
            fn()

    def test_exception_types_match(self):
        @handle_exceptions(
            exception_types=(ValueError,),
            default_return="caught",
        )
        def fn():
            raise ValueError("matched")

        assert fn() == "caught"

    def test_subclass_matches_filter(self):
        class MyValueError(ValueError):
            pass

        @handle_exceptions(
            exception_types=(ValueError,),
            default_return="caught",
        )
        def fn():
            raise MyValueError("subclass")

        assert fn() == "caught"

    def test_log_level(self, caplog):
        @handle_exceptions(default_return="x", log_level="WARNING")
        def fn():
            raise ValueError("logged")

        with caplog.at_level(logging.WARNING, logger="infra.utils.error_handler"):
            # 直接捕获
            caplog.clear()
            result = fn()
        assert result == "x"
        # 至少一条 warning
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1

    def test_invalid_log_level_falls_back_to_error(self, caplog):
        @handle_exceptions(default_return="x", log_level="NONEXISTENT")
        def fn():
            raise ValueError("x")

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="infra.utils.error_handler"):
            fn()
        # fallback 到 error
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) >= 1

    def test_appexception_error_code_in_extra(self, caplog):
        @handle_exceptions()
        def fn():
            raise AppException("oops", error_code="MY_CODE", details={"k": 1})

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="infra.utils.error_handler"):
            result = fn()
        assert result is None
        # 找到包含 error_code 的记录
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any(getattr(r, "error_code", None) == "MY_CODE" for r in error_records)
        assert any(getattr(r, "details", None) == {"k": 1} for r in error_records)


# ============================================================
# 2. async_handle_exceptions 装饰器
# ============================================================
class TestAsyncHandleExceptionsDecorator:
    async def test_no_exception(self):
        @async_handle_exceptions(default_return="x")
        async def fn():
            return "ok"

        assert await fn() == "ok"

    async def test_exception_returns_default(self):
        @async_handle_exceptions(default_return="fallback")
        async def fn():
            raise ValueError("boom")

        assert await fn() == "fallback"

    async def test_reraise(self):
        @async_handle_exceptions(reraise=True)
        async def fn():
            raise ValueError("must raise")

        with pytest.raises(ValueError, match="must raise"):
            await fn()

    async def test_exception_types(self):
        @async_handle_exceptions(exception_types=(ValueError,), default_return="caught")
        async def fn():
            raise ValueError("matched")

        assert await fn() == "caught"


# ============================================================
# 3. ExceptionContext 上下文管理器
# ============================================================
class TestExceptionContext:
    def test_no_exception_enter_exit(self):
        with ExceptionContext("test-op") as ctx:
            assert ctx is not None
            x = 1 + 1
        # 正常完成，无返回值需要断言

    def test_exception_suppressed_returns_none(self):
        # reraise=False → 抑制
        with ExceptionContext("test-op", reraise=False) as ctx:
            raise ValueError("suppressed")
        # 正常离开 with，说明异常被吞掉

    def test_exception_reraises(self):
        with pytest.raises(ValueError, match="must raise"):
            with ExceptionContext("test-op", reraise=True):
                raise ValueError("must raise")

    def test_exception_types_filter_unmatched_reraise_false(self):
        """不在 exception_types 中且 reraise=False：仍然抑制（已知行为）。"""
        with ExceptionContext(
            "test-op",
            exception_types=(KeyError,),
            reraise=False,
        ):
            raise ValueError("not in filter")
        # 正常离开 → 被抑制

    def test_exception_types_match_returns_implicit_none(self):
        """匹配时 _log + 按 reraise 决定，但 default_return 在 ctx 中未使用。"""
        with ExceptionContext(
            "test-op",
            exception_types=(ValueError,),
            reraise=False,
        ):
            raise ValueError("matched")
        # 正常离开，无返回（default_return 字段对 ctx 路径无效）

    def test_default_return_not_used_in_context(self):
        """ctx.__exit__ 不消费 default_return 属性——这是已知行为。

        ExceptionContext 是 with 语句的 ctx mgr，__exit__ 不会因为异常就
        "返回" default_return 给外部。default_return 字段对 ctx 路径完全无意义，
        仅装饰器路径会用到。
        """
        ctx = ExceptionContext(
            "test-op",
            default_return="never_seen",
            reraise=False,
        )
        # default_return 存在，但 __exit__ 不使用
        assert ctx.default_return == "never_seen"
        # __exit__ 在没有异常时直接返回 True
        assert ctx.__exit__(None, None, None) is True
        # 有异常 + reraise=False + 匹配：仍返回 True（抑制），不读 default_return
        exc = ValueError("x")
        result = ctx.__exit__(type(exc), exc, exc.__traceback__)
        assert result is True  # 抑制 = True（吞掉异常）


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
