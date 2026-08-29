#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm.retry.LLMRetryConfig 测试

覆盖：
1. 包装 RetryConfig
2. should_retry 行为：deny 优先 / allow 命中 / 都不匹配 → False
3. default / aggressive / conservative 三个工厂方法的参数
4. 自定义 RetryConfig 构造 LLMRetryConfig
5. openai 包存在时，deny/allow 包含 openai 异常类
6. openai 包不存在时，deny/allow 是空元组（仅通用异常）

注意：_build_default_deny_exceptions / _build_default_allow_exceptions 在
import 时执行；如果 openai 已装，会导入 openai 的异常类。
"""

from __future__ import annotations

import pytest

from infra.utils.retry import RetryConfig
from infra.llm.retry import (
    LLMRetryConfig,
    _build_default_deny_exceptions,
    _build_default_allow_exceptions,
)


# ============================================================
# 1. 基本构造
# ============================================================
class TestConstruction:
    def test_wraps_retry_config(self):
        rc = RetryConfig(max_retries=5, base_delay=0.1)
        cfg = LLMRetryConfig(rc)
        assert cfg.retry_config is rc
        assert cfg.retry_config.max_retries == 5


# ============================================================
# 2. should_retry
# ============================================================
class TestShouldRetry:
    def test_deny_exception_returns_false(self):
        """deny 列表里的异常一律不重试。"""
        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(ValueError,),
            allow_exceptions=(ValueError, ConnectionError),
        )
        cfg = LLMRetryConfig(rc)
        # ValueError 同时在 deny 和 allow → deny 优先
        assert cfg.should_retry(ValueError("x")) is False

    def test_allow_exception_returns_true(self):
        """allow 列表命中且不在 deny → 重试。"""
        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(),
            allow_exceptions=(ConnectionError,),
        )
        cfg = LLMRetryConfig(rc)
        assert cfg.should_retry(ConnectionError("x")) is True

    def test_unmatched_exception_returns_false(self):
        """既不在 deny 也不在 allow → 不重试。"""
        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(ValueError,),
            allow_exceptions=(ConnectionError,),
        )
        cfg = LLMRetryConfig(rc)
        # KeyError 不匹配
        assert cfg.should_retry(KeyError("x")) is False

    def test_subclass_of_allow_passes(self):
        """isinstance 匹配子类。"""

        class MyConnError(ConnectionError):
            pass

        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(),
            allow_exceptions=(ConnectionError,),
        )
        cfg = LLMRetryConfig(rc)
        assert cfg.should_retry(MyConnError("x")) is True

    def test_subclass_of_deny_blocked(self):
        """isinstance 匹配 deny 子类也返回 False。"""

        class MyValueError(ValueError):
            pass

        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(ValueError,),
            allow_exceptions=(),
        )
        cfg = LLMRetryConfig(rc)
        assert cfg.should_retry(MyValueError("x")) is False

    def test_empty_deny_and_allow(self):
        """都为空 → 所有异常都不重试（与 RetryExecutor 行为一致）。"""
        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(),
            allow_exceptions=(),
        )
        cfg = LLMRetryConfig(rc)
        assert cfg.should_retry(ConnectionError("x")) is False
        assert cfg.should_retry(ValueError("x")) is False
        assert cfg.should_retry(Exception("x")) is False


# ============================================================
# 3. 工厂方法
# ============================================================
class TestFactoryMethods:
    def test_default_uses_retry_config_default(self):
        """default() 复用 RetryConfig.default() 的参数。"""
        cfg = LLMRetryConfig.default()
        rc = cfg.retry_config
        assert rc.max_retries == 3
        assert rc.base_delay == 1.0
        assert rc.max_delay == 30.0
        assert rc.exponential_base == 2.0

    def test_aggressive_uses_retry_config_aggressive(self):
        cfg = LLMRetryConfig.aggressive()
        rc = cfg.retry_config
        assert rc.max_retries == 10
        assert rc.base_delay == 0.5
        assert rc.max_delay == 10.0
        assert rc.exponential_base == 1.5

    def test_conservative_uses_retry_config_conservative(self):
        cfg = LLMRetryConfig.conservative()
        rc = cfg.retry_config
        assert rc.max_retries == 2
        assert rc.base_delay == 5.0
        assert rc.max_delay == 60.0

    def test_all_factories_have_deny_exceptions(self):
        """三个 factory 都有非空 deny_exceptions。"""
        for cfg in [LLMRetryConfig.default(), LLMRetryConfig.aggressive(), LLMRetryConfig.conservative()]:
            assert len(cfg.retry_config.deny_exceptions) >= 3  # 至少 3 个 openai 异常

    def test_all_factories_have_allow_exceptions(self):
        for cfg in [LLMRetryConfig.default(), LLMRetryConfig.aggressive(), LLMRetryConfig.conservative()]:
            assert len(cfg.retry_config.allow_exceptions) >= 3

    def test_deny_includes_authentication_error_when_openai_present(self):
        """如果 openai 装了，deny 包含 AuthenticationError。"""
        openai = pytest.importorskip("openai")
        deny = _build_default_deny_exceptions()
        # 应包含 openai.AuthenticationError
        from openai import AuthenticationError, BadRequestError, PermissionDeniedError
        assert AuthenticationError in deny
        assert BadRequestError in deny
        assert PermissionDeniedError in deny

    def test_allow_includes_rate_limit_when_openai_present(self):
        openai = pytest.importorskip("openai")
        allow = _build_default_allow_exceptions()
        from openai import APITimeoutError, RateLimitError, APIError
        assert APITimeoutError in allow
        assert RateLimitError in allow
        assert APIError in allow

    def test_allow_always_includes_builtins(self):
        """无论 openai 装不装，allow 都至少包含 ConnectionError/TimeoutError/OSError。"""
        allow = _build_default_allow_exceptions()
        assert ConnectionError in allow
        assert TimeoutError in allow
        assert OSError in allow


# ============================================================
# 4. 自定义 RetryConfig 构造
# ============================================================
class TestCustomConfig:
    def test_custom_retry_params(self):
        rc = RetryConfig(
            max_retries=7,
            base_delay=0.2,
            max_delay=15.0,
            exponential_base=3.0,
            deny_exceptions=(ValueError,),
            allow_exceptions=(KeyError,),
        )
        cfg = LLMRetryConfig(rc)
        assert cfg.retry_config.max_retries == 7
        assert cfg.retry_config.exponential_base == 3.0
        assert cfg.should_retry(ValueError("x")) is False  # deny
        assert cfg.should_retry(KeyError("x")) is True  # allow

    def test_aggressive_with_different_deny(self):
        """aggressive 工厂的 deny 可被 override。"""
        rc = RetryConfig.aggressive()
        rc_modified = rc.copy(deny_exceptions=(KeyError,))
        cfg = LLMRetryConfig(rc_modified)
        assert cfg.should_retry(KeyError("x")) is False
        # 原来 RetryConfig.aggressive().allow_exceptions 仍生效
        assert cfg.retry_config.max_retries == 10


# ============================================================
# 5. should_retry 与 RetryExecutor 行为一致
# ============================================================
class TestConsistencyWithRetryExecutor:
    def test_matches_retry_executor_should_retry(self):
        """LLMRetryConfig.should_retry 与 RetryExecutor.should_retry 一致。"""
        from infra.utils.retry import RetryExecutor
        rc = RetryConfig(
            max_retries=3,
            base_delay=0.001,
            deny_exceptions=(ValueError,),
            allow_exceptions=(ConnectionError,),
        )
        llm_cfg = LLMRetryConfig(rc)
        exe = RetryExecutor(rc)

        for exc in [ValueError("x"), ConnectionError("x"), KeyError("x"), Exception("x")]:
            assert llm_cfg.should_retry(exc) == exe.should_retry(exc)


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
