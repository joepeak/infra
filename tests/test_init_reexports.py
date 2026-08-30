#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
各 __init__.py 的 re-export 完整性测试

覆盖：
1. infra.__init__ 的 __all__ 全部符号可导入
2. infra.db.__init__ 的 __all__ 全部符号可导入
3. infra.redis.__init__ 的 __all__ 全部符号可导入
4. infra.llm.__init__ 的 __all__ 全部符号可导入
5. infra.utils.__init__ 的 __all__ 全部符号可导入
6. 几个关键符号类型校验

目的：保护向后兼容——重命名/删除被 re-export 的符号会立即被这些测试捕获。
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import DeclarativeBase


def _check_all_defined(module_name: str):
    """import module.__all__ 每个名字都能 getattr。"""
    import importlib
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "__all__"), f"{module_name} 没有 __all__"
    for name in mod.__all__:
        assert hasattr(mod, name), f"{module_name}.{name} 不存在（__all__ 声明了但模块没）"
    return mod


# ============================================================
# 1. infra 顶层
# ============================================================
class TestInfraRoot:
    def test_all_exports_resolvable(self):
        mod = _check_all_defined("infra")
        # 关键符号类型断言
        from infra.data_quality import DataQuality
        from infra.db.database_model import Base
        from infra.utils.time_util import to_utc_event_time
        assert mod.DataQuality is DataQuality
        assert mod.to_utc_event_time is to_utc_event_time
        assert mod.Base is Base

    def test_db_submodule_aliases(self):
        """__init__ 中重新导出的 db 符号都存在。"""
        mod = _check_all_defined("infra")
        # 不重复 import，直接通过 mod 访问
        assert callable(mod.init_db_manager)
        assert callable(mod.get_db_manager)
        assert callable(mod.get_db_session)

    def test_bootstrap_aliases(self):
        mod = _check_all_defined("infra")
        assert callable(mod.bootstrap_all)
        assert callable(mod.shutdown_bootstrap)
        assert callable(mod.load_config)

    def test_redis_aliases(self):
        mod = _check_all_defined("infra")
        assert callable(mod.init_redis)
        assert callable(mod.close_redis)
        # get_redis 可能未初始化时返 None——只是 callable 测试


# ============================================================
# 2. infra.db
# ============================================================
class TestInfraDb:
    def test_all_exports_resolvable(self):
        mod = _check_all_defined("infra.db")
        # 关键符号类型
        from infra.db.database_model import Base, BaseModel, TimeSeriesBaseModel
        from infra.db.database_repository import DatabaseRepository
        from infra.db.operations import db_operation
        assert mod.Base is Base
        assert mod.BaseModel is BaseModel
        assert mod.TimeSeriesBaseModel is TimeSeriesBaseModel
        assert mod.DatabaseRepository is DatabaseRepository
        assert mod.db_operation is db_operation

    def test_db_manager_aliases(self):
        mod = _check_all_defined("infra.db")
        assert callable(mod.init_db_manager)
        assert callable(mod.get_db_manager)
        assert callable(mod.get_db_session)
        assert callable(mod.get_db_health)
        assert callable(mod.close_all_db_connections)


# ============================================================
# 3. infra.redis
# ============================================================
class TestInfraRedis:
    def test_all_exports_resolvable(self):
        mod = _check_all_defined("infra.redis")
        from infra.redis.client import RedisClient, init_redis, get_redis, close_redis
        from infra.redis.decorators import cache
        assert mod.RedisClient is RedisClient
        assert mod.init_redis is init_redis
        assert mod.get_redis is get_redis
        assert mod.close_redis is close_redis
        assert mod.cache is cache


# ============================================================
# 4. infra.llm
# ============================================================
class TestInfraLlm:
    def test_all_exports_resolvable(self):
        mod = _check_all_defined("infra.llm")
        from infra.llm.client import (
            LLMClient,
            OpenAIClient,
            LLMFactory,
            init_llm_client,
            get_llm_client,
        )
        from infra.llm.models import LLMRequest, LLMResponse
        from infra.llm.retry import LLMRetryConfig
        from infra.llm.usage import (
            set_llm_purpose, reset_llm_purpose, record_usage,
            get_usage_summary, reset_usage_summary, log_usage_summary,
        )
        # _reset_for_test 直接在 infra.llm 包中定义
        from infra.llm import _reset_for_test
        # 类型
        assert mod.LLMClient is LLMClient
        assert mod.LLMRequest is LLMRequest
        assert mod.LLMResponse is LLMResponse
        assert mod.LLMRetryConfig is LLMRetryConfig
        # 函数
        assert callable(mod.init_llm_client)
        assert callable(mod.get_llm_client)
        assert callable(mod.set_llm_purpose)
        assert callable(mod.reset_llm_purpose)
        assert callable(mod.record_usage)
        assert mod._reset_for_test is _reset_for_test


# ============================================================
# 5. infra.utils
# ============================================================
class TestInfraUtils:
    def test_all_exports_resolvable(self):
        mod = _check_all_defined("infra.utils")
        from infra.utils.time_util import to_utc_event_time
        from infra.utils.error_handler import (
            handle_exceptions, async_handle_exceptions, ExceptionContext,
        )
        assert mod.to_utc_event_time is to_utc_event_time
        assert mod.handle_exceptions is handle_exceptions
        assert mod.async_handle_exceptions is async_handle_exceptions
        assert mod.ExceptionContext is ExceptionContext


# ============================================================
# 6. infra.exceptions 内部 __all__ 也有
# ============================================================
class TestInfraExceptions:
    def test_all_classes_importable(self):
        """infra.exceptions 导出 11 个异常类，类型校验。"""
        from infra.exceptions import (
            AppException,
            ConfigurationError,
            DatabaseError,
            RedisError,
            MessageQueueError,
            SchedulerError,
            DataProducerError,
            DataConsumerError,
            APIError,
            BusinessException,
            SystemException,
            AuthException,
        )
        # 全部继承 AppException（除了基类本身）
        for cls in [
            ConfigurationError, DatabaseError, RedisError, MessageQueueError,
            SchedulerError, DataProducerError, DataConsumerError, APIError,
            BusinessException, SystemException, AuthException,
        ]:
            assert issubclass(cls, AppException)


# ============================================================
# 7. infra.data_quality
# ============================================================
class TestInfraDataQuality:
    def test_dataquality_is_strenum(self):
        """DataQuality 应是 StrEnum（修复后），且 8 个成员。"""
        from infra.data_quality import DataQuality
        from enum import StrEnum
        # 3.11+ StrEnum 是 str + Enum 的子类
        assert issubclass(DataQuality, str)
        assert issubclass(DataQuality, StrEnum)
        assert len(DataQuality) == 8


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
