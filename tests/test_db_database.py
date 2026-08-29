#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db.database 向后兼容 shim 测试

覆盖：
1. re-exports 完整性（__all__ 中每个符号都可导入）
2. get_db_session_dependency / get_timeseries_session_dependency 是 async generator
3. get_db_session_legacy 是 async 函数
4. get_db_health 是 async 函数
5. 函数确实转发到 connection_manager 的对应 API（mock 验证）

这个模块是个 shim——没有真实逻辑，只测转发对不对。
"""

from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


# ============================================================
# 1. re-exports
# ============================================================
class TestReExports:
    def test_all_symbols_importable(self):
        """__all__ 中每个符号都应能正常导入。"""
        from infra.db import database
        for name in database.__all__:
            assert hasattr(database, name), f"missing re-export: {name}"

    def test_specific_exports(self):
        from infra.db.database import (
            get_business_session,
            get_timeseries_session,
            get_db_session_dependency,
            get_timeseries_session_dependency,
            get_db_session_legacy,
            get_business_health,
            get_timeseries_health,
            get_business_db_manager,
            get_timeseries_db_manager,
            get_db_health,
        )
        # 全部 callable
        for fn in [
            get_business_session, get_timeseries_session,
            get_db_session_dependency, get_timeseries_session_dependency,
            get_db_session_legacy, get_business_health, get_timeseries_health,
            get_db_session_legacy, get_db_health,
        ]:
            assert callable(fn)


# ============================================================
# 2. dependency 函数：async generator
# ============================================================
class TestDependencyFunctions:
    def test_get_db_session_dependency_is_async_gen(self):
        from infra.db.database import get_db_session_dependency
        # 应该是 async generator function
        assert inspect.isasyncgenfunction(get_db_session_dependency)

    def test_get_timeseries_session_dependency_is_async_gen(self):
        from infra.db.database import get_timeseries_session_dependency
        assert inspect.isasyncgenfunction(get_timeseries_session_dependency)


# ============================================================
# 3. legacy / health 是 async 函数
# ============================================================
class TestAsyncFunctions:
    def test_get_db_session_legacy_is_async(self):
        from infra.db.database import get_db_session_legacy
        assert inspect.iscoroutinefunction(get_db_session_legacy)

    def test_get_db_health_is_async(self):
        from infra.db.database import get_db_health
        assert inspect.iscoroutinefunction(get_db_health)


# ============================================================
# 4. 转发验证
# ============================================================
class TestForwarding:
    async def test_get_db_health_forwards_to_get_business_health(self):
        from infra.db.database import get_db_health
        with patch("infra.db.database.get_business_health", new=AsyncMock(return_value={"status": "ok"})) as mock:
            result = await get_db_health()
        assert result == {"status": "ok"}
        mock.assert_awaited_once()

    async def test_get_db_session_legacy_forwards(self):
        from infra.db.database import get_db_session_legacy

        class FakeSession:
            pass

        fake_session = FakeSession()

        class FakeCtx:
            async def __aenter__(self):
                return fake_session
            async def __aexit__(self, *args):
                return False

        # get_business_session 是 @asynccontextmanager 函数；
        # 调用它直接返回 async context manager（无需 await）
        def fake_get_business_session():
            return FakeCtx()

        with patch("infra.db.database.get_business_session", side_effect=fake_get_business_session):
            result = await get_db_session_legacy()
        assert result is fake_session


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
