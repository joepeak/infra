#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db.database shim 测试

激进方案收编后，database.py 只剩 FastAPI DI 用的 get_db_session_dependency。
测试覆盖：
1. 该函数是 async generator（FastAPI Depends 需要这个形态）
2. yield 出来的 session 来自 get_db_session("db")（mock 验证）
"""

from __future__ import annotations

import inspect
from unittest.mock import patch, AsyncMock

import pytest


class TestDependency:
    def test_is_async_gen(self):
        """get_db_session_dependency 是 async generator function——FastAPI Depends 必需。"""
        from infra.db.database import get_db_session_dependency
        assert inspect.isasyncgenfunction(get_db_session_dependency)


class TestForwarding:
    async def test_forwards_to_get_db_session_db_key(self):
        """应该走 get_db_session("db")——验证 key 写对（不是其它 db）。"""
        from infra.db.database import get_db_session_dependency

        class FakeSession:
            pass

        fake_session = FakeSession()

        class FakeCtx:
            async def __aenter__(self):
                return fake_session
            async def __aexit__(self, *args):
                return False

        captured = {"key": None}

        def fake_get_db_session(key):
            captured["key"] = key
            return FakeCtx()

        with patch("infra.db.database.get_db_session", side_effect=fake_get_db_session):
            gen = get_db_session_dependency()
            # 触发 generator body
            try:
                yielded = await gen.__anext__()
                assert yielded is fake_session
                assert captured["key"] == "db"
            finally:
                await gen.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
