#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db.database — FastAPI DI shim（仅保留向后兼容）

早期 infra.db.database 是 connection_manager 的 1-to-1 shim 集合
（get_business_session / get_timeseries_session / get_db_session_dependency 等）。
激进方案把 6 个 shim 收编为 3 个统一 API（init_db_manager / get_db_session /
session_scope），database.py 只剩 FastAPI 依赖注入场景常用的两个。
"""

from typing import AsyncGenerator

from infra.db.connection_manager import get_db_session, get_db_health
from infra.exceptions import DatabaseError  # noqa: F401  # 兼容旧 import


async def get_db_session_dependency() -> AsyncGenerator:
    """FastAPI 依赖注入：业务库 session（key="db"）。"""
    async with get_db_session("db") as session:
        yield session


__all__ = [
    'get_db_session_dependency',
]
