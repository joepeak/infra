#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库配置和连接管理
使用新的异步连接管理器提供更好的连接池管理和健康检查
"""

from datetime import datetime, timezone
from typing import Optional
from decimal import Decimal



# 导入新的异步连接管理器
from infra.db.connection_manager import (
    get_business_session,
    get_timeseries_session,
    get_business_health,
    get_timeseries_health,
    get_business_db_manager,
    get_timeseries_db_manager,
)


async def get_db_session_dependency():
    """FastAPI依赖注入的数据库会话（异步）"""
    async with get_business_session() as db:
        yield db


async def get_timeseries_session_dependency():
    """FastAPI依赖注入的时序数据库会话（异步）"""
    async with get_timeseries_session() as db:
        yield db


# ============================================================
# 向后兼容（保留原有函数名，但改为异步）
# ============================================================

async def get_db_session_legacy():
    """获取业务数据库会话（向后兼容）"""
    async with get_business_session() as session:
        return session


async def get_db_health():
    """获取业务数据库健康状态（向后兼容）"""
    return await get_business_health()


__all__ = [
    'get_business_session',
    'get_timeseries_session',
    'get_db_session_dependency',
    'get_timeseries_session_dependency',
    'get_db_session_legacy',
    'get_business_health',
    'get_timeseries_health',
    'get_business_db_manager',
    'get_timeseries_db_manager',
    'get_db_health',
]