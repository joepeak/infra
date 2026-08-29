#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库模块 - 统一管理 PostgreSQL（业务库）和 TimescaleDB（时序库）
"""

from infra.db.connection_manager import (
    # 初始化
    init_db_manager,
    init_timeseries_db_manager,
    # 业务数据库
    get_business_db_manager,
    get_business_session,
    get_business_health,
    # 时序数据库
    get_timeseries_db_manager,
    get_timeseries_session,
    get_timeseries_health,
    # 关闭所有连接
    close_all_db_connections,

)

from infra.db.operations import db_operation
from infra.db.database_repository import DatabaseRepository
from infra.db.database_model import Base, BaseModel, ObservationBaseModel, TimeSeriesBaseModel, ObservationBase, TimeUtil, JSONBCompatible
from infra.time_util import TimeUtil

__all__ = [
    # 初始化
    'init_db_manager',
    'init_timeseries_db_manager',
    # 业务数据库
    'get_business_db_manager',
    'get_business_session',
    'get_business_health',
    # 时序数据库
    'get_timeseries_db_manager',
    'get_timeseries_session',
    'get_timeseries_health',
    # 关闭连接
    'close_all_db_connections',
    # 通用
    'db_operation',
    'DatabaseRepository',
    'Base',
    'BaseModel',
    'TimeSeriesBaseModel',
    'TimeUtil',
    'JSONBCompatible',
]