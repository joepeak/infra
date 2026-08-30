#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库模块入口

API 设计：按 db_config_key 注册 db manager——
yaml 加一个块 + init_db_manager(key) 即接入新数据库，
get_db_manager(key) / get_db_session(key) 任意取用。
"""

from infra.db.connection_manager import (
    # 注册表
    init_db_manager,
    get_db_manager,
    get_db_manager_or_raise,
    get_db_session,
    session_scope,
    get_db_health,
    list_db_keys,
    close_all_db_connections,
    # 类
    DatabaseConnectionManager,
)

from infra.db.operations import db_operation
from infra.db.database_repository import DatabaseRepository
from infra.db.database_model import (
    Base, BaseModel, ObservationBaseModel, TimeSeriesBaseModel,
    ObservationBase, TimeUtil, JSONBCompatible,
)

# infra.time_util.TimeUtil 是旧 class shim——和 infra.db.database_model.TimeUtil 同名冲突
# 旧路径是 backward compat；infra.db 自己用 database_model.TimeUtil
# 此处不 import infra.time_util，避免覆盖

__all__ = [
    # 注册表 API
    'init_db_manager',
    'get_db_manager',
    'get_db_manager_or_raise',
    'get_db_session',
    'session_scope',
    'get_db_health',
    'list_db_keys',
    'close_all_db_connections',
    # 类
    'DatabaseConnectionManager',
    # 通用
    'db_operation',
    'DatabaseRepository',
    'Base',
    'BaseModel',
    'TimeSeriesBaseModel',
    'ObservationBaseModel',
    'TimeUtil',
    'JSONBCompatible',
]