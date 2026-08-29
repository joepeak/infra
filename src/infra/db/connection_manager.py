#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库连接管理器 - 异步版，支持 PostgreSQL + SQLite（可选）
"""

import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Optional, Dict, Any, AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.pool import StaticPool

from infra.logger import get_logger
from infra.exceptions import DatabaseError
from infra.config import get_config

logger = get_logger(__name__)


# ============================================================
# Ambient session（Unit of Work 模式）
# ============================================================
# session_scope() 内的所有 DatabaseRepository 调用复用同一 session，
# 由 session_scope 统一 commit/rollback，实现跨引擎 ACID 事务。
# ContextVar 按 asyncio task 隔离 + token reset，不会污染 standalone 调用。
_current_session: ContextVar[Optional[AsyncSession]] = ContextVar(
    "macro_monitor_current_session", default=None
)


def get_current_session() -> Optional[AsyncSession]:
    """获取当前 ambient session（无则 None，调用方应回退到独立 session）。"""
    return _current_session.get()


@asynccontextmanager
async def session_scope(db_type: str = "timeseries") -> AsyncGenerator[AsyncSession, None]:
    """Unit of Work 事务边界：开一个 session 设为 ambient，成功 commit / 异常 rollback。

    scope 内所有 DatabaseRepository.create/update/query/find_by 复用此 session；
    scope 退出时统一提交或回滚。用法：
        async with session_scope():
            await repo.create(...)   # 复用 ambient session，不单独 commit
            await other_repo.create(...)
        # exit → 统一 commit（原子）；异常 → 统一 rollback
    """
    manager = get_timeseries_db_manager() if db_type == "timeseries" else get_business_db_manager()
    if manager.async_session_factory is None:
        raise DatabaseError("异步会话工厂未初始化", operation="session_scope")

    session = manager.async_session_factory()
    token = _current_session.set(session)
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"session_scope 事务回滚: {e}")
        raise
    finally:
        _current_session.reset(token)
        await session.close()


class DatabaseConnectionManager:
    """
    异步数据库连接管理器 - 支持多数据库

    Args:
        db_config_key: 配置键名
            - "db" 用于业务数据库 (PostgreSQL)
            - "timescaledb" 用于时序数据库 (TimescaleDB)
        require_db: 是否强制要求配置存在（默认 True，严格模式）
    """

    def __init__(self, db_config_key: str = "db", require_db: bool = True):
        self.db_config_key = db_config_key
        self.require_db = require_db
        self.engine: Optional[AsyncEngine] = None
        self.async_session_factory: Optional[async_sessionmaker] = None
        self.config = self._load_config()

        if self.config:
            self._initialize_engine()
        else:
            logger.info(f"无配置（{db_config_key}）：跳过数据库连接初始化")

    def _load_config(self) -> Dict[str, Any]:
        """加载数据库配置，支持优雅降级或强制报错"""
        full_config = get_config()
        db_config = full_config.get(self.db_config_key)

        if not db_config:
            if self.require_db:
                raise DatabaseError(
                    f"数据库配置缺失: {self.db_config_key}，请检查 config.yaml 中的数据库配置",
                    operation="load_config"
                )
            logger.warning(f"数据库配置 {self.db_config_key} 未找到，跳过初始化")
            return {}

        required_fields = ['host', 'port', 'user', 'password', 'database']
        # 如果使用 URL 直连，则不需要拆分字段
        if 'url' not in db_config:
            missing_fields = [f for f in required_fields if f not in db_config]
            if missing_fields:
                raise DatabaseError(
                    f"数据库配置 {self.db_config_key} 缺少必要字段: {missing_fields}",
                    operation="load_config"
                )

        engine_defaults = {
            'pool_pre_ping': True,
            'pool_recycle': 3600,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'echo': False,
            'connect_timeout': 10
        }
        engine_config = db_config.get('engine', {})
        merged_engine_config = engine_defaults.copy()
        merged_engine_config.update(engine_config)
        db_config['engine'] = merged_engine_config

        return db_config

    def _initialize_engine(self) -> None:
        """初始化异步数据库引擎"""
        try:
            self._create_engine()
            self._create_session_factory()
            logger.info(f"异步数据库引擎初始化成功 ({self.db_config_key})")
        except Exception as e:
            logger.error(f"异步数据库引擎初始化失败: {e}")
            raise DatabaseError(f"异步数据库引擎初始化失败: {e}", operation="initialize_engine")

    def _create_engine(self) -> None:
        """创建异步数据库引擎，优先使用url，其次拼接"""
        db_config = self.config
        engine_config = db_config.get('engine', {})

        database_url = db_config.get('url')
        if not database_url:
            logger.info("URL未提供，拼接PostgreSQL连接字符串")
            host = db_config['host']
            port = db_config['port']
            user = db_config['user']
            password = db_config['password']
            database = db_config['database']
            database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

        # 基础引擎参数
        engine_args = {
            "pool_pre_ping": engine_config.get('pool_pre_ping', True),
            "pool_recycle": engine_config.get('pool_recycle', 3600),
            "pool_size": engine_config.get('pool_size', 10),
            "max_overflow": engine_config.get('max_overflow', 20),
            "pool_timeout": engine_config.get('pool_timeout', 30),
            "echo": engine_config.get('echo', False),
            "pool_reset_on_return": 'commit',
        }

        if "sqlite" in database_url:
            logger.info("检测到SQLite数据库，移除不支持的连接池参数")
            # SQLite 不支持这些参数
            engine_args.pop("pool_size", None)
            engine_args.pop("max_overflow", None)
            engine_args.pop("pool_timeout", None)
            engine_args.pop("pool_recycle", None)
            engine_args["poolclass"] = StaticPool
            engine_args["connect_args"] = {'check_same_thread': False}
        else:
            engine_args["connect_args"] = {
                'timeout': engine_config.get('connect_timeout', 10),
                'command_timeout': 60,
                'server_settings': {
                    'application_name': 'dramacraft',
                }
            }
            if 'connect_args' in engine_config:
                engine_args["connect_args"].update(engine_config['connect_args'])

        self.engine = create_async_engine(database_url, **engine_args)
        logger.info(f"异步引擎创建成功 for {self.db_config_key}")

    def _create_session_factory(self) -> None:
        """创建异步会话工厂"""
        self.async_session_factory = async_sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取异步数据库会话上下文管理器"""
        if self.async_session_factory is None:
            raise DatabaseError("异步会话工厂未初始化", operation="get_session")

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"异步数据库会话操作失败: {e}")
            raise DatabaseError(f"异步数据库会话操作失败: {e}", operation="session_transaction")
        finally:
            await session.close()

    async def health_check(self) -> Dict[str, Any]:
        """数据库健康检查"""
        health_status = {
            'status': 'healthy',
            'db_config_key': self.db_config_key,
            'errors': []
        }

        try:
            if self.engine is None:
                health_status['status'] = 'unhealthy'
                health_status['errors'].append('数据库引擎未初始化')
                return health_status

            async with self.engine.connect() as conn:
                result = await conn.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()
                if row[0] != 1:
                    health_status['status'] = 'unhealthy'
                    health_status['errors'].append('健康检查查询失败')

            logger.debug(f"异步数据库健康检查完成 ({self.db_config_key})")

        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append(str(e))
            logger.error(f"异步数据库健康检查失败: {e}")

        return health_status

    async def close(self) -> None:
        """关闭数据库连接"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info(f"异步数据库连接已关闭 ({self.db_config_key})")
        except Exception as e:
            logger.error(f"关闭异步数据库连接时出错: {e}")

    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        if not self.engine:
            return {'error': '数据库引擎未初始化'}

        pool = self.engine.pool
        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
        }

    def get_engine(self) -> Optional[AsyncEngine]:
        """获取当前管理器使用的 SQLAlchemy 引擎实例"""
        return self.engine


# ============================================================
# 全局数据库管理器实例
# ============================================================

_business_db_manager: Optional[DatabaseConnectionManager] = None
_timeseries_db_manager: Optional[DatabaseConnectionManager] = None


async def init_db_manager(require_db: bool = False) -> DatabaseConnectionManager:
    """初始化全局业务数据库管理器实例"""
    global _business_db_manager
    if _business_db_manager is not None:
        logger.warning("业务数据库管理器实例已存在，将被替换")

    _business_db_manager = DatabaseConnectionManager("db", require_db=require_db)
    logger.info("全局业务数据库管理器初始化完成")
    return _business_db_manager


async def init_timeseries_db_manager(require_db: bool = False) -> DatabaseConnectionManager:
    """初始化全局时序数据库管理器实例"""
    global _timeseries_db_manager
    if _timeseries_db_manager is not None:
        logger.warning("时序数据库管理器实例已存在，将被替换")

    _timeseries_db_manager = DatabaseConnectionManager("timescaledb", require_db=require_db)
    logger.info("全局时序数据库管理器初始化完成")
    return _timeseries_db_manager


def get_business_db_manager() -> DatabaseConnectionManager:
    """获取业务数据库管理器实例"""
    global _business_db_manager
    if _business_db_manager is None:
        # 兼容旧代码：尝试用默认参数初始化
        logger.warning("业务数据库管理器未初始化，使用默认配置（降级模式）")
        _business_db_manager = DatabaseConnectionManager("db", require_db=False)
    return _business_db_manager


def get_timeseries_db_manager() -> DatabaseConnectionManager:
    """获取时序数据库管理器实例"""
    global _timeseries_db_manager
    if _timeseries_db_manager is None:
        logger.warning("时序数据库管理器未初始化，使用默认配置（降级模式）")
        _timeseries_db_manager = DatabaseConnectionManager("timescaledb", require_db=False)
    return _timeseries_db_manager


@asynccontextmanager
async def get_business_session() -> AsyncGenerator[AsyncSession, None]:
    """获取业务数据库会话"""
    async with get_business_db_manager().get_session() as session:
        yield session


@asynccontextmanager
async def get_timeseries_session() -> AsyncGenerator[AsyncSession, None]:
    """获取时序数据库会话"""
    async with get_timeseries_db_manager().get_session() as session:
        yield session


async def get_business_health() -> Dict[str, Any]:
    """获取业务数据库健康状态"""
    return await get_business_db_manager().health_check()


async def get_timeseries_health() -> Dict[str, Any]:
    """获取时序数据库健康状态"""
    return await get_timeseries_db_manager().health_check()


async def close_all_db_connections() -> None:
    """关闭所有数据库连接"""
    global _business_db_manager, _timeseries_db_manager

    if _business_db_manager:
        await _business_db_manager.close()
        _business_db_manager = None

    if _timeseries_db_manager:
        await _timeseries_db_manager.close()
        _timeseries_db_manager = None

    logger.info("所有异步数据库连接已关闭")