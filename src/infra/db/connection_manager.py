#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库连接管理器 - 异步版，支持 PostgreSQL + SQLite（可选）
"""

import os
import re
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

    @staticmethod
    def _expand_env_vars(value: Any) -> Any:
        """展开字符串中的 ${VAR} 形式环境变量（来自 .env / 系统环境）。

        未定义的变量保留原样——不阻断启动，便于排查配置遗漏。
        """
        if not isinstance(value, str):
            return value
        import re
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def _sub(m: "re.Match") -> str:
            var = m.group(1)
            val = os.getenv(var)
            if val is None:
                logger.warning(f"环境变量 {var} 未定义（.env 或系统环境），保留占位符")
                return m.group(0)
            return val

        return pattern.sub(_sub, value)

    @staticmethod
    def _is_sqlite_engine(engine: Any) -> bool:
        """判断 engine 是否 SQLite（用 engine.dialect.name 单一真理）。"""
        if engine is None:
            return False
        dialect_name = getattr(getattr(engine, "dialect", None), "name", "")
        return dialect_name.startswith("sqlite")

    def _create_engine(self) -> None:
        """创建异步数据库引擎，优先使用 url，其次拼接。

        URL 解析策略（按优先级）：
        1. db.url（支持 ${ENV_VAR} 展开）
        2. db.writer.url（兼容旧结构）
        3. 环境变量 POSTGRES_URL / DATABASE_URL
        4. db.host/port/user/password/database 拼接

        SQLite 走 StaticPool——用 _is_sqlite_engine 后续判断（不要靠 URL 字符串解析）。
        """
        db_config = self.config
        engine_config = db_config.get('engine', {})
        db_type = str(db_config.get('type', 'postgresql')).lower()

        # 1. 从 db.url 取
        database_url = self._expand_env_vars(db_config.get('url'))

        # 2. 兼容旧 dev.yaml 的 writer 子结构
        if not database_url and isinstance(db_config.get('writer'), dict):
            database_url = self._expand_env_vars(db_config['writer'].get('url'))

        # 3. 整体回退：环境变量（来自 .env 或系统）
        if not database_url:
            for env_key in ("POSTGRES_URL", "DATABASE_URL"):
                env_url = self._expand_env_vars(os.getenv(env_key))
                if env_url:
                    if env_url.startswith("postgres://"):
                        env_url = env_url.replace("postgres://", "postgresql+asyncpg://", 1)
                    elif env_url.startswith("postgresql://"):
                        env_url = env_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                    database_url = env_url
                    logger.info(f"db.url 未配置，使用环境变量 {env_key}")
                    break

        # 4. url 缺驱动前缀时按 type 补齐
        if database_url:
            has_dialect = "://" in database_url and database_url.split("://", 1)[0].startswith(
                ("postgresql", "postgres", "sqlite", "mysql")
            )
            if not has_dialect:
                if db_type == "postgresql":
                    database_url = f"postgresql+asyncpg://{database_url}"
                elif db_type == "sqlite":
                    database_url = f"sqlite+aiosqlite:///{database_url}"
            # url 含密码——日志脱敏
            masked_url = re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", database_url) if database_url else ""
            logger.info(f"使用 URL 创建数据库引擎: {masked_url}")
        else:
            host = db_config.get('host')
            port = db_config.get('port')
            user = self._expand_env_vars(db_config.get('user'))
            password = self._expand_env_vars(db_config.get('password'))
            database = db_config.get('database')
            missing = [k for k, v in (("host", host), ("user", user), ("password", password), ("database", database)) if v in (None, "")]
            if missing:
                raise DatabaseError(
                    f"数据库配置不完整：db.url 未提供且分项缺少 {missing}（检查 configs/*.yaml 与 .env）",
                    operation="load_config",
                )
            if db_type == "sqlite":
                database_url = f"sqlite+aiosqlite:///{database}"
            else:
                database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
            masked_url = re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", database_url) if database_url else ""
            logger.info(f"URL 未提供，按 type={db_type} 拼接: {masked_url}")

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

        # application_name 配置化（不再硬编码 'dramacraft'）
        app_name = engine_config.get('application_name', 'crypto-watcher')
        server_settings = {
            'application_name': app_name,
        }
        # 允许用户覆盖
        custom_server_settings = engine_config.get('server_settings', {})
        server_settings.update(custom_server_settings)

        # 通用 connect_args
        connect_args = {
            'timeout': engine_config.get('connect_timeout', 10),
            'command_timeout': 60,
            'server_settings': server_settings,
        }
        if 'connect_args' in engine_config:
            connect_args.update(engine_config['connect_args'])

        self.engine = create_async_engine(database_url, **engine_args)

        # 5. **创建 engine 后**——用 engine.dialect.name 决定特性（不靠 URL 字符串）
        # SQLAlchemy 2.0+ create_async_engine 不允许事后改 connect_args——
        # 重建 engine 一次以应用正确参数（同步 dispose——非 async）
        if self._is_sqlite_engine(self.engine):
            logger.info("检测到 SQLite 数据库，移除不支持的连接池参数")
            engine_args.pop("pool_size", None)
            engine_args.pop("max_overflow", None)
            engine_args.pop("pool_timeout", None)
            engine_args.pop("pool_recycle", None)
            engine_args["poolclass"] = StaticPool
            engine_args["connect_args"] = {'check_same_thread': False}
        else:
            # PostgreSQL 等标准方言——使用 connect_args（之前默认值是 'dramacraft'）
            engine_args["connect_args"] = connect_args

        # 重新创建以应用最终参数
        try:
            self.engine.sync_engine.dispose()  # 同步 dispose 旧 engine
        except Exception:
            pass  # ignore——首次创建时无 sync_engine

        self.engine = create_async_engine(database_url, **engine_args)
        logger.info(f"异步引擎创建成功 for {self.db_config_key}（dialect={self.engine.dialect.name}）")

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