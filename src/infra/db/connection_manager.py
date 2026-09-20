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
    "current_session", default=None
)


def get_current_session() -> Optional[AsyncSession]:
    """获取当前 ambient session（无则 None，调用方应回退到独立 session）。"""
    return _current_session.get()


class DatabaseConnectionManager:
    """
    异步数据库连接管理器 - 支持任意多数据库

    Args:
        db_config_key: 配置键名（与 yaml 子配置块同名）
            - "db" 默认业务库
            - "timescaledb" 时序库
            - 业务项目可传任意 key（"analytics" / "cache" 等）
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

        # db_type 优先从 config 子节点的 type 字段读；
        # 若 type 未显式配置（空字符串""），推迟到 _resolve_url_and_type 从 URL dialect 推断
        db_type = str(
            db_config.get("type",
                          full_config.get("db_type", ""))
        ).lower()
        db_config["type"] = db_type  # 下游 _resolve_url_and_type/_create_engine 直接读

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
        db_config['type'] = db_type

        return db_config  # type: ignore[no-any-return]

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
                return m.group(0)  # type: ignore[no-any-return]
            return val

        return pattern.sub(_sub, value)

    @staticmethod
    def _is_sqlite_engine(engine: Any) -> bool:
        """判断 engine 是否 SQLite（用 engine.dialect.name 单一真理）。"""
        if engine is None:
            return False
        dialect_name = getattr(getattr(engine, "dialect", None), "name", "")
        return dialect_name.startswith("sqlite")

    # 各 db_type 优先查的环境变量（fallback 顺序）
    _DB_TYPE_ENV_URL_KEYS: Dict[str, tuple] = {
        "postgresql": ("POSTGRES_URL", "DATABASE_URL"),
        "mysql": ("MYSQL_URL",),
        "sqlite": (),
    }

    def _resolve_url_and_type(self, db_config: Dict[str, Any]) -> tuple[str, str]:
        """
        决定 db_type 和最终 url。

        URL 优先级：
        1. db.url 是字面量（不含 ${...}）→ 走字面量，env 不可覆盖
        2. db.url 含 ${VAR} 语法 → 强制走 env 展开
        3. db.url 字段不存在 → 按 db_type 查对应 env 变量（POSTGRES_URL / MYSQL_URL 等）
        4. 兜底：按 host/port/user/password/database 拼接（仅 PG / MySQL 适用）

        Returns:
            (db_type, database_url) 元组
        """
        db_type = str(db_config.get("type", "")).lower()

        # 1+2: db.url 路径
        url_value = db_config.get("url")
        database_url: Optional[str] = None

        if url_value is not None:
            url_str = str(url_value)
            has_env_syntax = "${" in url_str
            database_url = self._expand_env_vars(url_str) if has_env_syntax else url_str

        # 若 type 未显式指定（空字符串），从 URL dialect 推断
        if not db_type and database_url:
            scheme = database_url.split("://", 1)[0] if "://" in database_url else ""
            if scheme.startswith("sqlite"):
                db_type = "sqlite"
            elif scheme in ("postgres", "postgresql"):
                db_type = "postgresql"
            elif scheme in ("mysql", "mysql+aiomysql"):
                db_type = "mysql"

        # 3: yaml 无 url 字段 → 按 type 查环境变量
        if not database_url:
            env_keys = self._DB_TYPE_ENV_URL_KEYS.get(db_type, ())
            for env_key in env_keys:
                env_url = self._expand_env_vars(os.getenv(env_key))
                if env_url:
                    database_url = env_url
                    logger.info(f"db.url 未配置，使用环境变量 {env_key}")
                    break

        # 4: 兜底分项拼接
        if not database_url:
            database_url = self._build_url_from_components(db_config, db_type)

        # url 缺驱动前缀时按 type 补齐
        database_url = self._ensure_dialect_prefix(database_url, db_type)

        # 日志脱敏 + 措辞
        masked_url = re.sub(
            r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", database_url
        ) if database_url else ""
        if database_url:
            # 简明 log：实际从哪条路径来的在前面已经 log 过
            logger.info(f"数据库引擎 url: {masked_url} (type={db_type})")

        return db_type, database_url

    def _build_url_from_components(
        self, db_config: Dict[str, Any], db_type: str
    ) -> str:
        """分项拼接 URL；缺字段抛 DatabaseError。"""
        host = db_config.get("host")
        port = db_config.get("port")
        user = self._expand_env_vars(db_config.get("user"))
        password = self._expand_env_vars(db_config.get("password"))
        database = db_config.get("database")
        missing = [
            k for k, v in (
                ("host", host), ("user", user),
                ("password", password), ("database", database),
            ) if v in (None, "")
        ]
        if missing:
            raise DatabaseError(
                f"数据库配置不完整：db.url 未提供且分项缺少 {missing}（检查 configs/*.yaml 与 .env）",
                operation="load_config",
            )
        if db_type == "sqlite":
            return f"sqlite+aiosqlite:///{database}"
        if db_type == "postgresql":
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        if db_type == "mysql":
            # MySQL async driver 待定；此处占位
            return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"
        raise DatabaseError(
            f"db_type={db_type} 暂不支持分项拼接，请用 url 字段",
            operation="load_config",
        )

    def _ensure_dialect_prefix(self, url: str, db_type: str) -> str:
        """url 缺驱动前缀时按 db_type 补齐；裸 dialect 升级为 async driver。

        已知 dialect：postgres / postgresql / sqlite / mysql → 升级为 async driver
        其它 scheme（含 +asyncpg / +aiosqlite / +aiomysql）→ 原样
        """
        if "://" in url:
            scheme = url.split("://", 1)[0]
            # 已知 dialect 但无 async driver——升级
            if scheme == "postgres":
                return url.replace("postgres://", "postgresql+asyncpg://", 1)
            if scheme == "postgresql":
                return url.replace("postgresql://", "postgresql+asyncpg://", 1)
            if scheme == "sqlite":
                return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
            if scheme == "mysql":
                return url.replace("mysql://", "mysql+aiomysql://", 1)
            # 已是 async driver 或未知 driver（clickhouse / oracle / duckdb / ...）→ 原样
            return url
        # 无前缀——按 type 补
        if db_type == "postgresql":
            return f"postgresql+asyncpg://{url}"
        if db_type == "sqlite":
            return f"sqlite+aiosqlite:///{url}"
        if db_type == "mysql":
            return f"mysql+aiomysql://{url}"
        return url

    def _build_engine_args(
        self, db_type: str, engine_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        按 db_type 返回 create_async_engine 的 kwargs（一次到位，不再重建）。

        关键差异化：
        - postgresql: 完整连接池 + asyncpg connect_args (server_settings, timeout)
        - sqlite: StaticPool + check_same_thread=False（多线程兼容）
        - mysql:   基本池 + 占位 connect_args

        备选：用 create_async_engine 后 engine.dialect.name 二次判断更稳，
        但 type 是显式声明——业务 yaml 想用 SQLite 时应写 type: sqlite。
        """
        # 通用默认值
        args: Dict[str, Any] = {
            "echo": engine_config.get("echo", False),
        }

        if db_type == "sqlite":
            # SQLite 不支持池参数
            args["poolclass"] = StaticPool
            args["connect_args"] = {
                "check_same_thread": engine_config.get("check_same_thread", False),
            }
        elif db_type == "postgresql":
            # asyncpg 专属 connect_args
            server_settings: Dict[str, Any] = dict(
                engine_config.get("server_settings", {})
            )
            args.update({
                "pool_pre_ping": engine_config.get("pool_pre_ping", True),
                "pool_recycle": engine_config.get("pool_recycle", 3600),
                "pool_size": engine_config.get("pool_size", 10),
                "max_overflow": engine_config.get("max_overflow", 20),
                "pool_timeout": engine_config.get("pool_timeout", 30),
                "pool_reset_on_return": "commit",
            })
            args["connect_args"] = {
                "timeout": engine_config.get("connect_timeout", 10),
                "command_timeout": engine_config.get("command_timeout", 60),
                "server_settings": server_settings,
            }
            # 用户 connect_args 覆盖
            if "connect_args" in engine_config:
                args["connect_args"].update(engine_config["connect_args"])
        elif db_type == "mysql":
            # MySQL async 暂未用上——留接口
            args.update({
                "pool_pre_ping": engine_config.get("pool_pre_ping", True),
                "pool_recycle": engine_config.get("pool_recycle", 3600),
                "pool_size": engine_config.get("pool_size", 10),
                "max_overflow": engine_config.get("max_overflow", 20),
                "pool_timeout": engine_config.get("pool_timeout", 30),
            })
        else:
            raise DatabaseError(
                f"不支持的 db_type: {db_type}（已实现：postgresql/sqlite/mysql）",
                operation="create_engine",
            )

        return args

    def _create_engine(self) -> None:
        """创建异步数据库引擎（一次创建——按 db_type 走差异化参数）。

        db_type 决定路径：URL 解析 + 引擎参数构建都用 db_type。
        若 type 未显式配置，自 URL dialect 推断（sqlite+aiosqlite → sqlite）。
        """
        db_config = self.config
        engine_config = db_config.get("engine", {})

        # 1. 决定 db_type + url
        db_type, database_url = self._resolve_url_and_type(db_config)

        # 2. 按 type 返回差异化 engine_args
        engine_args = self._build_engine_args(db_type, engine_config)

        # 3. 一次创建
        self.engine = create_async_engine(database_url, **engine_args)
        logger.info(
            f"异步引擎创建成功 for {self.db_config_key}"
            f"（dialect={self.engine.dialect.name}, type={db_type}）"
        )

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
        health_status: Dict[str, Any] = {
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
                if row is None or row[0] != 1:
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
        # mypy 推断 Pool 是 sync 版（AsyncAdaptedQueuePool）——调方法用 type ignore
        return {
            'pool_size': pool.size(),  # type: ignore[attr-defined]
            'checked_in': pool.checkedin(),  # type: ignore[attr-defined]
            'checked_out': pool.checkedout(),  # type: ignore[attr-defined]
            'overflow': pool.overflow(),  # type: ignore[attr-defined]
        }

    def get_engine(self) -> Optional[AsyncEngine]:
        """获取当前管理器使用的 SQLAlchemy 引擎实例"""
        return self.engine


# ============================================================
# 多 db manager 注册表（按 db_config_key 索引）
# ============================================================
# 替代原来的 _business_db_manager / _timeseries_db_manager 双写死 slot。
# 用 dict 支持任意多个 db 实例——yaml 加一个块、init 时传 key 即可。
# ============================================================
_db_managers: Dict[str, DatabaseConnectionManager] = {}


def init_db_manager(key: str = "db", require_db: bool = False) -> DatabaseConnectionManager:
    """
    按 key 初始化/重建 db manager（同步——__init__ 不涉及网络）。

    key 对应 yaml 配置块名（如 "db" / "timescaledb" / "analytics"）。
    同一 key 重复 init 会替换并 warn（便于 reload 测试场景）。

    Args:
        key: db 配置 key（同时作为 manager 标识 + yaml 子配置块名）
        require_db: 是否强制要求配置存在（False 时优雅降级）
    """
    if key in _db_managers:
        logger.warning(f"db manager '{key}' 已存在，将被替换")

    manager = DatabaseConnectionManager(db_config_key=key, require_db=require_db)
    _db_managers[key] = manager
    logger.info(f"db manager '{key}' 初始化完成")
    return manager


def get_db_manager(key: str = "db") -> Optional[DatabaseConnectionManager]:
    """
    按 key 取 db manager。未初始化返 None（不抛错——便于测试降级场景）。

    想 strict 模式请用 get_db_manager_or_raise。
    """
    return _db_managers.get(key)


def get_db_manager_or_raise(key: str = "db") -> DatabaseConnectionManager:
    """取 db manager；未初始化抛 RuntimeError（业务调用首选）。"""
    mgr = _db_managers.get(key)
    if mgr is None:
        raise RuntimeError(
            f"db manager '{key}' 未初始化——请先调用 init_db_manager('{key}', ...)"
        )
    return mgr


@asynccontextmanager
async def get_db_session(key: str = "db") -> AsyncGenerator[AsyncSession, None]:
    """按 key 取 db session 上下文管理器。"""
    async with get_db_manager_or_raise(key).get_session() as session:
        yield session


@asynccontextmanager
async def session_scope(key: str = "db") -> AsyncGenerator[AsyncSession, None]:
    """Unit of Work 事务边界（按 key）。

    scope 内所有 DatabaseRepository 调用复用同一 session，由本 scope 统一
    commit/rollback——实现跨引擎 ACID 事务。ContextVar 隔离，scope 退出后
    ambient session 自动 reset。
    """
    manager = get_db_manager_or_raise(key)
    if manager.async_session_factory is None:
        raise DatabaseError(
            f"异步会话工厂未初始化 (key='{key}')", operation="session_scope"
        )

    session = manager.async_session_factory()
    token = _current_session.set(session)
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"session_scope 事务回滚 (key='{key}'): {e}")
        raise
    finally:
        _current_session.reset(token)
        await session.close()


async def get_db_health(key: str = "db") -> Dict[str, Any]:
    """按 key 取 db 健康状态。"""
    return await get_db_manager_or_raise(key).health_check()


def list_db_keys() -> list[str]:
    """列出已初始化的所有 db key（诊断/测试用）。"""
    return list(_db_managers.keys())


async def close_all_db_connections() -> None:
    """关闭所有 db 连接并清空注册表。"""
    for key, manager in list(_db_managers.items()):
        try:
            await manager.close()
        except Exception as e:
            logger.warning(f"关闭 db manager '{key}' 失败: {e}")
    _db_managers.clear()
    logger.info("所有异步数据库连接已关闭")