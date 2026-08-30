#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db 新 API（激进方案 Map 化后）测试

覆盖：
1. _db_managers 注册表：init_db_manager / get_db_manager / get_db_manager_or_raise / list_db_keys / close_all_db_connections
2. DatabaseConnectionManager._create_engine 重构后行为：
   - 单次创建 engine（不再 dispose+重建）
   - _build_engine_args 按 db_type 返回差异化参数
   - _resolve_url_and_type URL 优先级（yaml 字面量 > ${VAR} > env 变量 > 分项拼接）
3. db_type fallback：子节点 type 优先，顶层 db_type 作兜底
4. _load_config：db_type 写到 db_config['type']
5. application_name 已被删（不再硬编码 'crypto-watcher'）
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

import infra.config.loader as loader_module
from infra.config import init_config
from infra.config.loader import ConfigLoader
import infra.db.connection_manager as cm_module
from infra.db.connection_manager import (
    DatabaseConnectionManager,
    _db_managers,
    close_all_db_connections,
    get_db_manager,
    get_db_manager_or_raise,
    init_db_manager,
    list_db_keys,
)


# ============================================================
# Fixture：每个测试前后重置 config + db_managers
# ============================================================
@pytest.fixture(autouse=True)
def _reset_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """清 config loader 状态 + db_managers dict"""
    saved_env = {}
    for k in ("MACRO_MONITOR_ENV", "ENVIRONMENT",
              "POSTGRES_URL", "DATABASE_URL", "MYSQL_URL"):
        saved_env[k] = os.environ.pop(k, None)

    loader_module._config = None
    loader_module._config_loaded = False
    loader_module._loader = None
    ConfigLoader._instance = None
    _db_managers.clear()

    yield

    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    loader_module._config = None
    loader_module._config_loaded = False
    loader_module._loader = None
    ConfigLoader._instance = None
    _db_managers.clear()


def _write_cfg(dir_: Path, name: str, data: Dict[str, Any]) -> None:
    (dir_ / name).write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


# ============================================================
# 1. _db_managers 注册表 API
# ============================================================
class TestDbManagerRegistry:
    def _make_cfg(self, tmp_path: Path) -> Path:
        _write_cfg(tmp_path, "config.yaml", {
            "db": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
        })
        return tmp_path

    def test_init_registers_in_dict(self, tmp_path):
        self._make_cfg(tmp_path)
        init_config(tmp_path)
        mgr = init_db_manager("db")
        assert mgr is get_db_manager("db")
        assert "db" in list_db_keys()

    def test_get_db_manager_returns_none_if_uninitialized(self):
        assert get_db_manager("db") is None

    def test_get_db_manager_or_raise_raises(self):
        with pytest.raises(RuntimeError, match="db manager 'db' 未初始化"):
            get_db_manager_or_raise("db")

    def test_multiple_keys(self, tmp_path):
        """多 db key 注册——激进方案核心。"""
        _write_cfg(tmp_path, "config.yaml", {
            "db": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
            "timescaledb": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
            "analytics": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
        })
        init_config(tmp_path)
        m1 = init_db_manager("db")
        m2 = init_db_manager("timescaledb")
        m3 = init_db_manager("analytics")
        # 三个独立实例
        assert m1 is not m2 and m2 is not m3 and m1 is not m3
        assert set(list_db_keys()) == {"db", "timescaledb", "analytics"}

    def test_init_replaces_existing_warns(self, tmp_path):
        self._make_cfg(tmp_path)
        init_config(tmp_path)
        # 先注册一次，第二次 init 应 warn
        init_db_manager("db")
        # 抓 logger 的所有记录
        import logging
        handler_records = []
        handler = _ListHandler(handler_records)
        logger_obj = logging.getLogger("infra.db.connection_manager")
        logger_obj.addHandler(handler)
        try:
            init_db_manager("db")  # 第二次
            warnings = [r for r in handler_records if "已存在" in r.getMessage() and "替换" in r.getMessage()]
            assert len(warnings) == 1
            assert "db manager 'db'" in warnings[0].getMessage()
        finally:
            logger_obj.removeHandler(handler)

    def test_close_all_clears_dict(self, tmp_path):
        self._make_cfg(tmp_path)
        init_config(tmp_path)
        init_db_manager("db")
        init_db_manager("timescaledb")
        assert len(list_db_keys()) == 2
        # close 是 async
        import asyncio
        asyncio.run(close_all_db_connections())
        assert list_db_keys() == []


class _ListHandler(logging.Handler):
    """测试用 handler——记录所有 LogRecord 到 list 便于断言。"""

    def __init__(self, records: list):
        super().__init__(level=logging.DEBUG)  # 收所有 level
        self.records = records

    def emit(self, record):
        self.records.append(record)


# ============================================================
# 2. _build_engine_args 按 db_type 差异化
# ============================================================
class TestBuildEngineArgs:
    def test_postgresql_includes_pool_and_asyncpg_args(self):
        cfg = {"pool_size": 5, "connect_timeout": 8}
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        args = mgr._build_engine_args("postgresql", cfg)
        assert args["pool_size"] == 5
        assert "poolclass" not in args
        assert "connect_args" in args
        assert "command_timeout" in args["connect_args"]
        assert "server_settings" in args["connect_args"]

    def test_sqlite_uses_staticpool_no_pool_args(self):
        cfg: Dict[str, Any] = {}
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        args = mgr._build_engine_args("sqlite", cfg)
        from sqlalchemy.pool import StaticPool
        assert args["poolclass"] is StaticPool
        # SQLite 不应有 pool_size 等连接池参数
        assert "pool_size" not in args
        assert "max_overflow" not in args
        # connect_args 应有 check_same_thread
        assert args["connect_args"]["check_same_thread"] is False

    def test_mysql_basic_pool(self):
        cfg: Dict[str, Any] = {}
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        args = mgr._build_engine_args("mysql", cfg)
        assert "pool_size" in args
        # MySQL 不应走 asyncpg 专属 connect_args
        assert "server_settings" not in args.get("connect_args", {})

    def test_unsupported_type_raises(self):
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        with pytest.raises(Exception, match="不支持的 db_type"):
            mgr._build_engine_args("oracle", {})

    def test_no_application_name_in_args(self):
        """激进方案：application_name 已删——不应在 args 里。"""
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        for db_type in ("postgresql", "sqlite", "mysql"):
            args = mgr._build_engine_args(db_type, {})
            # server_settings 可能含 user 配的，但不应有 'application_name' 默认值
            if db_type == "postgresql":
                assert "application_name" not in args["connect_args"].get("server_settings", {})


# ============================================================
# 3. _resolve_url_and_type URL 优先级
# ============================================================
class TestResolveUrlAndType:
    def test_yaml_literal_url_wins(self):
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)  # 不走 __init__
        cfg = {"type": "postgresql", "url": "postgresql+asyncpg://literal/x"}
        db_type, url = mgr._resolve_url_and_type(cfg)
        assert db_type == "postgresql"
        # 字面量不展开 env
        assert url == "postgresql+asyncpg://literal/x"

    def test_yaml_url_with_env_syntax_expands(self, monkeypatch):
        monkeypatch.setenv("MY_DB_HOST", "db.example.com")
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        cfg = {"type": "postgresql", "url": "postgresql+asyncpg://user:pwd@${MY_DB_HOST}/x"}
        db_type, url = mgr._resolve_url_and_type(cfg)
        assert "db.example.com" in url

    def test_no_yaml_url_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("POSTGRES_URL", "postgresql://env.example.com/x")
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        cfg = {"type": "postgresql"}
        db_type, url = mgr._resolve_url_and_type(cfg)
        # env_url 会被 dialect prefix 补齐
        assert "env.example.com" in url
        assert url.startswith("postgresql+asyncpg://")

    def test_no_yaml_url_no_env_falls_back_to_components(self):
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        cfg = {
            "type": "postgresql",
            "host": "h", "port": 5432,
            "user": "u", "password": "p", "database": "d",
        }
        db_type, url = mgr._resolve_url_and_type(cfg)
        assert "h:5432" in url
        assert "u:p" in url or "%3A" in url  # URL 编码与否都接受
        assert "/d" in url

    def test_components_missing_raises(self):
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        cfg = {"type": "postgresql", "host": "h", "user": "u"}  # 缺 password/database
        with pytest.raises(Exception, match="分项缺少"):
            mgr._resolve_url_and_type(cfg)

    def test_dialect_prefix_ensured(self):
        """裸 scheme 应升级为 async driver（关键 bug 修复：sqlite://x 之前不升级导致用同步 driver）。"""
        mgr = DatabaseConnectionManager.__new__(DatabaseConnectionManager)
        # 无前缀
        assert mgr._ensure_dialect_prefix("no-prefix-url", "postgresql") == "postgresql+asyncpg://no-prefix-url"
        # 裸 scheme → 升级为 async driver
        assert mgr._ensure_dialect_prefix("postgres://x", "postgresql") == "postgresql+asyncpg://x"
        assert mgr._ensure_dialect_prefix("postgresql://x", "postgresql") == "postgresql+asyncpg://x"
        assert mgr._ensure_dialect_prefix("sqlite:///tmp/x.db", "sqlite") == "sqlite+aiosqlite:///tmp/x.db"
        assert mgr._ensure_dialect_prefix("mysql://x", "mysql") == "mysql+aiomysql://x"
        # 已是 async driver → 原样
        assert mgr._ensure_dialect_prefix("postgresql+asyncpg://x", "postgresql") == "postgresql+asyncpg://x"
        assert mgr._ensure_dialect_prefix("sqlite+aiosqlite:///x", "sqlite") == "sqlite+aiosqlite:///x"
        # 未知 scheme → 原样
        assert mgr._ensure_dialect_prefix("clickhouse://x", "postgresql") == "clickhouse://x"


# ============================================================
# 4. db_type fallback
# ============================================================
class TestDbTypeFallback:
    def test_subkey_type_takes_precedence(self, tmp_path):
        """子节点有 type 时，顶层 db_type 不影响。"""
        _write_cfg(tmp_path, "config.yaml", {
            "db_type": "postgresql",  # 顶层
            "db": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
        })
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)
        assert mgr.config["type"] == "sqlite"

    def test_top_level_db_type_used_when_subkey_missing(self, tmp_path):
        _write_cfg(tmp_path, "config.yaml", {
            "db_type": "postgresql",
            "db": {"url": "postgresql+asyncpg://u:p@h:5432/d"},
        })
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)
        assert mgr.config["type"] == "postgresql"

    def test_default_postgresql(self, tmp_path):
        """顶层 / 子节点都没 type → postgresql。"""
        _write_cfg(tmp_path, "config.yaml", {
            "db": {"url": "postgresql+asyncpg://u:p@h:5432/d"},
        })
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)
        assert mgr.config["type"] == "postgresql"


# ============================================================
# 5. _create_engine 单次创建
# ============================================================
class TestCreateEngineSingleShot:
    def test_create_engine_called_once(self, tmp_path):
        """不应有 dispose + 重建。"""
        _write_cfg(tmp_path, "config.yaml", {
            "db": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
        })
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)

        with patch("infra.db.connection_manager.create_async_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.dialect.name = "sqlite"
            mock_create.return_value = mock_engine
            mgr.engine = None
            mgr._create_engine()
            # 只调一次
            assert mock_create.call_count == 1
            # 不应调 sync_engine.dispose
            assert mock_engine.sync_engine.dispose.call_count == 0

    def test_no_application_name_passed_to_engine(self, tmp_path):
        """激进方案：application_name 已删——不应传给 create_async_engine。"""
        _write_cfg(tmp_path, "config.yaml", {
            "db": {"type": "postgresql", "url": "postgresql+asyncpg://u:p@h:5432/d"},
        })
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)

        with patch("infra.db.connection_manager.create_async_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.dialect.name = "postgresql"
            mock_create.return_value = mock_engine
            mgr.engine = None
            mgr._create_engine()
            # connect_args["server_settings"] 不应含 application_name
            kwargs = mock_create.call_args.kwargs
            server_settings = kwargs.get("connect_args", {}).get("server_settings", {})
            assert "application_name" not in server_settings


# ============================================================
# 6. _load_config 行为
# ============================================================
class TestLoadConfig:
    def test_missing_db_key_with_require_db_raises(self, tmp_path):
        _write_cfg(tmp_path, "config.yaml", {"other_key": {}})
        init_config(tmp_path)
        with pytest.raises(Exception, match="数据库配置缺失"):
            DatabaseConnectionManager("db", require_db=True)

    def test_missing_db_key_without_require_db_returns_empty(self, tmp_path):
        _write_cfg(tmp_path, "config.yaml", {"other_key": {}})
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)
        assert mgr.config == {}
        assert mgr.engine is None  # 不应尝试 init engine

    def test_url_present_skips_required_field_check(self, tmp_path):
        """有 url 时不必填 host/port/user/password/database。"""
        _write_cfg(tmp_path, "config.yaml", {
            "db": {"type": "sqlite", "url": "sqlite+aiosqlite:///:memory:"},
        })
        init_config(tmp_path)
        mgr = DatabaseConnectionManager("db", require_db=False)  # 不抛
        assert mgr.config["url"].startswith("sqlite")


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
