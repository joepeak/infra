#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra 项目的 pytest 公共配置

设计要点（参考 crypto-watcher/tests/conftest.py 风格）：
1. session 级：创建一个**临时 SQLite 数据库文件**，DB URL = sqlite+aiosqlite:///<tmpfile>。
   session 末 drop_all → 关闭连接 → 删除文件。零外部依赖（不依赖 PG / Redis）。
2. session 级 autouse DB fixture **只在测试主动声明 db_session/engine/repo 时**激活
   （用 is-in-conftest-but-non-autouse + 测试显式声明的模式——见下）。
   单元测试（test_retry / test_exceptions / ...）不依赖任何 DB fixture，**不会**触发
   session 级 bootstrap，跑得飞快。
3. 提供 function-scoped 通用 fixture：db_session / clean_db。
4. 不依赖 macro_monitor、不读 yaml config——infra 是库。
   直接给 init_config 喂一个空目录，bootstrap 走 env / config 注入的 URL。

环境前置：
- aiosqlite（已加入 dev extras，uv sync --extra dev 即装）
"""

from __future__ import annotations

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional

import pytest
import pytest_asyncio


# ============================================================
# 1. 把 src/ 加入 sys.path（让 `import infra.xxx` 可工作）
# ============================================================
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ============================================================
# 2. 全局：临时数据库文件句柄（session 级）
# ============================================================
_temp_db_file = None
_temp_db_path: Optional[str] = None


def _make_temp_sqlite_path() -> str:
    """创建临时 .db 文件，返回绝对路径。"""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False, prefix="infra_test_")
    f.close()  # 让 aiosqlite 自己打开；关闭只是释放 fd
    return f.name


# ============================================================
# 3. 日志（最小化）
# ============================================================
@pytest.fixture(scope="session", autouse=True)
def _configure_test_logging() -> None:
    """Session 级日志配置——参考文档风格。"""
    root = logging.getLogger()
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    file_h = logging.FileHandler(
        os.path.join(tempfile.gettempdir(), "infra_test.log"), mode="w"
    )
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_h)

    logging.getLogger("infra").setLevel(logging.INFO)

    # 第三方降噪
    for name in ("aiosqlite", "sqlalchemy", "asyncpg", "urllib3", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ============================================================
# 4. session 级：SQLite 临时库 bootstrap
#    注：不 autouse——只有声明了 db_session/engine/repo 的测试才会激活
# ============================================================
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def _sqlite_bootstrap() -> AsyncIterator[None]:
    """
    session 级一次性建/拆临时 SQLite DB：
    1) 创建临时 .db 文件 → 写入 config.yaml 指向该文件
    2) init_config + init_db_manager(require_db=True)
    3) yield（让所有 function-scoped 测试共享同一个 engine）
    4) TEARDOWN：drop_all → close_all_db_connections → unlink 临时文件

    用法：测试通过 `db_session` / `engine` / `repo` fixture 间接依赖本 fixture。
    """
    global _temp_db_file, _temp_db_path

    _temp_db_path = _make_temp_sqlite_path()
    test_url = f"sqlite+aiosqlite:///{_temp_db_path}"

    saved_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = test_url

    with tempfile.TemporaryDirectory(prefix="infra_test_cfg_") as cfg_dir:
        cfg_path = Path(cfg_dir)
        # 写最小 config.yaml：让 init_db_manager 走到 config['db'].url 分支
        (cfg_path / "config.yaml").write_text(
            "db:\n"
            "  url: " + test_url + "\n"
            "  engine:\n"
            "    pool_pre_ping: false\n"
            "    pool_recycle: -1\n"
            "    echo: false\n",
            encoding="utf-8",
        )

        from infra.config import init_config
        init_config(cfg_path)

        from infra.db import init_db_manager, close_all_db_connections
        await init_db_manager(require_db=True)

        try:
            yield
        finally:
            # drop_all（如果测试注册了 metadata；这里是 empty——实际表由各测试自己 create）
            try:
                from infra.db import get_business_db_manager
                mgr = get_business_db_manager()
                if mgr:
                    engine = mgr.get_engine()
                    if engine:
                        # 让用户注册的 metadata 也能 drop（如果有）
                        # 这里只清理测试在本次 session 期间建的所有表
                        from sqlalchemy import text
                        async with engine.begin() as conn:
                            rows = await conn.execute(text(
                                "SELECT name FROM sqlite_master WHERE type='table'"
                            ))
                            tables = [r[0] for r in rows.fetchall()]
                            for tbl in tables:
                                try:
                                    await conn.execute(text(f'DROP TABLE IF EXISTS "{tbl}"'))
                                except Exception:
                                    pass
            except Exception as e:  # pragma: no cover
                print(f"[conftest] ⚠️  drop tables 失败: {e!r}")

            await close_all_db_connections()
            if saved_db_url is None:
                os.environ.pop("DATABASE_URL", None)
            else:
                os.environ["DATABASE_URL"] = saved_db_url

    # 删临时文件
    if _temp_db_path and os.path.exists(_temp_db_path):
        try:
            os.unlink(_temp_db_path)
            print(f"[conftest] 🧹 临时数据库文件已删除: {_temp_db_path}")
        except Exception as e:
            print(f"[conftest] ⚠️ 删除临时数据库文件失败: {e}")

    _temp_db_file = None
    _temp_db_path = None


# ============================================================
# 5. function 级通用 fixture
# ============================================================
@pytest_asyncio.fixture
async def engine(_sqlite_bootstrap):
    """function-scoped engine——复用 session 级单例。"""
    from infra.db import get_business_db_manager
    return get_business_db_manager().get_engine()


@pytest_asyncio.fixture
async def db_session(_sqlite_bootstrap):
    """Function-scoped fixture: 提供一个干净的 DB session（每个测试一个新 session）。"""
    from infra.db import get_business_session

    async with get_business_session() as session:
        yield session


@pytest_asyncio.fixture
async def clean_db(_sqlite_bootstrap):
    """Function-scoped: 测试结束后清空所有表数据（保留 schema）。"""
    yield
    # TEARDOWN: 清空所有表数据
    from infra.db import get_business_db_manager
    from sqlalchemy import text
    mgr = get_business_db_manager()
    if not mgr:
        return
    eng = mgr.get_engine()
    if not eng:
        return
    async with eng.begin() as conn:
        rows = await conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        tables = [r[0] for r in rows.fetchall()]
        for tbl in tables:
            try:
                await conn.execute(text(f'DELETE FROM "{tbl}"'))
            except Exception as e:
                print(f"[conftest] ⚠️ 清空表 {tbl} 失败: {e}")


# ============================================================
# 6. Pytest 配置钩子
#    旧参考 conftest 用了 event_loop fixture + collection_modifyitems，
#    那是 pytest-asyncio <0.21 时代的写法，新版本（1.4+）已废弃 event_loop，
#    通过 pyproject.toml 的 asyncio_default_test_loop_scope=session 即可。
# ============================================================
