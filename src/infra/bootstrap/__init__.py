#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
应用启动引导模块
封装配置加载、日志、数据库、Redis 等初始化逻辑
"""

import os
from pathlib import Path
from typing import Optional

from infra.logger import get_logger, configure_logging_from_dict
from infra.redis import init_redis, close_redis
from infra.db.connection_manager import (
    init_db_manager,
    init_timeseries_db_manager,
    close_all_db_connections,
)
from infra.config import init_config

_logger = None


def get_bootstrap_logger():
    """获取引导模块的日志器"""
    global _logger
    if _logger is None:
        _logger = get_logger(__name__)
    return _logger


def _load_dotenv_once() -> Optional[str]:
    """
    自动从候选路径加载 .env（如果存在）。

    候选顺序：
    1. CWD/.env
    2. 仓库根/.env（infra.bootstrap 自身向上 3 级 = 仓库根）

    Returns:
        加载的 .env 绝对路径，无 .env 时返 None。

    副作用：把 .env 中的变量导入 os.environ（不覆盖已存在的环境变量）。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None  # python-dotenv 未装

    candidates = [
        Path.cwd() / ".env",  # 1. CWD
        Path(__file__).resolve().parent.parent.parent / ".env",  # 2. 仓库根
    ]
    for env_path in candidates:
        if env_path.exists() and env_path.is_file():
            load_dotenv(str(env_path), override=False)
            return str(env_path)
    return None


# 模块导入时自动加载一次（仅一次——避免重复 import 时重复加载）
_dotenv_loaded_path: Optional[str] = _load_dotenv_once()


def load_config(config_path: Optional[str] = None) -> dict:
    """
    加载配置（同步操作）

    Args:
        config_path: 配置文件目录路径（必填）——主项目用 bootstrap_all(config_dir)
    """
    if config_path is None:
        raise ValueError(
            "config_path is required——infra.bootstrap 不能猜主项目路径。"
            "主项目调用：from infra.bootstrap import bootstrap_all; "
            "await bootstrap_all(config_dir=Path('src/macro_monitor/configs'))"
        )

    config_dir = Path(config_path)

    config = init_config(config_dir)
    logger = get_bootstrap_logger()
    logger.info(
        f"配置加载完成，环境: {os.getenv('MACRO_MONITOR_ENV', 'dev')}"
        + (f"，.env: {_dotenv_loaded_path}" if _dotenv_loaded_path else "")
    )
    return config


def setup_logging(config: dict) -> None:
    """配置日志系统（同步操作）"""
    try:
        configure_logging_from_dict(config)
        logger = get_bootstrap_logger()
        logger.info("日志系统配置完成")
    except Exception as e:
        print(f"日志系统配置失败: {e}")
        raise


async def init_databases() -> None:
    """初始化所有数据库连接"""
    logger = get_bootstrap_logger()
    try:
        await init_db_manager()
        await init_timeseries_db_manager()
        logger.info("数据库连接管理器初始化完成")
    except Exception as e:
        logger.error(f"数据库连接管理器初始化失败: {e}")
        raise


async def init_redis_connection(config: dict) -> None:
    """初始化 Redis 连接"""
    logger = get_bootstrap_logger()
    try:
        await init_redis(config)
        logger.info("Redis 连接管理器初始化完成")
    except Exception as e:
        logger.error(f"Redis 连接管理器初始化失败: {e}")
        raise


async def bootstrap_all(config_path: Optional[str] = None) -> dict:
    """
    一次性初始化所有基础组件

    Args:
        config_path: 配置文件目录路径

    Returns:
        配置字典

    Note:
        启动时自动加载 .env（候选路径：CWD/.env → 仓库根/.env）——
        如未找到 .env 不报错（不强求）。
    """
    # 1. 加载配置（同步）—— .env 已在模块导入时自动加载
    config = load_config(config_path)

    # 2. 配置日志（同步）
    setup_logging(config)

    # 3. 初始化数据库（异步）
    await init_databases()

    # 4. 初始化 Redis（异步）
    await init_redis_connection(config)

    logger = get_bootstrap_logger()
    logger.info("所有基础组件初始化完成")

    return config


async def shutdown_bootstrap() -> None:
    """关闭所有连接（用于优雅关闭）"""
    logger = get_bootstrap_logger()
    logger.info("开始关闭所有基础组件...")

    # 关闭 Redis
    try:
        await close_redis()
        logger.info("Redis 连接已关闭")
    except Exception as e:
        logger.error(f"关闭 Redis 连接失败: {e}")

    # 关闭数据库连接
    try:
        await close_all_db_connections()
        logger.info("所有数据库连接已关闭")
    except Exception as e:
        logger.error(f"关闭数据库连接失败: {e}")

    logger.info("所有基础组件已关闭")
