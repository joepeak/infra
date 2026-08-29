#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
应用启动引导模块
封装配置、日志、数据库、Redis 等初始化逻辑
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


def load_config(config_path: Optional[str] = None) -> dict:
    """
    加载配置（同步操作）

    Args:
        config_path: 配置文件目录路径（必填）——主项目用 bootstrap_all(config_dir)

    Returns:
        配置字典
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
    logger.info(f"配置加载完成，环境: {os.getenv('MACRO_MONITOR_ENV', 'dev')}")
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
    """
    # 1. 加载配置（同步）
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