#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本通用日志工具
为独立脚本（scripts/、cron、CLI 工具）提供控制台 + 文件双输出日志——自用类库 infra 的一部分
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# 默认日志目录（相对于项目根）
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50MB
DEFAULT_BACKUP_COUNT = 5


def _ensure_log_dir(log_dir: str) -> None:
    """确保日志目录存在"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)


def get_script_logger(name: str,
                      level: str = DEFAULT_LOG_LEVEL,
                      log_dir: str = DEFAULT_LOG_DIR,
                      console: bool = True) -> logging.Logger:
    """
    获取脚本专用的日志器（控制台 + 文件双输出）

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录
        console: 是否同时输出到控制台

    Returns:
        配置好的 logging.Logger 实例
    """
    # 强制用字符串名，避免 __name__ 在脚本中变成 "__main__"
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 清空已有 handlers（防止重复添加）
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(console_handler)

    # 文件输出
    _ensure_log_dir(log_dir)
    log_file = str(Path(log_dir) / f"{name.replace('.', '_')}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(file_handler)

    return logger