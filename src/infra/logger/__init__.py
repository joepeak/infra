#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的日志模块
只从配置文件加载，避免过度设计
修复点：文件Handler绑定日志等级、开启日志传播、支持DEBUG输出
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# 默认配置（硬编码，避免依赖外层包）
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50MB
DEFAULT_BACKUP_COUNT = 5


class LogConfig:
    """日志配置类"""

    def __init__(self,
                 level: str = DEFAULT_LOG_LEVEL,
                 log_dir: str = DEFAULT_LOG_DIR,
                 max_bytes: int = DEFAULT_MAX_BYTES,
                 backup_count: int = DEFAULT_BACKUP_COUNT):
        self.level = level
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count


# 全局配置
_log_config: Optional[LogConfig] = None
_initialized_loggers = set()


def configure_logging(config: LogConfig):
    """配置日志（依赖注入）"""
    global _log_config
    _log_config = config
    _ensure_log_dir()


def configure_logging_from_dict(config_dict: Dict[str, Any]):
    """从配置字典初始化日志"""
    logging_config = config_dict.get('logging', {})
    
    config = LogConfig(
        level=logging_config.get('level', DEFAULT_LOG_LEVEL),
        log_dir=logging_config.get('log_dir', DEFAULT_LOG_DIR),
        max_bytes=logging_config.get('max_bytes', DEFAULT_MAX_BYTES),
        backup_count=logging_config.get('backup_count', DEFAULT_BACKUP_COUNT)
    )
    
    configure_logging(config)


def _ensure_log_dir():
    """确保日志目录存在"""
    if _log_config:
        os.makedirs(_log_config.log_dir, exist_ok=True)


def _get_log_config() -> LogConfig:
    """获取日志配置"""
    global _log_config
    if _log_config is None:
        _log_config = LogConfig()  # 使用默认配置
        _ensure_log_dir()
    return _log_config


def _get_log_level(level_str: str) -> int:
    """将字符串日志级别转换为 logging 常量"""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_str.upper(), logging.INFO)


def _create_file_handler(log_file: str, config: LogConfig) -> RotatingFileHandler:
    """创建文件处理器，绑定配置日志等级"""
    handler = RotatingFileHandler(
        log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    # 核心修复：文件Handler同步设置日志等级，放行DEBUG
    handler.setLevel(_get_log_level(config.level))
    return handler


# def _raw_get_logger(name: str) -> logging.Logger:
#     global _initialized_loggers
#     config = _get_log_config()
#     target_level = _get_log_level(config.level)

#     # 存在缓存，但等级和当前配置不匹配，销毁旧实例
#     if name in _initialized_loggers:
#         old_logger = logging.getLogger(name)
#         if old_logger.level != target_level:
#             old_logger.handlers.clear()
#             _initialized_loggers.remove(name)
#         else:
#             return old_logger

#     # 全新创建logger，等级、handler同步当前最新配置
#     logger = logging.getLogger(name)
#     logger.setLevel(target_level)
#     logger.handlers.clear()

#     log_file = os.path.join(config.log_dir, f'{name.replace(".", "_")}.log')
#     file_handler = _create_file_handler(log_file, config)
#     logger.addHandler(file_handler)
#     logger.propagate = True

#     _initialized_loggers.add(name)
#     return logger


def reset_logging():
    """重置日志配置（主要用于测试）"""
    global _log_config, _initialized_loggers
    _log_config = None
    _initialized_loggers.clear()

class LazyLogger:
    """延迟加载logger代理，每次打印实时获取最新配置的logger实例"""
    def __init__(self, name: str):
        self._name = name

    def _get_real_logger(self):
        # 每次输出日志都重新拿当前最新配置的原生logger
        return _raw_get_logger(self._name)

    # 代理全部日志方法
    def debug(self, msg, *args, **kwargs):
        self._get_real_logger().debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs):
        self._get_real_logger().info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):
        self._get_real_logger().warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        self._get_real_logger().error(msg, *args, **kwargs)
    def critical(self, msg, *args, **kwargs):
        self._get_real_logger().critical(msg, *args, **kwargs)

# 把原来创建logger的逻辑改名成内部原生函数
def _raw_get_logger(name: str) -> logging.Logger:
    global _initialized_loggers
    config = _get_log_config()
    target_level = _get_log_level(config.level)

    # 存在缓存，但等级和当前配置不匹配，销毁旧实例
    if name in _initialized_loggers:
        old_logger = logging.getLogger(name)
        if old_logger.level != target_level:
            old_logger.handlers.clear()
            _initialized_loggers.remove(name)
        else:
            return old_logger

    # 全新创建logger，等级、handler同步当前最新配置
    logger = logging.getLogger(name)
    logger.setLevel(target_level)
    logger.handlers.clear()

    log_file = os.path.join(config.log_dir, f'{name.replace(".", "_")}.log')
    file_handler = _create_file_handler(log_file, config)
    logger.addHandler(file_handler)
    logger.propagate = True

    _initialized_loggers.add(name)
    return logger

# 对外暴露的函数名不变，依旧叫 get_logger，返回延迟代理
def get_logger(name: str) -> LazyLogger:
    return LazyLogger(name)