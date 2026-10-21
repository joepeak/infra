#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的日志模块
只从配置文件加载，避免过度设计
修复点：文件Handler绑定日志等级、开启日志传播、支持DEBUG输出
"""

import os
import sys
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
_initialized_loggers: set = set()
# 每个 logger 建立时的配置签名 (level, log_dir)——配置变了要重建 handler
_logger_signatures: Dict[str, tuple] = {}


def configure_logging(config: LogConfig) -> None:
    """配置日志（依赖注入）"""
    global _log_config
    _log_config = config
    _ensure_log_dir()


def configure_logging_from_dict(config_dict: Dict[str, Any]) -> None:
    """从配置字典初始化日志"""
    logging_config = config_dict.get('logging', {})
    
    config = LogConfig(
        level=logging_config.get('level', DEFAULT_LOG_LEVEL),
        log_dir=logging_config.get('log_dir', DEFAULT_LOG_DIR),
        max_bytes=logging_config.get('max_bytes', DEFAULT_MAX_BYTES),
        backup_count=logging_config.get('backup_count', DEFAULT_BACKUP_COUNT)
    )
    
    configure_logging(config)


def _ensure_log_dir() -> None:
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


def reset_logging() -> None:
    """重置日志配置（主要用于测试）"""
    global _log_config, _initialized_loggers
    _log_config = None
    _initialized_loggers.clear()
    _logger_signatures.clear()


def _has_console_handler(logger: logging.Logger) -> bool:
    """判断 logger 是否已挂了「控制台」handler。

    注意：``RotatingFileHandler`` 是 ``StreamHandler`` 的子类，
    所以不能直接用 ``isinstance(h, logging.StreamHandler)`` 判断——
    那会把文件 handler 误判成控制台 handler，导致控制台输出永远装不上。
    这里显式排除 ``FileHandler``。
    """
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def _create_console_handler(config: LogConfig) -> logging.StreamHandler:
    """创建控制台 handler（stdout）"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    handler.setLevel(_get_log_level(config.level))
    return handler


class LazyLogger:
    """延迟加载logger代理，每次打印实时获取最新配置的logger实例"""

    def __init__(self, name: str, *, console: bool = False) -> None:
        self._name = name
        #: True 时额外挂 stdout handler——适合独立脚本/CLI 在终端观察流程
        self._console = console

    def _get_real_logger(self) -> logging.Logger:
        # 每次输出日志都重新拿当前最新配置的原生logger
        lg = _raw_get_logger(self._name)
        if self._console and not _has_console_handler(lg):
            # 幂等：已有控制台 handler 就不重复挂
            lg.addHandler(_create_console_handler(_get_log_config()))
        return lg

    # 代理全部日志方法
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_real_logger().debug(msg, *args, **kwargs)
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_real_logger().info(msg, *args, **kwargs)
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_real_logger().warning(msg, *args, **kwargs)
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_real_logger().error(msg, *args, **kwargs)
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._get_real_logger().critical(msg, *args, **kwargs)
    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """记录 ERROR 级别 + 异常堆栈（等价于原生 logger.exception）。"""
        self._get_real_logger().exception(msg, *args, **kwargs)
    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """按指定级别记录（等价于原生 logger.log）。"""
        self._get_real_logger().log(level, msg, *args, **kwargs)
    def isEnabledFor(self, level: int) -> bool:
        """该级别是否会输出（等价于原生 logger.isEnabledFor）。"""
        return self._get_real_logger().isEnabledFor(level)
    def setLevel(self, level: int) -> None:
        """设置该 logger 的级别（等价于原生 logger.setLevel）。"""
        self._get_real_logger().setLevel(level)

# 把原来创建logger的逻辑改名成内部原生函数
def _raw_get_logger(name: str) -> logging.Logger:
    global _initialized_loggers
    config = _get_log_config()
    target_level = _get_log_level(config.level)
    # 配置签名——level 或 log_dir 变了都要重建 handler，
    # 否则改完 log_dir 旧 logger 仍写老目录（测试间日志互相污染）
    signature = (target_level, config.log_dir)

    # 存在缓存，但配置签名不匹配 → 销毁旧实例重建
    if name in _initialized_loggers:
        old_logger = logging.getLogger(name)
        if _logger_signatures.get(name) != signature:
            old_logger.handlers.clear()
            _initialized_loggers.discard(name)
            _logger_signatures.pop(name, None)
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
    _logger_signatures[name] = signature
    return logger

# 对外暴露的函数名不变，依旧叫 get_logger，返回延迟代理
def get_logger(name: str, *, console: bool = False) -> LazyLogger:
    """获取 logger（返回延迟代理，每次输出取最新配置）。

    Args:
        name: logger 名称（通常是 ``__name__``）
        console: True 时额外挂 stdout handler——适合独立脚本 / CLI 工具
            在终端观察程序流程；服务端程序保持默认 False（避免 stdout 噪音）。
    """
    return LazyLogger(name, console=console)
