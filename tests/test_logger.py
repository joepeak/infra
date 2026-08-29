#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.logger 测试

覆盖：
1. LogConfig 默认值 / 自定义
2. configure_logging + _ensure_log_dir 创建目录
3. configure_logging_from_dict 解析 {'logging': {...}} 字典
4. reset_logging 还原状态
5. get_logger 返回 LazyLogger
6. LazyLogger 代理 debug/info/warning/error/critical
7. _get_log_level 解析 'DEBUG'/'INFO'/'WARNING'/'ERROR'/'CRITICAL'/大小写/未知
8. _create_file_handler 设置正确等级 + formatter
9. _raw_get_logger 缓存 + 等级变更时重建
10. log_file 路径生成（name.replace('.', '_')）
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import infra.logger as logger_module
from infra.logger import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_BYTES,
    DEFAULT_BACKUP_COUNT,
    LogConfig,
    configure_logging,
    configure_logging_from_dict,
    get_logger,
    reset_logging,
    LazyLogger,
    _get_log_level,
    _create_file_handler,
    _raw_get_logger,
)


# ============================================================
# Fixture
# ============================================================
@pytest.fixture(autouse=True)
def _reset_logger_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """每个测试：还原全局 + 切到 tmp_path（避免污染 cwd）。"""
    monkeypatch.chdir(tmp_path)
    reset_logging()
    yield
    reset_logging()


# ============================================================
# 1. LogConfig
# ============================================================
class TestLogConfig:
    def test_default_values(self):
        c = LogConfig()
        assert c.level == "INFO"
        assert c.log_dir == "logs"
        assert c.max_bytes == 50 * 1024 * 1024
        assert c.backup_count == 5

    def test_custom_values(self):
        c = LogConfig(level="DEBUG", log_dir="/tmp/x", max_bytes=1024, backup_count=3)
        assert c.level == "DEBUG"
        assert c.log_dir == "/tmp/x"
        assert c.max_bytes == 1024
        assert c.backup_count == 3

    def test_module_constants_match_defaults(self):
        c = LogConfig()
        assert c.level == DEFAULT_LOG_LEVEL
        assert c.log_dir == DEFAULT_LOG_DIR
        assert c.max_bytes == DEFAULT_MAX_BYTES
        assert c.backup_count == DEFAULT_BACKUP_COUNT


# ============================================================
# 2. configure_logging
# ============================================================
class TestConfigureLogging:
    def test_creates_log_dir(self, tmp_path):
        target = tmp_path / "subdir" / "logs"
        assert not target.exists()
        configure_logging(LogConfig(log_dir=str(target)))
        assert target.exists()
        assert target.is_dir()

    def test_existing_log_dir_ok(self, tmp_path):
        """已存在的目录不应报错。"""
        target = tmp_path / "logs"
        target.mkdir()
        configure_logging(LogConfig(log_dir=str(target)))
        assert target.exists()

    def test_config_stored(self):
        c = LogConfig(level="DEBUG", log_dir="/tmp/x")
        configure_logging(c)
        assert logger_module._log_config is c


# ============================================================
# 3. configure_logging_from_dict
# ============================================================
class TestConfigureLoggingFromDict:
    def test_full_dict(self):
        configure_logging_from_dict({
            "logging": {
                "level": "DEBUG",
                "log_dir": "/tmp/dict",
                "max_bytes": 2048,
                "backup_count": 2,
            }
        })
        c = logger_module._log_config
        assert c.level == "DEBUG"
        assert c.log_dir == "/tmp/dict"
        assert c.max_bytes == 2048
        assert c.backup_count == 2

    def test_empty_dict_uses_defaults(self, tmp_path):
        configure_logging_from_dict({})
        c = logger_module._log_config
        # level 缺失 → INFO；log_dir 缺失 → "logs"（但 _ensure_log_dir 用它建目录）
        assert c.level == DEFAULT_LOG_LEVEL
        assert c.log_dir == DEFAULT_LOG_DIR

    def test_dict_without_logging_key(self):
        configure_logging_from_dict({"other_key": 1})
        c = logger_module._log_config
        # 没有 'logging' 键 → 走默认
        assert c.level == DEFAULT_LOG_LEVEL

    def test_partial_dict_overrides_only_given(self):
        configure_logging_from_dict({"logging": {"level": "ERROR"}})
        c = logger_module._log_config
        assert c.level == "ERROR"
        # 其他用默认
        assert c.log_dir == DEFAULT_LOG_DIR


# ============================================================
# 4. reset_logging
# ============================================================
class TestResetLogging:
    def test_reset_clears_config(self):
        configure_logging(LogConfig(level="DEBUG"))
        assert logger_module._log_config is not None
        reset_logging()
        assert logger_module._log_config is None

    def test_reset_clears_initialized_loggers(self):
        configure_logging(LogConfig(log_dir="/tmp/reset_test"))
        _raw_get_logger("infra.x")
        assert "infra.x" in logger_module._initialized_loggers
        reset_logging()
        assert logger_module._initialized_loggers == set()


# ============================================================
# 5. get_logger + LazyLogger
# ============================================================
class TestGetLogger:
    def test_returns_lazy_logger_instance(self):
        lg = get_logger("infra.foo")
        assert isinstance(lg, LazyLogger)

    def test_lazy_logger_stores_name(self):
        lg = get_logger("infra.foo")
        assert lg._name == "infra.foo"

    def test_same_name_returns_new_proxy_each_call(self):
        """每次 get_logger 返回新的 LazyLogger（不是单例），但行为一致。"""
        a = get_logger("infra.x")
        b = get_logger("infra.x")
        # 不同实例（每次都 new）
        assert a is not b
        # 但 _name 相同
        assert a._name == b._name


# ============================================================
# 6. LazyLogger 代理方法
# ============================================================
class TestLazyLoggerProxy:
    @pytest.fixture
    def configured_logger(self, tmp_path):
        """配置日志到 tmp_path，logger 输出到文件。"""
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))
        return get_logger("infra.proxy_test")

    def test_debug_writes_to_file(self, configured_logger, tmp_path):
        configured_logger.debug("hello debug")
        log_file = tmp_path / "logs" / "infra_proxy_test.log"
        content = log_file.read_text(encoding="utf-8")
        assert "hello debug" in content
        assert "DEBUG" in content

    def test_info_writes_to_file(self, configured_logger, tmp_path):
        configured_logger.info("hello info")
        log_file = tmp_path / "logs" / "infra_proxy_test.log"
        content = log_file.read_text(encoding="utf-8")
        assert "hello info" in content
        assert "INFO" in content

    def test_warning_writes(self, configured_logger, tmp_path):
        configured_logger.warning("hi warn")
        content = (tmp_path / "logs" / "infra_proxy_test.log").read_text()
        assert "hi warn" in content
        assert "WARNING" in content

    def test_error_writes(self, configured_logger, tmp_path):
        configured_logger.error("oops")
        content = (tmp_path / "logs" / "infra_proxy_test.log").read_text()
        assert "oops" in content
        assert "ERROR" in content

    def test_critical_writes(self, configured_logger, tmp_path):
        configured_logger.critical("fatal")
        content = (tmp_path / "logs" / "infra_proxy_test.log").read_text()
        assert "fatal" in content
        assert "CRITICAL" in content

    def test_format_includes_module_name(self, configured_logger, tmp_path):
        configured_logger.info("formatted")
        content = (tmp_path / "logs" / "infra_proxy_test.log").read_text()
        assert "infra.proxy_test" in content


# ============================================================
# 7. _get_log_level
# ============================================================
class TestGetLogLevel:
    @pytest.mark.parametrize("level_str,expected", [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ])
    def test_known_levels(self, level_str, expected):
        assert _get_log_level(level_str) == expected

    @pytest.mark.parametrize("level_str,expected", [
        ("debug", logging.DEBUG),
        ("Info", logging.INFO),
        ("warning", logging.WARNING),
    ])
    def test_case_insensitive(self, level_str, expected):
        assert _get_log_level(level_str) == expected

    def test_unknown_level_falls_back_to_info(self):
        assert _get_log_level("NONEXISTENT") == logging.INFO


# ============================================================
# 8. _create_file_handler
# ============================================================
class TestCreateFileHandler:
    def test_handler_level_matches_config(self):
        c = LogConfig(level="WARNING")
        h = _create_file_handler("/tmp/test_handler.log", c)
        try:
            assert h.level == logging.WARNING
        finally:
            h.close()

    def test_handler_max_bytes(self):
        c = LogConfig(max_bytes=2048, backup_count=2)
        h = _create_file_handler("/tmp/test_handler2.log", c)
        try:
            assert h.maxBytes == 2048
            assert h.backupCount == 2
        finally:
            h.close()

    def test_handler_has_formatter(self):
        c = LogConfig()
        h = _create_file_handler("/tmp/test_handler3.log", c)
        try:
            assert h.formatter is not None
            # formatter 模板
            assert "%(name)s" in h.formatter._fmt
            assert "%(levelname)s" in h.formatter._fmt
        finally:
            h.close()


# ============================================================
# 9. _raw_get_logger 缓存 + 重建
# ============================================================
class TestRawGetLogger:
    def test_creates_file_handler_for_new_logger(self, tmp_path):
        configure_logging(LogConfig(level="INFO", log_dir=str(tmp_path / "logs")))
        lg = _raw_get_logger("infra.new")
        try:
            assert len(lg.handlers) == 1
            assert lg.level == logging.INFO
            assert lg.propagate is True
        finally:
            lg.handlers.clear()
            logging.getLogger("infra.new").handlers.clear()

    def test_returns_cached_instance(self, tmp_path):
        configure_logging(LogConfig(log_dir=str(tmp_path / "logs")))
        a = _raw_get_logger("infra.cached")
        b = _raw_get_logger("infra.cached")
        assert a is b

    def test_recreates_when_level_changes(self, tmp_path):
        """等级不匹配时清空 handlers + 重建。"""
        configure_logging(LogConfig(level="INFO", log_dir=str(tmp_path / "logs")))
        lg = _raw_get_logger("infra.level_chg")
        original_handler = lg.handlers[0] if lg.handlers else None
        assert original_handler is not None

        # 改 level 到 DEBUG
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))
        lg2 = _raw_get_logger("infra.level_chg")
        # 新 handler，level=DEBUG
        assert lg2.level == logging.DEBUG
        assert lg2.handlers[0] is not original_handler

    def test_log_file_path_uses_underscores(self, tmp_path):
        """'.' 替换为 '_' 作为文件名。"""
        configure_logging(LogConfig(log_dir=str(tmp_path / "logs")))
        _raw_get_logger("infra.sub.module")
        expected = tmp_path / "logs" / "infra_sub_module.log"
        assert expected.exists()


# ============================================================
# 10. 配合 script_logger 风格测试
# ============================================================
class TestGetLoggerTwice:
    def test_get_logger_twice_same_name(self, tmp_path):
        """两次 get_logger(name) 都返回新 LazyLogger，但 _raw_get_logger 缓存底层。"""
        configure_logging(LogConfig(log_dir=str(tmp_path / "logs")))
        a = get_logger("infra.foo")
        b = get_logger("infra.foo")
        # LazyLogger 不同实例
        assert a is not b
        # 但内部 raw logger 共享缓存
        assert a._get_real_logger() is b._get_real_logger()


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
