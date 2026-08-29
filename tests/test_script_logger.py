#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.script_logger.get_script_logger 测试

覆盖：
1. 默认参数：level=INFO, log_dir=logs, console=True
2. 自定义参数
3. 返回的是真实 logging.Logger（非 LazyLogger）
4. 重复调用同名 logger → 返回**同一个**（已有 handlers 时直接 return，不重加）
5. 写入消息到文件
6. 控制台 handler 输出到 stdout
7. 关闭 console（只文件输出）
8. log 文件名：'.' 替换为 '_'
9. 日志级别大小写
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from infra.script_logger import (
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_BYTES,
    DEFAULT_BACKUP_COUNT,
    get_script_logger,
)


# ============================================================
# Fixture：每个测试隔离 logger（避免 handlers 累积）
# ============================================================
@pytest.fixture(autouse=True)
def _isolate_loggers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """测试间不互相影响：保存所有 logger 状态 + 切到 tmp_path。"""
    monkeypatch.chdir(tmp_path)

    # 清理可能存在的测试 logger
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("script_test_"):
            lg = logging.getLogger(name)
            lg.handlers.clear()
            logging.Logger.manager.loggerDict.pop(name, None)

    yield

    # TEARDOWN
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("script_test_"):
            lg = logging.getLogger(name)
            lg.handlers.clear()
            logging.Logger.manager.loggerDict.pop(name, None)


# ============================================================
# 1. 返回类型
# ============================================================
class TestReturnType:
    def test_returns_real_logger(self, tmp_path):
        lg = get_script_logger(
            "script_test_type",
            level="INFO",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        # 真实 Logger（不是 LazyLogger）
        assert isinstance(lg, logging.Logger)
        # 不是 LazyLogger
        from infra.logger import LazyLogger
        assert not isinstance(lg, LazyLogger)

    def test_logger_name(self, tmp_path):
        lg = get_script_logger(
            "script_test_name",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        assert lg.name == "script_test_name"


# ============================================================
# 2. 默认参数
# ============================================================
class TestDefaults:
    def test_default_log_dir(self, tmp_path, monkeypatch):
        # 切到 tmp_path 让默认 "logs" 目录在 tmp 下创建
        monkeypatch.chdir(tmp_path)
        lg = get_script_logger("script_test_default", console=False)
        assert (tmp_path / "logs" / "script_test_default.log").exists()

    def test_default_level_info(self, tmp_path):
        lg = get_script_logger(
            "script_test_lvl",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        assert lg.level == logging.INFO

    def test_default_max_bytes_constant(self):
        assert DEFAULT_MAX_BYTES == 50 * 1024 * 1024
        assert DEFAULT_BACKUP_COUNT == 5
        assert DEFAULT_LOG_DIR == "logs"
        assert DEFAULT_LOG_LEVEL == "INFO"


# ============================================================
# 3. 自定义 level / log_dir / console
# ============================================================
class TestCustomArgs:
    def test_custom_level_debug(self, tmp_path):
        lg = get_script_logger(
            "script_test_debug",
            level="DEBUG",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        assert lg.level == logging.DEBUG

    def test_custom_log_dir(self, tmp_path):
        custom = tmp_path / "custom_logs"
        lg = get_script_logger(
            "script_test_dir",
            level="INFO",
            log_dir=str(custom),
            console=False,
        )
        assert (custom / "script_test_dir.log").exists()

    def test_console_false_no_streamhandler(self, tmp_path):
        lg = get_script_logger(
            "script_test_noconsole",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        # 应该有 1 个 file handler，无 StreamHandler
        assert len(lg.handlers) == 1
        from logging.handlers import RotatingFileHandler
        assert isinstance(lg.handlers[0], RotatingFileHandler)

    def test_console_true_has_streamhandler(self, tmp_path):
        lg = get_script_logger(
            "script_test_console",
            log_dir=str(tmp_path / "logs"),
            console=True,
        )
        # 2 个 handlers: StreamHandler + RotatingFileHandler
        assert len(lg.handlers) == 2
        types = {type(h).__name__ for h in lg.handlers}
        assert "StreamHandler" in types
        assert "RotatingFileHandler" in types


# ============================================================
# 4. 重复调用返回同一个（防 handlers 累积）
# ============================================================
class TestIdempotent:
    def test_second_call_returns_same_logger(self, tmp_path):
        lg1 = get_script_logger(
            "script_test_idem",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        lg2 = get_script_logger(
            "script_test_idem",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        assert lg1 is lg2

    def test_second_call_does_not_add_handlers(self, tmp_path):
        lg1 = get_script_logger(
            "script_test_idem2",
            log_dir=str(tmp_path / "logs"),
            console=True,  # 第一次会加 2 个
        )
        n1 = len(lg1.handlers)
        # 第二次调用（即使参数不同）
        lg2 = get_script_logger(
            "script_test_idem2",
            level="DEBUG",  # 想改成 DEBUG
            log_dir=str(tmp_path / "logs"),
            console=True,
        )
        # 仍然是同一个 logger，handlers 数量不变
        assert lg1 is lg2
        assert len(lg2.handlers) == n1  # 没新加
        # level 在短路 return 之前已 setLevel（源码行为）
        assert lg2.level == logging.DEBUG


# ============================================================
# 5. 写入消息
# ============================================================
class TestWriteMessages:
    def test_info_writes_to_file(self, tmp_path, caplog):
        log_dir = tmp_path / "logs"
        with caplog.at_level(logging.INFO, logger="script_test_write"):
            lg = get_script_logger(
                "script_test_write",
                log_dir=str(log_dir),
                console=False,
            )
            lg.info("hello info")

        log_file = log_dir / "script_test_write.log"
        content = log_file.read_text(encoding="utf-8")
        assert "hello info" in content
        assert "INFO" in content

    def test_error_writes_to_file(self, tmp_path):
        log_dir = tmp_path / "logs"
        lg = get_script_logger(
            "script_test_err",
            log_dir=str(log_dir),
            console=False,
        )
        lg.error("boom")
        content = (log_dir / "script_test_err.log").read_text()
        assert "boom" in content
        assert "ERROR" in content


# ============================================================
# 6. 日志级别大小写
# ============================================================
class TestLevelCase:
    def test_lowercase_level(self, tmp_path):
        lg = get_script_logger(
            "script_test_lvlcase",
            level="warning",  # 小写
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        assert lg.level == logging.WARNING

    def test_unknown_level_falls_back_to_info(self, tmp_path):
        """unknown level → getattr(logging, 'NONEXISTENT', logging.INFO)。"""
        lg = get_script_logger(
            "script_test_unknown_lvl",
            level="BOGUS",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        assert lg.level == logging.INFO


# ============================================================
# 7. log 文件名：'.' 替换为 '_'
# ============================================================
class TestLogFileName:
    def test_dotted_name_uses_underscores(self, tmp_path):
        get_script_logger(
            "script_test.dotted.name",
            log_dir=str(tmp_path / "logs"),
            console=False,
        )
        # 期望: script_test_dotted_name.log
        assert (tmp_path / "logs" / "script_test_dotted_name.log").exists()


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
