#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.llm 日志落盘回归测试（20261021 新增）

背景：``infra/llm/`` 下的 client / usage / capabilities 三个模块原先用
原生 ``logging.getLogger(__name__)``——原生 logger 自身没有 handler，
root logger 也没有，于是 LLM 层的所有日志（能力探测、重试、错误、耗时）
**根本没落进 infra 的日志文件**（只在 root 有 handler 时才打到 stderr）。

LLM 恰恰是最需要留痕的模块，所以这里锁定「LLM 日志必须进文件」这个契约。

覆盖：
1. 三个 llm 模块的 logger 都是 infra 的 LazyLogger
2. LLM 层日志实际写进 infra 的日志文件
3. 重试日志（含 model / attempt / 异常类型）也在文件里
"""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path

import pytest

from infra.logger import LogConfig, configure_logging, get_logger, reset_logging
import infra.llm.capabilities as capabilities_module
import infra.llm.client as client_module
import infra.llm.usage as usage_module
from infra.logger import LazyLogger
from infra.utils.retry import RetryConfig, retry_async


@pytest.fixture(autouse=True)
def _reset_logger_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """日志状态是全局的——每个测试后还原。"""
    monkeypatch.chdir(tmp_path)
    reset_logging()
    yield
    reset_logging()


# ============================================================
# 1. llm 模块用 infra 的 LazyLogger
# ============================================================
class TestLlmModulesUseInfraLogger:
    def test_client_uses_infra_logger(self):
        assert isinstance(client_module.logger, LazyLogger)

    def test_usage_uses_infra_logger(self):
        assert isinstance(usage_module.logger, LazyLogger)

    def test_capabilities_uses_infra_logger(self):
        assert isinstance(capabilities_module.logger, LazyLogger)


# ============================================================
# 2. LLM 层日志落文件
# ============================================================
class TestLlmLogsReachFile:
    def test_client_logger_writes_to_file(self, tmp_path):
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))
        client_module.logger.info("LLM 客户端这条日志应落文件")
        log_file = tmp_path / "logs" / "infra_llm_client.log"
        assert log_file.exists()
        assert "LLM 客户端这条日志应落文件" in log_file.read_text(encoding="utf-8")

    def test_capabilities_logger_writes_to_file(self, tmp_path):
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))
        capabilities_module.logger.info("能力注册表这条日志应落文件")
        log_file = tmp_path / "logs" / "infra_llm_capabilities.log"
        assert log_file.exists()
        assert "能力注册表这条日志应落文件" in log_file.read_text(encoding="utf-8")

    def test_usage_logger_writes_to_file(self, tmp_path):
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))
        usage_module.logger.info("用量统计这条日志应落文件")
        log_file = tmp_path / "logs" / "infra_llm_usage.log"
        assert log_file.exists()
        assert "用量统计这条日志应落文件" in log_file.read_text(encoding="utf-8")


# ============================================================
# 3. LLM 重试日志落文件 + 不泄露 prompt
# ============================================================
class TestLlmRetryLogReachesFile:
    def test_retry_log_in_file_with_model(self, tmp_path):
        """LLM 场景的重试日志（含 model）应落到文件里。"""
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))

        async def create(**kwargs):
            raise ConnectionError("gateway timeout")

        p = partial(
            create,
            model="gpt-4o",
            messages=[{"role": "user", "content": "私密提示词 SECRET_LLM"}],
        )
        cfg = RetryConfig.default().copy(max_retries=1, base_delay=0.01, jitter=False)

        with pytest.raises(ConnectionError):
            asyncio.run(retry_async(p, config=cfg))

        content = (tmp_path / "logs" / "infra_utils_retry.log").read_text(encoding="utf-8")
        assert "model='gpt-4o'" in content
        assert "attempt=1/2" in content
        assert "ConnectionError" in content
        # prompt 不能泄露
        assert "SECRET_LLM" not in content

    def test_llm_logger_and_retry_logger_both_in_files(self, tmp_path):
        """LLM 模块日志与 retry 日志落在各自的文件里。"""
        configure_logging(LogConfig(level="DEBUG", log_dir=str(tmp_path / "logs")))

        async def create(**kwargs):
            raise ConnectionError("boom")

        p = partial(create, model="deepseek-chat")
        cfg = RetryConfig.default().copy(max_retries=1, base_delay=0.01, jitter=False)
        with pytest.raises(ConnectionError):
            asyncio.run(retry_async(p, config=cfg))

        client_module.logger.info("客户端侧日志")
        logs_dir = tmp_path / "logs"
        assert (logs_dir / "infra_llm_client.log").exists()
        assert (logs_dir / "infra_utils_retry.log").exists()


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
