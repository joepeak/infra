#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db.monitor DatabaseMonitor + 全局单例测试

覆盖：
1. DatabaseMetrics dataclass 字段
2. DatabaseMonitor 初始状态
3. record_query 累加 query_count / error_count / response_times
4. response_times 自动截断到 100
5. add_alert_callback 添加回调
6. get_current_metrics：空/非空
7. get_metrics_summary：空 / 有数据
8. _check_alerts 各分支（pool 高使用率 / 溢出 / 无效连接 / 错误率高 / 响应慢）
9. 告警回调执行；回调异常被吞掉
10. start_monitoring / stop_monitoring：幂等
11. 便捷函数：start/stop/get_metrics/get_health_status/record_db_query/add_db_alert_callback
12. 模块级 db_monitor 单例

注意：DatabaseMonitor._collect_metrics 调 get_business_db_manager()，
涉及数据库 manager 初始化。测试用 mock 绕开。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from infra.db.monitor import (
    DatabaseMetrics,
    DatabaseMonitor,
    db_monitor,
    start_db_monitoring,
    stop_db_monitoring,
    get_db_metrics,
    get_db_health_status,
    record_db_query,
    add_db_alert_callback,
)
from infra.db import monitor as monitor_module


# ============================================================
# Fixture
# ============================================================
@pytest.fixture
def fresh_monitor():
    """每个测试用全新的 DatabaseMonitor 实例。"""
    return DatabaseMonitor()


# ============================================================
# 1. DatabaseMetrics
# ============================================================
class TestDatabaseMetrics:
    def test_dataclass_fields(self):
        m = DatabaseMetrics(
            timestamp=1.0,
            connection_pool_size=10,
            connections_checked_in=2,
            connections_checked_out=3,
            connection_pool_overflow=0,
            connection_pool_invalid=0,
            query_count=5,
            error_count=1,
            avg_response_time=0.5,
            status="healthy",
        )
        assert m.timestamp == 1.0
        assert m.connection_pool_size == 10
        assert m.status == "healthy"


# ============================================================
# 2. 初始状态
# ============================================================
class TestInitialState:
    def test_initial_state(self, fresh_monitor):
        assert fresh_monitor.is_running is False
        assert fresh_monitor.monitor_thread is None
        assert fresh_monitor.metrics_history == []
        assert fresh_monitor.query_count == 0
        assert fresh_monitor.error_count == 0
        assert fresh_monitor.response_times == []
        assert fresh_monitor.alert_callbacks == []
        assert fresh_monitor.max_history_size == 1000
        assert fresh_monitor.check_interval == 60  # 默认

    def test_custom_check_interval(self):
        m = DatabaseMonitor(check_interval=10)
        assert m.check_interval == 10


# ============================================================
# 3. record_query
# ============================================================
class TestRecordQuery:
    def test_record_success(self, fresh_monitor):
        fresh_monitor.record_query(0.1, success=True)
        assert fresh_monitor.query_count == 1
        assert fresh_monitor.error_count == 0
        assert fresh_monitor.response_times == [0.1]

    def test_record_failure(self, fresh_monitor):
        fresh_monitor.record_query(0.2, success=False)
        assert fresh_monitor.query_count == 1
        assert fresh_monitor.error_count == 1

    def test_record_multiple(self, fresh_monitor):
        for i in range(5):
            fresh_monitor.record_query(0.01 * i, success=True)
        assert fresh_monitor.query_count == 5
        assert fresh_monitor.error_count == 0
        assert fresh_monitor.response_times == [0.0, 0.01, 0.02, 0.03, 0.04]


# ============================================================
# 4. response_times 自动截断
# ============================================================
class TestResponseTimesTruncation:
    def test_truncates_at_100(self, fresh_monitor):
        # 直接 push 105 条（绕过 _collect_metrics 里的截断）
        for i in range(105):
            fresh_monitor.response_times.append(float(i))
        # 调一次 _collect_metrics 触发现有截断逻辑
        with patch("infra.db.monitor.get_db_manager_or_raise") as mock_get:
            mock_mgr = MagicMock()
            mock_mgr.get_connection_stats.return_value = {
                "pool_size": 10,
                "checked_in": 0,
                "checked_out": 0,
                "overflow": 0,
                "invalid": 0,
            }
            mock_get.return_value = mock_mgr
            fresh_monitor._collect_metrics()
        # 截断后保留最后 100 条（[5..104]）
        assert len(fresh_monitor.response_times) == 100
        assert fresh_monitor.response_times[0] == 5
        assert fresh_monitor.response_times[-1] == 104


# ============================================================
# 5. add_alert_callback
# ============================================================
class TestAlertCallback:
    def test_add_callback(self, fresh_monitor):
        cb = lambda msg, m: None
        fresh_monitor.add_alert_callback(cb)
        assert cb in fresh_monitor.alert_callbacks

    def test_multiple_callbacks(self, fresh_monitor):
        cb1 = lambda msg, m: None
        cb2 = lambda msg, m: None
        fresh_monitor.add_alert_callback(cb1)
        fresh_monitor.add_alert_callback(cb2)
        assert fresh_monitor.alert_callbacks == [cb1, cb2]


# ============================================================
# 6. get_current_metrics
# ============================================================
class TestGetCurrentMetrics:
    def test_empty_returns_none(self, fresh_monitor):
        assert fresh_monitor.get_current_metrics() is None

    def test_returns_last(self, fresh_monitor):
        m1 = DatabaseMetrics(
            timestamp=1.0, connection_pool_size=10, connections_checked_in=0,
            connections_checked_out=0, connection_pool_overflow=0,
            connection_pool_invalid=0, query_count=0, error_count=0,
            avg_response_time=0, status="healthy",
        )
        m2 = DatabaseMetrics(
            timestamp=2.0, connection_pool_size=20, connections_checked_in=0,
            connections_checked_out=0, connection_pool_overflow=0,
            connection_pool_invalid=0, query_count=0, error_count=0,
            avg_response_time=0, status="healthy",
        )
        fresh_monitor.metrics_history.append(m1)
        fresh_monitor.metrics_history.append(m2)
        assert fresh_monitor.get_current_metrics() is m2


# ============================================================
# 7. get_metrics_summary
# ============================================================
class TestGetMetricsSummary:
    def test_empty_history(self, fresh_monitor):
        result = fresh_monitor.get_metrics_summary()
        assert "error" in result
        assert "没有可用" in result["error"]

    def test_with_history(self, fresh_monitor):
        # 设 self.query_count / self.error_count（get_metrics_summary 读的是这些）
        fresh_monitor.query_count = 5
        fresh_monitor.error_count = 1
        # 加一个 30 分钟内的指标
        m = DatabaseMetrics(
            timestamp=time.time(),
            connection_pool_size=10, connections_checked_in=2,
            connections_checked_out=3, connection_pool_overflow=0,
            connection_pool_invalid=0, query_count=5, error_count=1,
            avg_response_time=0.5, status="healthy",
        )
        fresh_monitor.metrics_history.append(m)

        summary = fresh_monitor.get_metrics_summary(minutes=60)
        assert summary["sample_count"] == 1
        assert summary["time_range_minutes"] == 60
        assert summary["connection_pool"]["avg_size"] == 10
        assert summary["connection_pool"]["max_size"] == 10
        assert summary["performance"]["avg_response_time"] == 0.5
        assert summary["queries"]["total_count"] == 5
        assert summary["queries"]["error_count"] == 1
        assert summary["queries"]["error_rate"] == 0.2
        assert summary["status"] == "healthy"

    def test_old_data_filtered(self, fresh_monitor):
        # 加一个 2 小时前的指标
        m = DatabaseMetrics(
            timestamp=time.time() - 7200,  # 2 小时前
            connection_pool_size=10, connections_checked_in=0,
            connections_checked_out=0, connection_pool_overflow=0,
            connection_pool_invalid=0, query_count=0, error_count=0,
            avg_response_time=0, status="healthy",
        )
        fresh_monitor.metrics_history.append(m)
        # 查最近 60 分钟
        result = fresh_monitor.get_metrics_summary(minutes=60)
        assert "error" in result
        assert "60" in result["error"]


# ============================================================
# 8. _check_alerts
# ============================================================
class TestCheckAlerts:
    def _make_metrics(self, **kwargs) -> DatabaseMetrics:
        defaults = dict(
            timestamp=time.time(),
            connection_pool_size=10,
            connections_checked_in=0,
            connections_checked_out=5,
            connection_pool_overflow=0,
            connection_pool_invalid=0,
            query_count=10,
            error_count=0,
            avg_response_time=0.1,
            status="healthy",
        )
        defaults.update(kwargs)
        return DatabaseMetrics(**defaults)

    def test_high_pool_usage(self, fresh_monitor, caplog):
        """连接池使用率 > 80% 触发告警。"""
        import logging
        m = self._make_metrics(connection_pool_size=10, connections_checked_out=9)  # 90%
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            fresh_monitor._check_alerts(m)
        alerts = [r for r in caplog.records if "连接池使用率" in r.message]
        assert len(alerts) == 1

    def test_pool_overflow_alert(self, fresh_monitor, caplog):
        import logging
        m = self._make_metrics(connection_pool_overflow=3)
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            fresh_monitor._check_alerts(m)
        alerts = [r for r in caplog.records if "连接池溢出" in r.message]
        assert len(alerts) == 1

    def test_invalid_connections_alert(self, fresh_monitor, caplog):
        import logging
        m = self._make_metrics(connection_pool_invalid=10)  # > 5
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            fresh_monitor._check_alerts(m)
        alerts = [r for r in caplog.records if "无效连接" in r.message]
        assert len(alerts) == 1

    def test_high_error_rate_alert(self, fresh_monitor, caplog):
        import logging
        m = self._make_metrics(query_count=100, error_count=10)  # 10% > 5%
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            fresh_monitor._check_alerts(m)
        alerts = [r for r in caplog.records if "错误率" in r.message]
        assert len(alerts) == 1

    def test_slow_response_alert(self, fresh_monitor, caplog):
        import logging
        m = self._make_metrics(avg_response_time=2.0)  # > 1s
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            fresh_monitor._check_alerts(m)
        alerts = [r for r in caplog.records if "响应时间过长" in r.message]
        assert len(alerts) == 1

    def test_no_alert_when_healthy(self, fresh_monitor, caplog):
        import logging
        m = self._make_metrics(
            connection_pool_size=10, connections_checked_out=2,  # 20%
            connection_pool_overflow=0,
            connection_pool_invalid=0,
            query_count=100, error_count=1,  # 1%
            avg_response_time=0.1,
        )
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            fresh_monitor._check_alerts(m)
        alert_records = [r for r in caplog.records if "数据库告警" in r.message]
        assert alert_records == []


# ============================================================
# 9. 回调执行 + 异常吞掉
# ============================================================
class TestCallbackExecution:
    def test_callback_called_on_alert(self, fresh_monitor):
        called = []

        def cb(msg, m):
            called.append((msg, m))

        fresh_monitor.add_alert_callback(cb)

        # 直接调 _check_alerts 制造告警
        m = DatabaseMetrics(
            timestamp=time.time(),
            connection_pool_size=10, connections_checked_in=0,
            connections_checked_out=9,  # 90%
            connection_pool_overflow=0,
            connection_pool_invalid=0,
            query_count=10, error_count=0,
            avg_response_time=0.1, status="healthy",
        )
        fresh_monitor._check_alerts(m)
        assert len(called) == 1
        assert "数据库告警" in called[0][0]

    def test_callback_exception_swallowed(self, fresh_monitor, caplog):
        import logging

        def bad_cb(msg, m):
            raise RuntimeError("callback crashed")

        fresh_monitor.add_alert_callback(bad_cb)

        m = DatabaseMetrics(
            timestamp=time.time(),
            connection_pool_size=10, connections_checked_in=0,
            connections_checked_out=9,
            connection_pool_overflow=0,
            connection_pool_invalid=0,
            query_count=10, error_count=0,
            avg_response_time=0.1, status="healthy",
        )
        with caplog.at_level(logging.ERROR, logger="infra.db.monitor"):
            # 不应抛
            fresh_monitor._check_alerts(m)
        errors = [r for r in caplog.records if "告警回调执行失败" in r.message]
        assert len(errors) == 1


# ============================================================
# 10. start/stop
# ============================================================
class TestStartStop:
    def _quick_monitor(self) -> DatabaseMonitor:
        """用短 check_interval，避免线程在 sleep(60) 时残留。"""
        return DatabaseMonitor(check_interval=0.1)

    def test_start_sets_running(self):
        m = self._quick_monitor()
        m.start_monitoring()
        try:
            assert m.is_running is True
            assert m.monitor_thread is not None
        finally:
            m.stop_monitoring()

    def test_start_idempotent(self, caplog):
        import logging
        m = self._quick_monitor()
        with caplog.at_level(logging.WARNING, logger="infra.db.monitor"):
            m.start_monitoring()
            try:
                m.start_monitoring()  # 第二次
                # 第二次应记 warning，不重启
                warnings = [r for r in caplog.records if "已在运行" in r.message]
                assert len(warnings) == 1
            finally:
                m.stop_monitoring()

    def test_stop_idempotent(self):
        # 不在运行状态调 stop 不抛
        m = self._quick_monitor()
        m.stop_monitoring()  # OK

    def test_stop_clears_running(self):
        m = self._quick_monitor()
        m.start_monitoring()
        m.stop_monitoring()
        assert m.is_running is False

    def test_stop_joins_thread(self):
        """stop_monitoring 应 join 线程。

        用 check_interval=0.1 让监控循环里的 sleep 快速结束，
        避免默认 60s 让线程长时间卡在 sleep 里。
        """
        m = DatabaseMonitor(check_interval=0.1)
        m.start_monitoring()
        thread = m.monitor_thread
        assert thread is not None
        m.stop_monitoring()
        # stop_monitoring 在源码里 join(timeout=5)；线程是 daemon，
        # 离开 while 循环后自然结束。给一点点时间让 CPython 调度。
        for _ in range(50):
            if not thread.is_alive():
                break
            time.sleep(0.02)
        assert not thread.is_alive()


# ============================================================
# 11. 便捷函数（全局 db_monitor）
# ============================================================
class TestConvenienceFunctions:
    def test_record_db_query(self):
        # 重置 db_monitor 状态
        db_monitor.query_count = 0
        db_monitor.error_count = 0
        db_monitor.response_times = []

        record_db_query(0.1, True)
        assert db_monitor.query_count == 1

    def test_add_db_alert_callback(self):
        cb = lambda msg, m: None
        # 清空再加
        db_monitor.alert_callbacks.clear()
        add_db_alert_callback(cb)
        assert cb in db_monitor.alert_callbacks

    def test_get_db_metrics_returns_optional(self):
        # 不抛，返回 None 或 metrics
        result = get_db_metrics()
        assert result is None or isinstance(result, DatabaseMetrics)

    def test_start_stop_global(self):
        """全局 start/stop 能跑通。"""
        start_db_monitoring()
        try:
            assert db_monitor.is_running
        finally:
            stop_db_monitoring()


# ============================================================
# 12. 全局 db_monitor 单例
# ============================================================
class TestGlobalSingleton:
    def test_db_monitor_is_database_monitor_instance(self):
        assert isinstance(db_monitor, DatabaseMonitor)

    def test_global_singleton(self):
        from infra.db.monitor import db_monitor as dm2
        assert dm2 is db_monitor


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
