#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AppException 与全部子类的行为测试

覆盖：
1. AppException 基类：message / error_code / details / str 表现
2. 11 个子类的 error_code 固定值与扩展字段
3. SystemException / AuthException 无参可用
4. 子类扩展字段不会自动合并到 details 字典
"""

from __future__ import annotations

import pytest

from infra.exceptions import (
    AppException,
    ConfigurationError,
    DatabaseError,
    RedisError,
    MessageQueueError,
    SchedulerError,
    DataProducerError,
    DataConsumerError,
    APIError,
    BusinessException,
    SystemException,
    AuthException,
)


# ============================================================
# 1. AppException 基类
# ============================================================
class TestAppExceptionBase:
    def test_message_only(self):
        e = AppException("some failure")
        assert e.message == "some failure"
        assert e.error_code is None
        assert e.details == {}
        assert str(e) == "some failure"

    def test_with_error_code(self):
        e = AppException("some failure", error_code="X_001")
        assert e.error_code == "X_001"
        # str 形如 "[CODE] msg"
        assert str(e) == "[X_001] some failure"

    def test_with_details(self):
        e = AppException("oops", error_code="E1", details={"k": 1, "n": 2})
        assert e.details == {"k": 1, "n": 2}
        # str 形如 "[CODE] msg | Details: {...}"
        s = str(e)
        assert "[E1]" in s
        assert "oops" in s
        assert "Details:" in s
        assert "'k': 1" in s or '"k": 1' in s

    def test_details_defaults_to_empty_dict(self):
        e = AppException("oops")
        assert e.details == {}
        # 不传 details 时 str 不应包含 "Details:"
        assert "Details:" not in str(e)

    def test_error_code_none_does_not_appear_in_str(self):
        e = AppException("oops", details={"a": 1})
        # 没有 error_code，str 只有 message + Details
        s = str(e)
        assert "oops" in s
        assert "Details:" in s
        # 不应以 "[" 开头
        assert not s.startswith("[")

    def test_is_exception_subclass(self):
        assert issubclass(AppException, Exception)
        with pytest.raises(AppException):
            raise AppException("x")

    def test_details_none_becomes_empty_dict(self):
        # 显式传 None
        e = AppException("x", details=None)
        assert e.details == {}


# ============================================================
# 2. 各子类的固定 error_code 与扩展字段
# ============================================================
class TestSubclassesErrorCodes:
    def test_configuration_error(self):
        e = ConfigurationError("bad cfg", config_key="db.url")
        assert e.error_code == "CONFIG_ERROR"
        assert e.config_key == "db.url"
        assert e.message == "bad cfg"
        assert e.details == {}  # 扩展字段不会自动并入 details
        assert "CONFIG_ERROR" in str(e)

    def test_database_error(self):
        e = DatabaseError("connect fail", operation="connect")
        assert e.error_code == "DB_ERROR"
        assert e.operation == "connect"

    def test_redis_error(self):
        e = RedisError("redis down", operation="get")
        assert e.error_code == "REDIS_ERROR"
        assert e.operation == "get"

    def test_message_queue_error(self):
        e = MessageQueueError("queue fail", queue_name="orders", operation="publish")
        assert e.error_code == "MQ_ERROR"
        assert e.queue_name == "orders"
        assert e.operation == "publish"

    def test_scheduler_error(self):
        e = SchedulerError("job crashed", job_id="cron-001")
        assert e.error_code == "SCHEDULER_ERROR"
        assert e.job_id == "cron-001"

    def test_data_producer_error(self):
        e = DataProducerError("producer fail", source="binance")
        assert e.error_code == "PRODUCER_ERROR"
        assert e.source == "binance"

    def test_data_consumer_error(self):
        e = DataConsumerError("consumer fail", task_type="kline", task_id="t-1")
        assert e.error_code == "CONSUMER_ERROR"
        assert e.task_type == "kline"
        assert e.task_id == "t-1"

    def test_api_error(self):
        e = APIError("http fail", status_code=500, endpoint="/v1/orders")
        assert e.error_code == "API_ERROR"
        assert e.status_code == 500
        assert e.endpoint == "/v1/orders"

    def test_business_exception(self):
        e = BusinessException("biz rule", suggestion="retry later", data={"x": 1})
        assert e.error_code == "BIZ_ERROR"
        assert e.suggestion == "retry later"
        assert e.data == {"x": 1}

    def test_system_exception_default_message(self):
        """SystemException 的 message 有默认值，可无参调用。"""
        e = SystemException()
        assert e.error_code == "SYS_ERROR"
        assert e.message  # 非空
        assert e.details == {}
        # 默认 message 应包含"系统异常"
        assert "系统" in e.message

    def test_system_exception_with_data(self):
        e = SystemException(data={"k": 1})
        assert e.error_code == "SYS_ERROR"
        assert e.data == {"k": 1}

    def test_auth_exception_default_message(self):
        e = AuthException()
        assert e.error_code == "AUTH_ERROR"
        assert e.message
        # 默认 message 应包含"微信号"或"绑定"
        assert ("微信号" in e.message) or ("绑定" in e.message)

    def test_auth_exception_with_suggestion(self):
        e = AuthException(suggestion="scan QR")
        assert e.suggestion == "scan QR"


# ============================================================
# 3. 子类能被 except AppException 接住
# ============================================================
class TestSubclassCatchAsBase:
    @pytest.mark.parametrize("exc_cls,args,kwargs", [
        (ConfigurationError, ("x",), {"config_key": "k"}),
        (DatabaseError, ("x",), {"operation": "o"}),
        (RedisError, ("x",), {"operation": "o"}),
        (MessageQueueError, ("x",), {"queue_name": "q", "operation": "o"}),
        (SchedulerError, ("x",), {"job_id": "j"}),
        (DataProducerError, ("x",), {"source": "s"}),
        (DataConsumerError, ("x",), {"task_type": "t", "task_id": "i"}),
        (APIError, ("x",), {"status_code": 500, "endpoint": "/x"}),
        (BusinessException, ("x",), {"suggestion": "s", "data": {}}),
        (SystemException, (), {}),
        (AuthException, (), {}),
    ])
    def test_can_be_caught_as_appexception(self, exc_cls, args, kwargs):
        with pytest.raises(AppException) as exc_info:
            raise exc_cls(*args, **kwargs)
        assert exc_info.value.error_code is not None

    @pytest.mark.parametrize("exc_cls,args,kwargs", [
        (ConfigurationError, ("x",), {"config_key": "k"}),
        (DatabaseError, ("x",), {"operation": "o"}),
        (SystemException, (), {}),
        (AuthException, (), {}),
    ])
    def test_can_be_caught_as_exception(self, exc_cls, args, kwargs):
        with pytest.raises(Exception):
            raise exc_cls(*args, **kwargs)


# ============================================================
# 4. 扩展字段不会并入 details（已知行为，可能未来修复）
# ============================================================
class TestExtensionFieldsIsolation:
    def test_config_key_not_in_details(self):
        e = ConfigurationError("x", config_key="k1", details={"explicit": 1})
        # 显式传的 details 保持
        assert e.details == {"explicit": 1}
        # config_key 是独立属性，不在 details 里
        assert "config_key" not in e.details

    def test_operation_not_in_details(self):
        e = DatabaseError("x", operation="op1")
        assert e.operation == "op1"
        assert "operation" not in e.details

    def test_status_code_not_in_details(self):
        e = APIError("x", status_code=503, endpoint="/foo")
        assert e.status_code == 503
        assert "status_code" not in e.details
        assert "endpoint" not in e.details


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
