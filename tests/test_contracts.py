#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra 公开 API 契约测试

infra 是下游项目的基础设施类库——下游业务依赖每个公开方法的
返回类型 + 参数签名。

mypy 在编译期抓类型错，但本测试是 CI 层的"第二道防线"：
- 锁住每个公开方法的返回类型
- 锁住每个公开方法的参数类型
- 锁住 dataclass 关键字段
- 锁住模块级函数的签名

AI 改方法签名时——mypy 报 + 契约测试报，双保险。

注意：本测试只检查签名（inspect.signature + typing.get_type_hints），
不验证行为。行为测试在专门的 test_*.py 里。

策略：每个断言"if "X" in hints: assert ..."——无注解时跳过（mypy 已抓）。
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, get_type_hints

import pytest

from infra.config import (
    init_config,
    get_config,
    set_config,
    replace_config,
    reload_config,
)
from infra.db import (
    init_db_manager,
    get_db_manager,
    get_db_manager_or_raise,
    get_db_session,
    close_all_db_connections,
    list_db_keys,
)
from infra.db.connection_manager import DatabaseConnectionManager
from infra.db.database_model import (
    BaseModel,
    ObservationBaseModel,
    TimeSeriesBaseModel,
    ObservationBase,
    TimeUtil,
    JSONBCompatible,
)
from infra.exceptions import DatabaseError
from infra.llm import (
    LLMClient,
    OpenAIClient,
    LLMFactory,
    init_llm_client,
    LLMRequest,
    LLMResponse,
)
from infra.messages import TaskMessage, create_message
from infra.mq import (
    RedisStreamMQ,
    init_mq,
    get_mq,
    MessagePriority,
    QueueConfig,
)
from infra.redis import (
    RedisClient,
    init_redis,
    get_redis,
    close_redis,
)
from infra.redis.decorators import cache
from infra.schedule import (
    SchedulerManager,
    UnifiedScheduler,
    TaskRegistry,
    get_scheduler_manager,
    get_unified_scheduler,
)
from infra.schedule.scheduler_manager import JobConfig
from infra.schedule.leader import LeaderElection


# ============================================================
# 工具
# ============================================================
def _hints(func) -> dict[str, Any]:
    """拿类型注解 dict。无注解时返 {}。"""
    try:
        return get_type_hints(func)
    except Exception:
        return {}


def _params(func, skip_self: bool = True) -> list[str]:
    """拿参数名列表。"""
    params = list(inspect.signature(func).parameters.keys())
    if skip_self and params and params[0] == "self":
        params = params[1:]
    return params


def _is_dict_type(t: Any) -> bool:
    """判断类型注解是 dict（兼容 typing.Dict[...] / dict | None 等）。"""
    if t is None:
        return False
    if t is dict:
        return True
    if getattr(t, "__origin__", None) is dict:
        return True
    args = getattr(t, "__args__", ())
    for arg in args:
        if arg is dict or getattr(arg, "__origin__", None) is dict:
            return True
    return False


def _has_typed_return(func) -> bool:
    """标了返回类型注解——无论类型是什么。"""
    return "return" in _hints(func)


def _is_bool_typed(func) -> bool:
    return _hints(func).get("return") is bool


def _is_int_typed(func) -> bool:
    return _hints(func).get("return") is int


def _is_none_typed(func) -> bool:
    return _hints(func).get("return") is type(None)


# ============================================================
# 1. infra.config
# ============================================================
class TestConfigContracts:
    def test_init_config_params(self):
        """init_config(config_dir, force_reload=True)。"""
        assert _params(init_config) == ["config_dir", "force_reload"]
        # 锁 force_reload 默认值
        assert inspect.signature(init_config).parameters["force_reload"].default is True

    def test_init_config_returns_dict(self):
        if _has_typed_return(init_config):
            assert _is_dict_type(_hints(init_config)["return"])

    def test_get_config_no_params(self):
        assert _params(get_config) == []

    def test_get_config_returns_dict(self):
        if _has_typed_return(get_config):
            assert _is_dict_type(_hints(get_config)["return"])

    def test_set_config_takes_conf(self):
        assert _params(set_config) == ["conf"]

    def test_set_config_returns_none(self):
        if _has_typed_return(set_config):
            assert _is_none_typed(set_config)

    def test_replace_config_takes_conf(self):
        assert _params(replace_config) == ["conf"]

    def test_replace_config_returns_none(self):
        if _has_typed_return(replace_config):
            assert _is_none_typed(replace_config)

    def test_reload_config_no_params(self):
        assert _params(reload_config) == []

    def test_reload_config_returns_dict(self):
        if _has_typed_return(reload_config):
            assert _is_dict_type(_hints(reload_config)["return"])


# ============================================================
# 2. infra.db
# ============================================================
class TestDbContracts:
    def test_init_db_manager_params_and_defaults(self):
        assert _params(init_db_manager) == ["key", "require_db"]
        sig = inspect.signature(init_db_manager)
        assert sig.parameters["key"].default == "db"
        assert sig.parameters["require_db"].default is False

    def test_init_db_manager_returns_manager(self):
        if _has_typed_return(init_db_manager):
            assert _hints(init_db_manager)["return"] is DatabaseConnectionManager

    def test_get_db_manager_params(self):
        assert _params(get_db_manager) == ["key"]
        assert inspect.signature(get_db_manager).parameters["key"].default == "db"

    def test_get_db_manager_or_raise_returns_manager(self):
        if _has_typed_return(get_db_manager_or_raise):
            assert _hints(get_db_manager_or_raise)["return"] is DatabaseConnectionManager

    def test_get_db_session_callable(self):
        assert callable(get_db_session)

    def test_close_all_db_connections_is_async(self):
        assert inspect.iscoroutinefunction(close_all_db_connections)

    def test_list_db_keys_no_params(self):
        assert _params(list_db_keys) == []

    def test_database_connection_manager_init_params(self):
        """DatabaseConnectionManager(db_config_key, require_db)。"""
        # _params skip_self=True
        assert _params(DatabaseConnectionManager.__init__) == ["db_config_key", "require_db"]
        sig = inspect.signature(DatabaseConnectionManager.__init__)
        assert sig.parameters["db_config_key"].default == "db"
        assert sig.parameters["require_db"].default is True

    def test_job_config_dataclass(self):
        from dataclasses import fields
        names = {f.name for f in fields(JobConfig)}
        for field_name in ["func", "trigger", "job_id", "max_instances",
                          "misfire_grace_time", "coalesce", "replace_existing",
                          "args", "kwargs", "run_immediately", "enabled"]:
            assert field_name in names, f"JobConfig 缺字段: {field_name}"


# ============================================================
# 3. infra.messages
# ============================================================
class TestMessagesContracts:
    def test_task_message_dataclass(self):
        from dataclasses import fields
        names = {f.name for f in fields(TaskMessage)}
        for field_name in ["task_id", "task_type", "scheduled_time",
                          "payload", "metadata", "priority"]:
            assert field_name in names, f"TaskMessage 缺字段: {field_name}"

    def test_create_message_params(self):
        assert _params(create_message) == ["task_type", "payload", "task_id", "source", "version", "priority"]

    def test_create_message_returns_task_message(self):
        if _has_typed_return(create_message):
            assert _hints(create_message)["return"] is TaskMessage


# ============================================================
# 4. infra.mq
# ============================================================
class TestMqContracts:
    def test_redis_stream_mq_init_takes_redis_config(self):
        assert _params(RedisStreamMQ.__init__) == ["redis_config"]

    def test_redis_stream_mq_create_queue_params(self):
        assert "stream_name" in _params(RedisStreamMQ.create_queue)
        assert "group_name" in _params(RedisStreamMQ.create_queue)

    def test_redis_stream_mq_create_queue_returns_self(self):
        if _has_typed_return(RedisStreamMQ.create_queue):
            assert _hints(RedisStreamMQ.create_queue)["return"] is RedisStreamMQ

    def test_redis_stream_mq_publish_is_async(self):
        assert inspect.iscoroutinefunction(RedisStreamMQ.publish)

    def test_redis_stream_mq_publish_simple_is_async(self):
        assert inspect.iscoroutinefunction(RedisStreamMQ.publish_simple)

    def test_init_mq_is_async(self):
        assert inspect.iscoroutinefunction(init_mq)
        assert "config" in _params(init_mq)
        assert "routing" in _params(init_mq)

    def test_init_mq_returns_redis_stream_mq(self):
        if _has_typed_return(init_mq):
            assert _hints(init_mq)["return"] is RedisStreamMQ

    def test_message_priority_constants(self):
        """锁住 4 个优先级——业务约定。"""
        assert MessagePriority.CRITICAL == "critical"
        assert MessagePriority.HIGH == "high"
        assert MessagePriority.NORMAL == "normal"
        assert MessagePriority.LOW == "low"

    def test_queue_config_dataclass(self):
        from dataclasses import fields
        names = {f.name for f in fields(QueueConfig)}
        for field_name in ["stream_name", "group_name", "maxlen", "block_ms",
                          "batch_size", "max_retries", "concurrency",
                          "claim_interval", "claim_min_idle_ms", "ttl_seconds"]:
            assert field_name in names, f"QueueConfig 缺字段: {field_name}"


# ============================================================
# 5. infra.schedule
# ============================================================
class TestScheduleContracts:
    def test_task_registry_register_factory_params(self):
        assert "name" in _params(TaskRegistry.register_factory)
        assert "factory" in _params(TaskRegistry.register_factory)

    def test_task_registry_create_task_params(self):
        assert "task_type" in _params(TaskRegistry.create_task)
        assert "config" in _params(TaskRegistry.create_task)

    def test_unified_scheduler_register_task_loader_params(self):
        assert "task_type" in _params(UnifiedScheduler.register_task_loader)
        assert "loader_func" in _params(UnifiedScheduler.register_task_loader)

    def test_scheduler_manager_add_job_takes_config(self):
        assert "config" in _params(SchedulerManager.add_job)

    def test_leader_election_init_takes_redis_config(self):
        assert "redis_config" in _params(LeaderElection.__init__)

    def test_leader_election_try_acquire_is_async(self):
        assert inspect.iscoroutinefunction(LeaderElection.try_acquire)
        if _has_typed_return(LeaderElection.try_acquire):
            assert _is_bool_typed(LeaderElection.try_acquire)

    def test_leader_election_release_is_async(self):
        assert inspect.iscoroutinefunction(LeaderElection.release)
        if _has_typed_return(LeaderElection.release):
            assert _is_bool_typed(LeaderElection.release)

    def test_leader_election_is_leader_returns_bool(self):
        if _has_typed_return(LeaderElection.is_leader):
            assert _is_bool_typed(LeaderElection.is_leader)


# ============================================================
# 6. infra.redis
# ============================================================
class TestRedisContracts:
    def test_redis_client_init_takes_config(self):
        assert _params(RedisClient.__init__) == ["config"]
        assert inspect.signature(RedisClient.__init__).parameters["config"].default is None

    def test_redis_client_get_is_async(self):
        assert inspect.iscoroutinefunction(RedisClient.get)
        assert "key" in _params(RedisClient.get)

    def test_redis_client_set_is_async(self):
        assert inspect.iscoroutinefunction(RedisClient.set)
        params = _params(RedisClient.set)
        assert "key" in params
        assert "value" in params
        assert "ttl" in params
        if _has_typed_return(RedisClient.set):
            assert _is_bool_typed(RedisClient.set)

    def test_redis_client_delete_is_async(self):
        assert inspect.iscoroutinefunction(RedisClient.delete)
        assert "keys" in _params(RedisClient.delete)
        if _has_typed_return(RedisClient.delete):
            assert _is_int_typed(RedisClient.delete)

    def test_redis_client_lock_is_async(self):
        assert inspect.iscoroutinefunction(RedisClient.lock)
        params = _params(RedisClient.lock)
        assert "key" in params
        assert "ttl" in params
        # 锁 ttl 默认值
        assert inspect.signature(RedisClient.lock).parameters["ttl"].default == 30
        if _has_typed_return(RedisClient.lock):
            assert _is_bool_typed(RedisClient.lock)

    def test_redis_client_unlock_is_async(self):
        assert inspect.iscoroutinefunction(RedisClient.unlock)
        if _has_typed_return(RedisClient.unlock):
            assert _is_bool_typed(RedisClient.unlock)

    def test_redis_client_lock_context_callable(self):
        assert callable(RedisClient.lock_context)

    def test_redis_client_get_or_set_is_async(self):
        assert inspect.iscoroutinefunction(RedisClient.get_or_set)
        params = _params(RedisClient.get_or_set)
        assert "key" in params
        assert "func" in params
        assert "ttl" in params

    def test_init_redis_is_async(self):
        assert inspect.iscoroutinefunction(init_redis)
        if _has_typed_return(init_redis):
            assert _hints(init_redis)["return"] is RedisClient

    def test_close_redis_is_async(self):
        assert inspect.iscoroutinefunction(close_redis)

    def test_cache_decorator_params(self):
        assert "key" in _params(cache)
        assert "ttl" in _params(cache)


# ============================================================
# 7. infra.llm
# ============================================================
class TestLlmContracts:
    def test_llm_client_is_abstract(self):
        """LLMClient 是抽象基类——不能直接实例化。"""
        assert hasattr(LLMClient, "__abstractmethods__")
        assert len(LLMClient.__abstractmethods__) > 0

    def test_openai_client_init_params(self):
        params = _params(OpenAIClient.__init__)
        for p in ["api_key", "base_url", "model", "max_retries", "timeout"]:
            assert p in params, f"OpenAIClient.__init__ 缺参数: {p}"
        # 锁 model 默认值——业务可能依赖 gpt-4o
        assert inspect.signature(OpenAIClient.__init__).parameters["model"].default == "gpt-4o"

    def test_openai_client_ainvoke_is_async(self):
        assert inspect.iscoroutinefunction(OpenAIClient.ainvoke)

    def test_llm_factory_create_takes_config(self):
        assert "config" in _params(LLMFactory.create)
        if _has_typed_return(LLMFactory.create):
            assert _hints(LLMFactory.create)["return"] is LLMClient

    def test_init_llm_client_is_async(self):
        assert inspect.iscoroutinefunction(init_llm_client)

    def test_llm_request_fields(self):
        """LLMRequest 关键字段——pydantic BaseModel 不是 dataclass，用 model_fields。"""
        names = set(LLMRequest.model_fields.keys())
        for p in ["messages", "model", "temperature", "max_tokens"]:
            assert p in names, f"LLMRequest 缺字段: {p}"

    def test_llm_response_fields(self):
        names = set(LLMResponse.model_fields.keys())
        for p in ["content", "tool_calls", "usage", "model"]:
            assert p in names, f"LLMResponse 缺字段: {p}"


# ============================================================
# 8. infra.db.database_model
# ============================================================
class TestDatabaseModelContracts:
    def test_basemodel_abstract(self):
        assert BaseModel.__abstract__ is True

    def test_observationbasemodel_abstract(self):
        assert ObservationBaseModel.__abstract__ is True

    def test_timeseriesbasemodel_abstract(self):
        assert TimeSeriesBaseModel.__abstract__ is True

    def test_observationbase_abstract(self):
        assert ObservationBase.__abstract__ is True

    def test_timeutil_to_utc_event_time_params(self):
        params = _params(TimeUtil.to_utc_event_time)
        assert "val" in params
        assert "source_tz" in params
        # 锁默认时区
        assert inspect.signature(TimeUtil.to_utc_event_time).parameters["source_tz"].default == "UTC"

    def test_jsonbcompatible_is_type_decorator(self):
        from sqlalchemy.types import TypeDecorator
        assert issubclass(JSONBCompatible, TypeDecorator)


# ============================================================
# 9. infra.exceptions
# ============================================================
class TestExceptionsContracts:
    def test_database_error_init_params(self):
        params = _params(DatabaseError.__init__)
        assert "message" in params
        assert "operation" in params


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
