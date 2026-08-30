#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.messages / infra.mq 模块 import 与核心 dataclass 构造测试

搬运自 macro_monitor.core.messages / mq stream_mq。
只验证能 import + 关键 dataclass 能构造——深度覆盖留待后续。
"""

from __future__ import annotations

import pytest


# ============================================================
# 1. infra.messages 导入
# ============================================================
class TestMessagesImport:
    def test_messages_module_importable(self):
        """infra.messages 能 import。"""
        import infra.messages
        assert hasattr(infra.messages, "TaskMessage")
        assert hasattr(infra.messages, "create_message")

    def test_task_message_construct(self):
        """TaskMessage dataclass 能构造。"""
        from infra.messages import TaskMessage
        msg = TaskMessage(
            task_id="t-1",
            task_type="etl",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v"},
        )
        assert msg.task_id == "t-1"
        assert msg.task_type == "etl"
        assert msg.payload == {"k": "v"}
        assert msg.priority == "normal"  # 默认值
        assert msg.metadata == {}        # 默认 factory

    def test_task_message_dict_protocol(self):
        """TaskMessage 支持 dict 风格访问（向后兼容）。"""
        from infra.messages import TaskMessage
        msg = TaskMessage(
            task_id="t-1",
            task_type="etl",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v"},
        )
        # __getitem__
        assert msg["task_id"] == "t-1"
        # __contains__
        assert "task_id" in msg
        assert "nonexistent" not in msg
        # get()
        assert msg.get("priority") == "normal"
        assert msg.get("nonexistent", "default") == "default"

    def test_create_message_factory(self):
        """create_message 工厂函数能跑。"""
        from infra.messages import create_message
        msg = create_message(
            task_type="etl",
            payload={"k": "v"},
            priority="high",
        )
        assert msg.task_type == "etl"
        assert msg.priority == "high"


# ============================================================
# 2. infra.mq 导入
# ============================================================
class TestMqImport:
    def test_mq_module_importable(self):
        """infra.mq 能 import + 6 个核心符号都在。"""
        from infra.mq import (
            RedisStreamMQ,
            init_mq,
            get_mq,
            MessagePriority,
            QueueConfig,
            QueueName,
        )
        assert RedisStreamMQ is not None
        assert init_mq is not None
        assert get_mq is not None
        assert MessagePriority is not None
        assert QueueConfig is not None
        assert QueueName is not None

    def test_message_priority_constants(self):
        """MessagePriority 类常量（critical/high/normal/low）。"""
        from infra.mq import MessagePriority
        assert MessagePriority.CRITICAL == "critical"
        assert MessagePriority.HIGH == "high"
        assert MessagePriority.NORMAL == "normal"
        assert MessagePriority.LOW == "low"

    def test_stream_mq_class_constructable(self):
        """RedisStreamMQ 类能实例化（不连真 redis）。"""
        from infra.mq import RedisStreamMQ
        # 构造时只存 redis_config，不连 redis
        mq = RedisStreamMQ(redis_config={"host": "localhost", "port": 6379})
        assert mq is not None
        assert mq.redis_config == {"host": "localhost", "port": 6379}


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
