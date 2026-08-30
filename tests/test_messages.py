#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.messages 深度测试

infra 是下游项目（crypto-watcher / dramacraft）的基础设施类库。
TaskMessage 是 cron 任务 / 消息队列的通用载体——所有下游业务都依赖。
覆盖度目标是：所有公开方法、序列化往返、边界、错误路径。
"""

from __future__ import annotations

import json
import pytest

from infra.messages import TaskMessage, create_message


# ============================================================
# 1. TaskMessage 构造 + 字段默认值
# ============================================================
class TestTaskMessageConstruct:
    def test_minimal_required_fields(self):
        """只传 4 个必填字段，其余走默认 factory。"""
        msg = TaskMessage(
            task_id="t-1",
            task_type="etl",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v"},
        )
        assert msg.task_id == "t-1"
        assert msg.task_type == "etl"
        assert msg.scheduled_time == "2026-01-01T00:00:00"
        assert msg.payload == {"k": "v"}
        # 默认值
        assert msg.metadata == {}
        assert msg.priority == "normal"

    def test_metadata_default_factory_not_shared(self):
        """metadata 用 default_factory——两个实例的 dict 不共享（避免可变默认值陷阱）。"""
        msg1 = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        msg2 = TaskMessage(
            task_id="t-2", task_type="x",
            scheduled_time="t", payload={},
        )
        msg1.metadata["k"] = "v1"
        # msg2 不应被影响
        assert msg2.metadata == {}

    def test_all_fields_explicit(self):
        """所有字段显式传值。"""
        msg = TaskMessage(
            task_id="t-1",
            task_type="etl",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v"},
            metadata={"src": "test"},
            priority="high",
        )
        assert msg.metadata == {"src": "test"}
        assert msg.priority == "high"

    def test_priority_arbitrary_value_no_validation(self):
        """priority 字段无验证——业务传非法值不报错（应改但当前是 by-design）。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
            priority="WRONG_VALUE",
        )
        # 锁住现状：不会因非法值 raise
        assert msg.priority == "WRONG_VALUE"


# ============================================================
# 2. 字典协议（向后兼容）
# ============================================================
class TestDictProtocol:
    def test_getitem_existing(self):
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={"k": "v"},
        )
        assert msg["task_id"] == "t-1"
        assert msg["payload"] == {"k": "v"}

    def test_getitem_missing_raises_keyerror(self):
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        with pytest.raises(KeyError):
            _ = msg["nonexistent"]

    def test_setitem_existing(self):
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        msg["priority"] = "high"
        assert msg.priority == "high"

    def test_setitem_missing_raises_keyerror(self):
        """__setitem__ 写不存在的 key 抛 KeyError（源码 line 40：raise KeyError）。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        with pytest.raises(KeyError):
            msg["nonexistent"] = "v"

    def test_contains(self):
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        assert "task_id" in msg
        assert "priority" in msg
        assert "nonexistent" not in msg

    def test_get_with_default(self):
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        assert msg.get("task_id") == "t-1"
        assert msg.get("nonexistent", "fallback") == "fallback"
        assert msg.get("nonexistent") is None

    def test_repr(self):
        msg = TaskMessage(
            task_id="t-1", task_type="etl",
            scheduled_time="t", payload={},
        )
        r = repr(msg)
        # 锁住现状格式
        assert "t-1" in r
        assert "etl" in r
        assert "TaskMessage" in r


# ============================================================
# 3. to_dict 转换
# ============================================================
class TestToDict:
    def test_to_dict_keys_complete(self):
        """to_dict 包含所有 6 个字段。"""
        msg = TaskMessage(
            task_id="t-1", task_type="etl",
            scheduled_time="2026-01-01T00:00:00", payload={"k": "v"},
            metadata={"src": "x"}, priority="high",
        )
        d = msg.to_dict()
        assert set(d.keys()) == {
            "task_id", "task_type", "scheduled_time",
            "payload", "metadata", "priority",
        }
        assert d["task_id"] == "t-1"
        assert d["task_type"] == "etl"
        assert d["payload"] == {"k": "v"}
        assert d["metadata"] == {"src": "x"}
        assert d["priority"] == "high"

    def test_to_dict_is_plain_dict(self):
        """to_dict 返普通 dict——不是 TaskMessage 自身。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        d = msg.to_dict()
        assert type(d) is dict
        # 改 d 不影响 msg（验证不是引用共享）
        d["task_id"] = "CHANGED"
        assert msg.task_id == "t-1"

    def test_to_dict_nested_payload(self):
        """嵌套 dict payload 完整保留。"""
        nested = {"a": {"b": [1, 2, {"c": "deep"}]}, "list": [{"x": 1}]}
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload=nested,
        )
        d = msg.to_dict()
        assert d["payload"] == nested


# ============================================================
# 4. to_mq 序列化（JSON 编码）
# ============================================================
class TestToMq:
    def test_to_mq_json_encodes_payload(self):
        """to_mq 把 payload 序列化为 JSON 字符串。"""
        msg = TaskMessage(
            task_id="t-1", task_type="etl",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v", "n": 42},
        )
        mq = msg.to_mq()
        # payload 应该是 JSON 字符串
        assert isinstance(mq["payload"], str)
        assert json.loads(mq["payload"]) == {"k": "v", "n": 42}

    def test_to_mq_metadata_empty_string(self):
        """metadata 为空时存空字符串（不是 '{}'）—— Redis Stream 节省空间。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        mq = msg.to_mq()
        assert mq["metadata"] == ""

    def test_to_mq_metadata_nonempty_json(self):
        """metadata 非空时 JSON 序列化。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
            metadata={"src": "test", "v": "1.0"},
        )
        mq = msg.to_mq()
        assert json.loads(mq["metadata"]) == {"src": "test", "v": "1.0"}

    def test_to_mq_added_created_at(self):
        """to_mq 自动加 created_at 字段（发送时间戳）。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={},
        )
        mq = msg.to_mq()
        assert "created_at" in mq
        # 应该是字符串化的 float
        float(mq["created_at"])  # 不抛异常即合法

    def test_to_mq_unicode_safe(self):
        """中文 payload 不被 escape（ensure_ascii=False）。"""
        msg = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={"city": "上海"},
        )
        mq = msg.to_mq()
        # ensure_ascii=False → 中文字符保留
        assert "上海" in mq["payload"]


# ============================================================
# 5. from_mq 反序列化（边界 + 错误路径）
# ============================================================
class TestFromMq:
    def test_round_trip_preserves_all_fields(self):
        """to_mq → from_mq 字段不丢。"""
        original = TaskMessage(
            task_id="t-1", task_type="etl",
            scheduled_time="2026-01-01T00:00:00",
            payload={"k": "v", "n": 42},
            metadata={"src": "test"},
            priority="high",
        )
        mq = original.to_mq()
        recovered = TaskMessage.from_mq(mq)
        assert recovered.task_id == "t-1"
        assert recovered.task_type == "etl"
        assert recovered.payload == {"k": "v", "n": 42}
        assert recovered.metadata == {"src": "test"}
        assert recovered.priority == "high"

    def test_from_mq_missing_keys_use_defaults(self):
        """from_mq 缺字段时用默认值——锁住 fallback 行为。"""
        mq_data = {"task_id": "t-1"}  # 只有 task_id
        msg = TaskMessage.from_mq(mq_data)
        assert msg.task_id == "t-1"
        assert msg.task_type == "unknown"
        assert msg.payload == {}
        assert msg.metadata == {}
        assert msg.priority == "normal"
        # scheduled_time 兜底为当前时间（datetime.isoformat）
        assert msg.scheduled_time  # 非空字符串

    def test_from_mq_corrupt_json_payload_fallback_to_empty(self):
        """payload 字段值是无效 JSON——fallback 到空 dict（不抛）。"""
        mq_data = {
            "task_id": "t-1", "task_type": "x",
            "scheduled_time": "t", "priority": "normal",
            "payload": "not valid json {",
        }
        msg = TaskMessage.from_mq(mq_data)
        assert msg.payload == {}  # 锁住 fallback

    def test_from_mq_corrupt_json_metadata_fallback_to_empty(self):
        """metadata 字段值是无效 JSON——fallback 到空 dict。"""
        mq_data = {
            "task_id": "t-1", "task_type": "x",
            "scheduled_time": "t", "priority": "normal",
            "payload": "{}",
            "metadata": "bad json{",
        }
        msg = TaskMessage.from_mq(mq_data)
        assert msg.metadata == {}

    def test_from_mq_unicode_payload(self):
        """to_mq 中文 payload 经 from_mq 还原。"""
        original = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload={"city": "上海"},
        )
        mq = original.to_mq()
        recovered = TaskMessage.from_mq(mq)
        assert recovered.payload == {"city": "上海"}

    def test_from_mq_complex_nested_payload(self):
        """复杂嵌套 payload（list of dict / 嵌套 dict）往返不丢。"""
        nested = {
            "users": [
                {"id": 1, "name": "alice", "tags": ["a", "b"]},
                {"id": 2, "name": "bob", "tags": []},
            ],
            "meta": {"count": 2, "page": 1},
        }
        original = TaskMessage(
            task_id="t-1", task_type="x",
            scheduled_time="t", payload=nested,
        )
        recovered = TaskMessage.from_mq(original.to_mq())
        assert recovered.payload == nested


# ============================================================
# 6. create_message 工厂
# ============================================================
class TestCreateMessage:
    def test_minimal_args(self):
        """只传 task_type + payload——自动生成 task_id / scheduled_time / metadata。"""
        msg = create_message("etl", {"k": "v"})
        assert msg.task_type == "etl"
        assert msg.payload == {"k": "v"}
        # 自动生成
        assert msg.task_id  # 非空（UUID）
        assert msg.scheduled_time  # 非空（ISO 格式）
        assert msg.metadata == {"source": "apscheduler", "version": "1.0"}
        assert msg.priority == "normal"

    def test_explicit_task_id_preserved(self):
        """显式 task_id 不被覆盖。"""
        msg = create_message("etl", {}, task_id="my-task-1")
        assert msg.task_id == "my-task-1"

    def test_explicit_source_and_version(self):
        """显式 source / version 写入 metadata。"""
        msg = create_message("etl", {}, source="custom", version="2.0")
        assert msg.metadata == {"source": "custom", "version": "2.0"}

    def test_priority_propagates(self):
        msg = create_message("etl", {}, priority="critical")
        assert msg.priority == "critical"

    def test_auto_task_id_is_uuid(self):
        """自动生成的 task_id 应该是 UUID 格式。"""
        msg = create_message("etl", {})
        # UUID 格式：8-4-4-4-12
        import re
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            msg.task_id,
        ), f"task_id 不是 UUID: {msg.task_id}"

    def test_auto_task_id_unique(self):
        """自动生成的 task_id 多次调用应唯一。"""
        ids = {create_message("etl", {}).task_id for _ in range(10)}
        assert len(ids) == 10  # 不重复


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
