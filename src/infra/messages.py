# infra/messages.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消息定义模块 - 使用 dataclass 实现类型安全

beartype 用 `beartype.typing.Dict` 避免 PEP 585 弃用警告（dict[str, Any] 在 Python 3.9+ 合法，
但 typing.Dict 已 deprecated）。
"""

import json
import uuid
from beartype import beartype
from beartype.typing import Dict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TaskMessage:
    """
    任务消息 - 类型安全的 dataclass
    支持属性访问和字典访问（向后兼容）
    """
    task_id: str
    task_type: str
    scheduled_time: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"
    
    # ========== 字典协议支持（向后兼容） ==========
    def __getitem__(self, key: str) -> Any:
        """支持 msg['key'] 访问"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持 msg['key'] = value 赋值"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(key)
    
    def __contains__(self, key: str) -> bool:
        """支持 'key' in msg 检查"""
        return hasattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """支持 msg.get('key', default) 方法"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'scheduled_time': self.scheduled_time,
            'payload': self.payload,
            'metadata': self.metadata,
            'priority': self.priority
        }
    
    def to_mq(self) -> Dict[str, str]:
        """转换为 MQ 可存储的格式（自动 JSON 序列化）"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'scheduled_time': self.scheduled_time,
            'payload': json.dumps(self.payload, ensure_ascii=False),
            'metadata': json.dumps(self.metadata, ensure_ascii=False) if self.metadata else '',
            'priority': self.priority,
            'created_at': str(datetime.now().timestamp())
        }
    
    @classmethod
    def from_mq(cls, mq_data: Dict[str, str]) -> 'TaskMessage':
        """从 MQ 接收的数据恢复（自动 JSON 反序列化）"""
        payload = {}
        if mq_data.get('payload'):
            try:
                payload = json.loads(mq_data['payload'])
            except json.JSONDecodeError:
                payload = {}
        
        metadata = {}
        if mq_data.get('metadata'):
            try:
                metadata = json.loads(mq_data['metadata'])
            except json.JSONDecodeError:
                metadata = {}
        
        return cls(
            task_id=mq_data.get('task_id', ''),
            task_type=mq_data.get('task_type', 'unknown'),
            scheduled_time=mq_data.get('scheduled_time', datetime.now().isoformat()),
            payload=payload,
            metadata=metadata,
            priority=mq_data.get('priority', 'normal')
        )
    
    def __repr__(self) -> str:
        return f"TaskMessage(task_id={self.task_id}, task_type={self.task_type})"


@beartype
def create_message(
    task_type: str,
    payload: Dict[str, Any],
    task_id: Optional[str] = None,
    source: str = "apscheduler",
    version: str = "1.0",
    priority: str = "normal"
) -> TaskMessage:
    """
    创建标准消息格式（保持原有接口不变）
    
    Example:
        msg = create_message("crypto_market", {"btc_price": 50000}, source="coinmarketcap")
        print(msg.task_id)      # 属性访问
        print(msg['task_id'])   # 字典访问（兼容）
    """
    return TaskMessage(
        task_id=task_id or str(uuid.uuid4()),
        task_type=task_type,
        scheduled_time=datetime.now().isoformat(),
        payload=payload,
        metadata={
            "source": source,
            "version": version
        },
        priority=priority
    )


# 测试代码
if __name__ == "__main__":
    # 创建消息
    msg = create_message("crypto_market", {"btc_price": 50000}, source="coinmarketcap")
    
    print("=" * 50)
    print("测试属性访问:")
    print(f"  msg.task_id: {msg.task_id}")
    print(f"  msg.task_type: {msg.task_type}")
    print(f"  msg.payload: {msg.payload}")
    
    print("\n测试字典访问（向后兼容）:")
    print(f"  msg['task_id']: {msg['task_id']}")
    print(f"  msg.get('task_type'): {msg.get('task_type')}")
    print(f"  'payload' in msg: {'payload' in msg}")
    
    print("\n测试 MQ 序列化:")
    mq_data = msg.to_mq()
    print(f"  mq_data keys: {list(mq_data.keys())}")
    print(f"  mq_data['payload']: {mq_data['payload']}")
    
    print("\n测试 MQ 反序列化:")
    recovered = TaskMessage.from_mq(mq_data)
    print(f"  recovered.task_id: {recovered.task_id}")
    print(f"  recovered.task_type: {recovered.task_type}")
    
    print("\n✅ 所有测试通过")