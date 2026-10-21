#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心任务调度模块

- SchedulerManager：APScheduler 封装（job 增删查、leader 模式启停）
- UnifiedScheduler：任务加载器编排（把「加载任务」与「调度」解耦）
- LeaderElection：基于 Redis 锁的 leader 选举（多实例下只有一个跑定时任务）
- TaskRegistry：任务类型注册中心（工厂 / 类）
"""

from typing import Optional

from .leader import LeaderElection
from .scheduler_manager import SchedulerManager, JobConfig
from .unified_scheduler import UnifiedScheduler
from .task_registry import task_registry, TaskRegistry

__all__ = [
    'SchedulerManager',
    'JobConfig',
    'UnifiedScheduler',
    'LeaderElection',
    'task_registry',
    'TaskRegistry',
    'get_scheduler_manager',
    'get_unified_scheduler',
]

# 全局实例（延迟创建）——
# 用下划线前缀避免与下面的 get_* 函数返回的实例名混淆
_scheduler_manager: Optional["SchedulerManager"] = None
_unified_scheduler: Optional["UnifiedScheduler"] = None


def get_scheduler_manager() -> "SchedulerManager":
    """获取全局调度器管理器实例（单例，首次调用时创建）。"""
    global _scheduler_manager
    if _scheduler_manager is None:
        _scheduler_manager = SchedulerManager()
    return _scheduler_manager


def get_unified_scheduler() -> "UnifiedScheduler":
    """获取全局统一调度器实例（单例，首次调用时创建）。"""
    global _unified_scheduler
    if _unified_scheduler is None:
        _unified_scheduler = UnifiedScheduler()
    return _unified_scheduler
