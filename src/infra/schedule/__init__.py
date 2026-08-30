#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心任务调度模块
"""

from .scheduler_manager import SchedulerManager, JobConfig
from .unified_scheduler import UnifiedScheduler
from .task_registry import task_registry, TaskRegistry

__all__ = [
    'SchedulerManager',
    'JobConfig', 
    'UnifiedScheduler',
    'task_registry',
    'TaskRegistry'
]

# 全局实例（延迟创建）
scheduler_manager = None
unified_scheduler = None


def get_scheduler_manager() -> "SchedulerManager":
    """获取全局调度器管理器实例"""
    global scheduler_manager
    if scheduler_manager is None:
        scheduler_manager = SchedulerManager()
    return scheduler_manager


def get_unified_scheduler() -> "UnifiedScheduler":
    """获取全局统一调度器实例"""
    global unified_scheduler
    if unified_scheduler is None:
        unified_scheduler = UnifiedScheduler()
    return unified_scheduler