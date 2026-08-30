#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务注册中心
统一管理所有任务的注册和创建
"""

from typing import Dict, Any, Callable, Type, Optional
from infra.logger import get_logger

logger = get_logger(__name__)


class TaskRegistry:
    """任务注册中心"""
    
    def __init__(self) -> None:
        self._task_factories: Dict[str, Callable] = {}
        self._task_classes: Dict[str, Type] = {}
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """注册任务工厂函数"""
        self._task_factories[name] = factory
        logger.info(f"注册任务工厂: {name}")
    
    def register_class(self, name: str, task_class: Type) -> None:
        """注册任务类"""
        self._task_classes[name] = task_class
        logger.info(f"注册任务类: {name}")
    
    def create_task(self, task_type: str, config: Dict[str, Any]) -> Any:
        """创建任务实例"""
        if task_type in self._task_factories:
            return self._task_factories[task_type](config)
        elif task_type in self._task_classes:
            # 对于类，需要实例化
            return self._task_classes[task_type](config)
        else:
            raise ValueError(f"未知的任务类型: {task_type}")
    
    def list_registered(self) -> Dict[str, Any]:
        """列出所有注册的任务"""
        return {
            'factories': list(self._task_factories.keys()),
            'classes': list(self._task_classes.keys())
        }
    
    def is_registered(self, name: str) -> bool:
        """检查任务是否已注册"""
        return name in self._task_factories or name in self._task_classes


# 全局任务注册中心
task_registry = TaskRegistry()
