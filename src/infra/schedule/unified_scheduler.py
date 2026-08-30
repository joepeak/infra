# infra/schedule/unified_scheduler.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一调度器 - 纯调度框架
只负责调度，不包含任何业务逻辑
"""

from typing import Callable, Dict, Any, List
from .scheduler_manager import SchedulerManager, JobConfig
from infra.logger import get_logger

logger = get_logger(__name__)


class UnifiedScheduler:
    """统一调度器 - 纯调度框架"""
    
    def __init__(self) -> None:
        self.scheduler = SchedulerManager()
        self._task_loaders: Dict[str, Callable[[SchedulerManager], Any]] = {}
    
    def register_task_loader(self, task_type: str, loader_func: Callable[[SchedulerManager], Any]) -> None:
        """注册任务加载器"""
        self._task_loaders[task_type] = loader_func
        logger.info(f"注册任务加载器: {task_type}")
    
    def start(self) -> None:
        """启动调度器（只加载任务，不重复启动）"""
        logger.info("启动统一调度器...")
        # 不要调用 self.scheduler.start()，因为调度器已经在 lifespan 中启动
        self._load_all_tasks()
        jobs = self.scheduler.get_jobs()
        logger.info(f"成功加载 {len(jobs)} 个任务")
    
    def stop(self) -> None:
        """停止调度器（仅 leader 执行）"""
        logger.info("关闭统一调度器...")
        # scheduler.stop 是 noop——真正停止调 self.scheduler.shutdown
        self.scheduler.shutdown()
        logger.info("统一调度器已关闭")
    
    def shutdown(self) -> None:
        """关闭调度器"""
        logger.info("关闭统一调度器...")
        self.scheduler.shutdown()  # 内部已有状态检查
        logger.info("统一调度器已关闭")
    
    def _load_all_tasks(self) -> None:
        """加载所有任务 - 通过注册的加载器"""
        for task_type, loader_func in self._task_loaders.items():
            try:
                logger.info(f"加载{task_type}任务...")
                # 传递 scheduler 对象，不是 scheduler_manager
                loader_func(self.scheduler)  # self.scheduler 是 SchedulerManager 实例
                logger.info(f"加载{task_type}任务完成")
            except Exception as e:
                logger.error(f"加载{task_type}任务失败: {e}")
                import traceback
                traceback.print_exc()
    
    def get_jobs(self) -> list:
        """获取任务状态"""
        return self.scheduler.get_jobs()

    def add_job(self, job_config: JobConfig) -> str:
        """添加任务"""
        result = self.scheduler.add_job(job_config)
        # SchedulerManager.add_job 返 Optional[str]（job_id 或 None）——narrow
        return result or ""
    
    def remove_job(self, job_id: str) -> None:
        """移除任务"""
        return self.scheduler.remove_job(job_id)


# 全局调度器实例
unified_scheduler = UnifiedScheduler()