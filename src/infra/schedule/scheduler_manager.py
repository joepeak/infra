# infra/schedule/scheduler_manager.py

from __future__ import annotations
import traceback
from datetime import datetime
import threading
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore

from infra.logger import get_logger
from .helpers import submit_task

logger = get_logger(__name__)


@dataclass
class JobConfig:
    """任务配置"""
    func: Callable
    trigger: Optional[str] = None
    trigger_args: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    name: Optional[str] = None
    max_instances: int = 1
    misfire_grace_time: int = 300
    coalesce: bool = True
    replace_existing: bool = False
    args: tuple = ()
    kwargs: dict = None
    run_immediately: bool = False
    enabled: bool = True


class SchedulerManager:
    """任务调度管理器 - 单例模式"""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.scheduler: Optional[AsyncIOScheduler] = None
            self._job_configs: Dict[str, JobConfig] = {}
            self._is_leader = False
            self._initialized = True

    def init_scheduler(self, is_leader: bool):
        """初始化调度器（由 lifespan 调用，传入是否 leader）"""
        self._is_leader = is_leader
        logger.info(f"init_scheduler: is_leader={is_leader}")
        if not is_leader:
            logger.info("非 Leader 实例，不启动 Scheduler")
            return
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            timezone="Asia/Shanghai"
        )
        logger.info("Scheduler 已初始化（Leader 模式）")

    def start(self):
        """启动调度器（仅 leader 执行）"""
        if not self._is_leader:
            logger.debug("非 Leader 实例，跳过 Scheduler 启动")
            return
        if self.scheduler:
            self.scheduler.start()
            logger.info("Scheduler 启动成功")
        else:
            logger.warning("Scheduler 未初始化，无法启动")

    def shutdown(self):
        """关闭调度器（可重复调用，安全）"""
        if self.scheduler is None:
            logger.debug("Scheduler 未初始化，跳过关闭")
            return
        
        if not self.scheduler.running:
            logger.debug("Scheduler 未运行，跳过关闭")
            return
        
        try:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler 已关闭")
        except Exception as e:
            logger.warning(f"关闭 Scheduler 时出现异常: {e}")

    def add_job(self, config: JobConfig) -> Optional[str]:
        """添加任务（立即执行对所有实例有效，定时任务仅 leader 添加）"""
        logger.debug(f"🔍 add_job 入口: job_id={config.job_id}, trigger={config.trigger}, enabled={config.enabled}")
        try:
            if not config.job_id:
                import time
                config.job_id = f"{config.func.__name__}_{int(time.time())}"

            self._job_configs[config.job_id] = config

            if not config.enabled:
                logger.info(f"任务 {config.job_id} 已禁用")
                return config.job_id

            # 立即执行（所有实例都会执行，通过 submit_task 提交到队列）
            if config.run_immediately:
                self._execute_immediately(config)

            logger.debug(f"🔍 检查定时任务条件: job_id={config.job_id}, trigger={config.trigger}, is_leader={self._is_leader}")

            # 定时任务（仅 leader 添加）
            if config.trigger and self._is_leader:
                self._add_scheduled_job(config)
            elif config.trigger and not self._is_leader:
                logger.debug(f"非 Leader 实例，跳过定时任务添加: {config.job_id}")

            return config.job_id
        except Exception as e:
            logger.error(f"添加任务失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _wrap_execution_result(result) -> dict:
        """统一包装任务执行返回值（20260822 失败语义修复）。

        - False（任务明确报告失败）→ status="error"，不再误报 success
        - True / None / 其他非空值 → success
        - dict 原样透传（任务自定义状态不被覆盖）
        """
        if result is None:
            return {"status": "success", "result": None}
        elif isinstance(result, bool):
            if result is False:
                return {"status": "error", "result": False}
            return {"status": "success", "result": True}
        elif isinstance(result, dict):
            return result
        else:
            return {"status": "success", "result": result}

    def _execute_immediately(self, config: JobConfig):
        async def wrapper():
            try:
                result = await config.func(*config.args, **(config.kwargs or {}))
                return self._wrap_execution_result(result)
            except Exception as e:
                logger.error(f"任务 {config.job_id} 执行异常: {e}")
                return {"status": "error", "error": str(e)}
        
        logger.debug(f"🚀 提交立即执行任务: {config.job_id}")
        submit_task(wrapper())

    def _add_scheduled_job(self, config: JobConfig):
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.date import DateTrigger
        logger.debug(f"📌 _add_scheduled_job 开始: {config.job_id}")
        trigger_args = config.trigger_args or {}
        if config.trigger.lower() == 'cron':
            trigger = CronTrigger(**trigger_args)
        elif config.trigger.lower() == 'interval':
            trigger = IntervalTrigger(**trigger_args)
        elif config.trigger.lower() == 'date':
            trigger = DateTrigger(**trigger_args)
        else:
            raise ValueError(f"不支持的触发器类型: {config.trigger}")

        async def wrapper(): 
            return await config.func(*config.args, **(config.kwargs or {}))

        def scheduled_job():
            logger.debug(f"⚠️ scheduled_job 被触发: {config.job_id}, 时间: {datetime.now()}")
            submit_task(wrapper())

        logger.debug(f"添加定时任务 {config.job_id}, max_instances={config.max_instances}")

        self.scheduler.add_job(
            func=scheduled_job,
            trigger=trigger,
            id=config.job_id,
            name=config.name or config.func.__name__,
            max_instances=config.max_instances,
            misfire_grace_time=config.misfire_grace_time,
            coalesce=config.coalesce,
            replace_existing=config.replace_existing
        )
        logger.debug(f"定时任务 {config.job_id} 添加成功")
        logger.debug(f"📌 _add_scheduled_job 完成: {config.job_id}")

    def remove_job(self, job_id: str):
        if self.scheduler:
            try:
                self.scheduler.remove_job(job_id)
            except Exception as e:
                logger.warning(f"从调度器删除任务失败: {e}")
        self._job_configs.pop(job_id, None)
        logger.debug(f"任务 {job_id} 已删除")

    def get_jobs(self) -> list:
        if not self.scheduler:
            return []
        jobs = self.scheduler.get_jobs()
        return [
            {
                'id': job.id,
                'name': job.name,
                'next_run_time': str(job.next_run_time) if job.next_run_time else None,
                'trigger': str(job.trigger)
            }
            for job in jobs
        ]

    def is_leader(self) -> bool:
        return self._is_leader