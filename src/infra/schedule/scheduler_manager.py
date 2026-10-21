# infra/schedule/scheduler_manager.py

from __future__ import annotations
import asyncio
import traceback
from datetime import datetime
import threading
from typing import Callable, Dict, Any, List, Optional
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
    trigger: Optional[str] = None  # add_job 时 narrow truthy → str；_add_scheduled_job 内部 .lower() 安全
    trigger_args: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    name: Optional[str] = None
    max_instances: int = 1
    misfire_grace_time: int = 300
    coalesce: bool = True
    replace_existing: bool = False
    args: tuple = ()
    kwargs: Optional[Dict[str, Any]] = None
    run_immediately: bool = False
    enabled: bool = True


class SchedulerManager:
    """任务调度管理器 - 单例模式"""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> "SchedulerManager":
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        # 显式 fallback（避免 mypy 推断 None）
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._initialized:
            self.scheduler: Optional[AsyncIOScheduler] = None
            self._job_configs: Dict[str, JobConfig] = {}
            # 类属性显式标 bool（否则 mypy 推断 Any → is_leader 返 Any）
            self._is_leader: bool = False
            self._initialized = True

    def init_scheduler(self, is_leader: bool) -> None:
        """初始化调度器（由 lifespan 调用，传入是否 leader）"""
        self._is_leader = is_leader
        logger.info(f"init_scheduler: is_leader={is_leader}")
        if not is_leader:
            logger.info("非 Leader 实例，不启动 Scheduler")
            return
        # 重复初始化时先关掉旧的——否则旧的 scheduler 仍在跑，
        # 但引用已被丢弃，没人能 shutdown 它 -> 定时任务重复执行。
        #
        # 注意：这里的 shutdown() 只是**发起**关闭（APScheduler 内部走
        # call_soon_threadsafe），真正停稳需要 event loop 跑一轮。
        # 这是同步 API 的固有限制——需要「确保停稳」的场景请用
        # async_shutdown()（如 on_lost_leader）。
        if self.scheduler is not None:
            logger.warning("Scheduler 已存在，先关闭旧的再重新初始化")
            self.shutdown()
        self.scheduler = AsyncIOScheduler(
            jobstores={'default': MemoryJobStore()},
            timezone="Asia/Shanghai"
        )
        logger.info("Scheduler 已初始化（Leader 模式）")

    async def on_lost_leader(self) -> None:
        """失去 leader 时的处理——**停掉调度器**，避免脑裂。

        应挂到 :class:`LeaderElection` 的 ``on_lost_leader`` 回调::

            leader = LeaderElection(cfg, on_lost_leader=scheduler.on_lost_leader)

        锁被别的实例抢走后若继续跑定时任务，同一个任务会被多个实例
        同时执行（实测：LeaderElection 已判定失去 leader，但调度器
        仍在 running）。

        这是 **async** 方法——必须 await 到关闭真正完成，
        否则「停止」只是发起停止，任务可能还会再执行一次。
        """
        if not self._is_leader:
            return
        logger.error("失去 Leader 身份，停止 Scheduler 以避免任务重复执行")
        self._is_leader = False
        await self.async_shutdown()

    def start(self) -> None:
        """启动调度器（仅 leader 执行）"""
        if not self._is_leader:
            logger.debug("非 Leader 实例，跳过 Scheduler 启动")
            return
        if self.scheduler:
            self.scheduler.start()
            logger.info("Scheduler 启动成功")
        else:
            logger.warning("Scheduler 未初始化，无法启动")

    def shutdown(self) -> None:
        """关闭调度器（可重复调用，安全）。

        .. note::
           APScheduler 的 ``AsyncIOScheduler.shutdown()`` 内部走
           ``call_soon_threadsafe``——**同步调用后关闭尚未真正完成**，
           需要 event loop 再跑一轮。

           因此在异步上下文里请用 :meth:`async_shutdown`（它会 await
           关闭完成）。本方法保持同步语义，适合「发起关闭即可」的场景
           （如进程退出）。
        """
        if self.scheduler is None:
            logger.debug("Scheduler 未初始化，跳过关闭")
            return

        if not self.scheduler.running:
            logger.debug("Scheduler 未运行，跳过关闭")
            return

        try:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler 已关闭（已发起）")
        except Exception as e:
            logger.warning(f"关闭 Scheduler 时出现异常: {e}")

    async def async_shutdown(self) -> None:
        """关闭调度器并**确保关闭完成**（异步版本）。

        APScheduler 的 ``shutdown()`` 是「异步投递」——同步调用后
        调度器可能仍在 running。异步场景必须用它：让 event loop 跑一轮，
        确认真正停下来了。

        失去 leader 时**必须**用这个（见 :meth:`on_lost_leader`）——
        否则「已停止」只是发起停止，任务可能还会再执行一次。
        """
        if self.scheduler is None:
            logger.debug("Scheduler 未初始化，跳过关闭")
            return

        if not self.scheduler.running:
            logger.debug("Scheduler 未运行，跳过关闭")
            return

        try:
            self.scheduler.shutdown(wait=True)
            # 让 event loop 处理内部的 call_soon_threadsafe——
            # 否则 shutdown 只是「发起」而非「已完成」
            await asyncio.sleep(0)
            logger.info("Scheduler 已关闭（已确认）")
        except Exception as e:
            logger.warning(f"关闭 Scheduler 时出现异常: {e}")

    def add_job(self, config: JobConfig) -> Optional[str]:
        """添加任务。

        立即执行的任务对所有实例有效；**定时任务仅 leader 会真正添加**。

        Returns:
            job_id——成功添加（或配置已登记）时返回；
            None 表示添加失败（异常已记录日志）

        Raises:
            不抛异常——失败时记日志并返回 None（保持既有调用契约）。

        Note:
            非 leader 实例收到带 trigger 的任务时，**只登记配置不实际调度**，
            返回的 job_id 仅代表「配置已记录」，不代表「定时任务已生效」。
            需要确认是否真正生效请用 :meth:`get_jobs`。
        """
        logger.debug(
            f"🔍 add_job 入口: job_id={config.job_id}, "
            f"trigger={config.trigger}, enabled={config.enabled}"
        )
        try:
            if not config.job_id:
                import time
                config.job_id = f"{config.func.__name__}_{int(time.time())}"

            if not config.enabled:
                # 禁用任务：只登记配置，不调度
                self._job_configs[config.job_id] = config
                logger.info(f"任务 {config.job_id} 已禁用")
                return config.job_id

            # 立即执行（所有实例都会执行，通过 submit_task 提交到队列）
            if config.run_immediately:
                self._execute_immediately(config)

            logger.debug(
                f"🔍 检查定时任务条件: job_id={config.job_id}, "
                f"trigger={config.trigger}, is_leader={self._is_leader}"
            )

            # 定时任务（仅 leader 添加）
            if config.trigger and self._is_leader:
                # 先添加成功再登记配置——避免「登记了但实际没加」的
                # 状态不一致（异常时 _job_configs 里不该有这个 job）
                self._add_scheduled_job(config)
                self._job_configs[config.job_id] = config
            elif config.trigger and not self._is_leader:
                # 非 leader：登记配置但不调度——让调用方知道「配置在这，
                # 但本实例不会执行它」
                self._job_configs[config.job_id] = config
                logger.debug(f"非 Leader 实例，跳过定时任务添加: {config.job_id}")
            else:
                self._job_configs[config.job_id] = config

            return config.job_id
        except Exception as e:
            # 失败时**不写** _job_configs——否则状态不一致：
            # 配置里记着这个 job，但实际并没有被调度
            logger.error(
                f"添加任务失败: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return None

    @staticmethod
    def _wrap_execution_result(result: Any) -> Dict[str, Any]:
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

    def _execute_immediately(self, config: JobConfig) -> None:
        async def wrapper() -> Any:
            try:
                result = await config.func(*config.args, **(config.kwargs or {}))
                return self._wrap_execution_result(result)
            except Exception as e:
                logger.error(f"任务 {config.job_id} 执行异常: {e}")
                return {"status": "error", "error": str(e)}
        
        logger.debug(f"🚀 提交立即执行任务: {config.job_id}")
        submit_task(wrapper())

    def _add_scheduled_job(self, config: JobConfig) -> None:
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.date import DateTrigger
        logger.debug(f"📌 _add_scheduled_job 开始: {config.job_id}")
        trigger_args = config.trigger_args or {}
        # narrow trigger Optional[str] → str（调用方 add_job 已 truthy 校验）
        trigger_type = (config.trigger or "").lower()
        if trigger_type == 'cron':
            trigger = CronTrigger(**trigger_args)
        elif trigger_type == 'interval':
            trigger = IntervalTrigger(**trigger_args)
        elif trigger_type == 'date':
            trigger = DateTrigger(**trigger_args)
        else:
            raise ValueError(f"不支持的触发器类型: {config.trigger}")

        async def wrapper() -> Any:
            return await config.func(*config.args, **(config.kwargs or {}))

        def scheduled_job() -> None:
            logger.debug(f"⚠️ scheduled_job 被触发: {config.job_id}, 时间: {datetime.now()}")
            submit_task(wrapper())

        logger.debug(f"添加定时任务 {config.job_id}, max_instances={config.max_instances}")

        scheduler = self.scheduler
        if scheduler is None:
            logger.warning(f"scheduler 未初始化，无法添加 {config.job_id}")
            return
        scheduler.add_job(
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

    def remove_job(self, job_id: str) -> None:
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
        result: List[Dict[str, Any]] = [
            {
                'id': job.id,
                'name': job.name,
                'next_run_time': str(job.next_run_time) if job.next_run_time else None,
                'trigger': str(job.trigger)
            }
            for job in jobs
        ]
        return result

    def is_leader(self) -> bool:
        return self._is_leader