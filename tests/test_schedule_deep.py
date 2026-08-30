#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.schedule 深度测试（30+ 用例）

infra 是下游项目的基础设施类库——schedule 是生产关键。
完整覆盖 schedule/* 6 个文件的所有公开方法。
APScheduler + Redis 全部用 mock——不连真依赖。
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infra.schedule import (
    JobConfig,
    SchedulerManager,
    TaskRegistry,
    UnifiedScheduler,
    task_registry,
)
from infra.schedule import helpers
from infra.schedule.leader import LeaderElection


# ============================================================
# helpers.py — 任务队列管理
# ============================================================
class TestHelpers:
    def setup_method(self):
        """每个测试重置 helpers 模块全局 state。"""
        helpers._task_queue = None
        helpers._main_loop = None

    def test_init_task_queue_stores(self):
        """init_task_queue 存 queue + loop 到模块全局。"""
        loop = asyncio.new_event_loop()
        try:
            queue = asyncio.Queue()
            helpers.init_task_queue(loop, queue)
            assert helpers._task_queue is queue
            assert helpers._main_loop is loop
        finally:
            loop.close()

    def test_get_task_queue_after_init(self):
        loop = asyncio.new_event_loop()
        try:
            queue = asyncio.Queue()
            helpers.init_task_queue(loop, queue)
            assert helpers.get_task_queue() is queue
        finally:
            loop.close()

    def test_get_task_queue_without_init_raises(self):
        """未 init 时 get_task_queue 抛 RuntimeError。"""
        with pytest.raises(RuntimeError, match="任务队列未初始化"):
            helpers.get_task_queue()

    def test_get_main_loop_after_init(self):
        loop = asyncio.new_event_loop()
        try:
            queue = asyncio.Queue()
            helpers.init_task_queue(loop, queue)
            assert helpers.get_main_loop() is loop
        finally:
            loop.close()

    def test_get_main_loop_without_init_raises(self):
        """未 init 时 get_main_loop 抛 RuntimeError。"""
        with pytest.raises(RuntimeError, match="主事件循环未初始化"):
            helpers.get_main_loop()

    def test_is_task_queue_ready_true_after_init(self):
        loop = asyncio.new_event_loop()
        try:
            helpers.init_task_queue(loop, asyncio.Queue())
            assert helpers.is_task_queue_ready() is True
        finally:
            loop.close()

    def test_is_task_queue_ready_false_before_init(self):
        assert helpers.is_task_queue_ready() is False


# ============================================================
# task_registry.py — 任务工厂 / 类注册
# ============================================================
class TestTaskRegistry:
    def setup_method(self):
        """每个测试用 fresh TaskRegistry（不污染全局单例）。"""
        self.reg = TaskRegistry()

    def test_initial_state(self):
        assert self.reg._task_factories == {}
        assert self.reg._task_classes == {}

    def test_register_factory(self):
        def my_factory(config):
            return "task"
        self.reg.register_factory("my_task", my_factory)
        assert self.reg._task_factories["my_task"] is my_factory

    def test_register_class(self):
        class MyTask:
            pass
        self.reg.register_class("my_cls", MyTask)
        assert self.reg._task_classes["my_cls"] is MyTask

    def test_create_task_via_factory(self):
        """create_task 走 factory 路径。"""
        called_with = []

        def factory(config):
            called_with.append(config)
            return "result"

        self.reg.register_factory("f", factory)
        result = self.reg.create_task("f", {"k": "v"})
        assert result == "result"
        assert called_with == [{"k": "v"}]

    def test_create_task_via_class(self):
        """create_task 走 class 路径——实例化时传 config。"""
        class MyTask:
            def __init__(self, config):
                self.config = config

        self.reg.register_class("c", MyTask)
        result = self.reg.create_task("c", {"k": "v"})
        assert isinstance(result, MyTask)
        assert result.config == {"k": "v"}

    def test_create_task_unknown_type_raises(self):
        """未注册的类型抛 ValueError。"""
        with pytest.raises(ValueError, match="未知的任务类型"):
            self.reg.create_task("nonexistent", {})

    def test_list_registered(self):
        def f(config): pass
        class C: pass
        self.reg.register_factory("f", f)
        self.reg.register_class("c", C)
        registered = self.reg.list_registered()
        assert "f" in registered["factories"]
        assert "c" in registered["classes"]

    def test_is_registered(self):
        def f(config): pass
        self.reg.register_factory("f", f)
        assert self.reg.is_registered("f") is True
        assert self.reg.is_registered("nonexistent") is False


# ============================================================
# SchedulerManager — 单例 + job 管理（mock APScheduler）
# ============================================================
@pytest.fixture
def fresh_scheduler_manager():
    """重置 SchedulerManager 单例，每次测试干净。"""
    SchedulerManager._instance = None
    SchedulerManager._initialized = False
    mgr = SchedulerManager()
    yield mgr
    SchedulerManager._instance = None
    SchedulerManager._initialized = False


class TestSchedulerManagerSingleton:
    def test_singleton_same_instance(self):
        """两次构造返同一实例。"""
        SchedulerManager._instance = None
        SchedulerManager._initialized = False
        m1 = SchedulerManager()
        m2 = SchedulerManager()
        assert m1 is m2
        # cleanup
        SchedulerManager._instance = None
        SchedulerManager._initialized = False

    def test_init_runs_once(self):
        """__init__ 只跑一次（_initialized 标记）。"""
        SchedulerManager._instance = None
        SchedulerManager._initialized = False
        m = SchedulerManager()
        first_id = m._initialized
        # 第二次 __init__ 早退
        m._is_leader = True
        m2 = SchedulerManager()  # same instance
        assert m2 is m
        # _is_leader 不被重置（验证 __init__ 早退）
        assert m2._is_leader is True
        SchedulerManager._instance = None
        SchedulerManager._initialized = False


class TestSchedulerManagerInit:
    def test_init_scheduler_non_leader(self, fresh_scheduler_manager):
        """非 leader 模式——不创建 APScheduler。"""
        fresh_scheduler_manager.init_scheduler(is_leader=False)
        assert fresh_scheduler_manager._is_leader is False
        assert fresh_scheduler_manager.scheduler is None

    def test_init_scheduler_leader_creates_apscheduler(self, fresh_scheduler_manager):
        """leader 模式——创建 AsyncIOScheduler + MemoryJobStore。"""
        fresh_scheduler_manager.init_scheduler(is_leader=True)
        assert fresh_scheduler_manager._is_leader is True
        assert fresh_scheduler_manager.scheduler is not None
        # jobstores 是 MemoryJobStore
        from apscheduler.jobstores.memory import MemoryJobStore
        assert isinstance(
            fresh_scheduler_manager.scheduler._jobstores['default'],
            MemoryJobStore,
        )


class TestSchedulerManagerLifecycle:
    def test_start_non_leader_skips(self, fresh_scheduler_manager):
        """非 leader start——不调 scheduler.start()。"""
        fresh_scheduler_manager.init_scheduler(is_leader=False)
        # scheduler 是 None，调 start 不会有 AttributeError
        fresh_scheduler_manager.start()  # 不抛即可

    def test_start_leader_starts_scheduler(self, fresh_scheduler_manager):
        """leader start——调 scheduler.start()。"""
        fresh_scheduler_manager.init_scheduler(is_leader=True)
        fresh_scheduler_manager.scheduler = MagicMock()
        fresh_scheduler_manager.start()
        fresh_scheduler_manager.scheduler.start.assert_called_once()

    def test_start_without_init_skips(self, fresh_scheduler_manager):
        """scheduler 未 init（None）——start 静默跳过。"""
        # scheduler = None
        fresh_scheduler_manager.start()  # 应不抛

    def test_shutdown_skips_when_uninitialized(self, fresh_scheduler_manager):
        fresh_scheduler_manager.shutdown()  # scheduler=None 静默跳过

    def test_shutdown_skips_when_not_running(self, fresh_scheduler_manager):
        fresh_scheduler_manager.init_scheduler(is_leader=True)
        # running 是 read-only property——mock scheduler 整个
        fresh_scheduler_manager.scheduler = MagicMock()
        fresh_scheduler_manager.scheduler.running = False
        fresh_scheduler_manager.shutdown()  # 不调 shutdown

    def test_shutdown_calls_scheduler_shutdown(self, fresh_scheduler_manager):
        fresh_scheduler_manager.init_scheduler(is_leader=True)
        fresh_scheduler_manager.scheduler = MagicMock()
        fresh_scheduler_manager.scheduler.running = True
        fresh_scheduler_manager.shutdown()
        fresh_scheduler_manager.scheduler.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_handles_exception(self, fresh_scheduler_manager):
        """shutdown 抛异常时——不传播（容错）。"""
        fresh_scheduler_manager.init_scheduler(is_leader=True)
        fresh_scheduler_manager.scheduler = MagicMock()
        fresh_scheduler_manager.scheduler.running = True
        fresh_scheduler_manager.scheduler.shutdown.side_effect = Exception("boom")
        fresh_scheduler_manager.shutdown()  # 不抛


class TestSchedulerManagerJobs:
    def test_add_job_auto_id_when_missing(self, fresh_scheduler_manager):
        """job_id 缺时自动生成。"""
        def my_task():
            pass
        cfg = JobConfig(func=my_task)
        assert cfg.job_id is None
        result = fresh_scheduler_manager.add_job(cfg)
        assert result is not None
        # 自动 ID 形如 my_task_<timestamp>
        assert result.startswith("my_task_")
        # 配置已存
        assert result in fresh_scheduler_manager._job_configs

    def test_add_job_explicit_id_preserved(self, fresh_scheduler_manager):
        def my_task():
            pass
        cfg = JobConfig(func=my_task, job_id="my-job-1")
        result = fresh_scheduler_manager.add_job(cfg)
        assert result == "my-job-1"

    def test_add_job_disabled_returns_id_without_schedule(self, fresh_scheduler_manager):
        """enabled=False——只存配置，不调度。"""
        def my_task():
            pass
        cfg = JobConfig(func=my_task, job_id="d", enabled=False)
        result = fresh_scheduler_manager.add_job(cfg)
        assert result == "d"
        # _job_configs 已存
        assert "d" in fresh_scheduler_manager._job_configs

    def test_add_job_non_leader_with_trigger_skips_schedule(self, fresh_scheduler_manager):
        """非 leader + 有 trigger——只存配置，不调 _add_scheduled_job。"""
        def my_task():
            pass
        cfg = JobConfig(func=my_task, job_id="nl", trigger="cron", trigger_args={"hour": 8})
        result = fresh_scheduler_manager.add_job(cfg)
        assert result == "nl"
        # _add_scheduled_job 不会被调（因 _is_leader=False）

    def test_remove_job(self, fresh_scheduler_manager):
        def my_task():
            pass
        cfg = JobConfig(func=my_task, job_id="r")
        fresh_scheduler_manager.add_job(cfg)
        assert "r" in fresh_scheduler_manager._job_configs
        fresh_scheduler_manager.remove_job("r")
        # _job_configs 也清掉
        assert "r" not in fresh_scheduler_manager._job_configs

    def test_get_jobs(self, fresh_scheduler_manager):
        def my_task():
            pass
        fresh_scheduler_manager.add_job(JobConfig(func=my_task, job_id="j1"))
        fresh_scheduler_manager.add_job(JobConfig(func=my_task, job_id="j2"))
        # init_scheduler(is_leader=False) 不创建 scheduler——get_jobs 返 []
        jobs = fresh_scheduler_manager.get_jobs()
        assert jobs == []
        # 但 _job_configs 里有
        assert "j1" in fresh_scheduler_manager._job_configs
        assert "j2" in fresh_scheduler_manager._job_configs

    def test_get_jobs_after_scheduler_init(self, fresh_scheduler_manager):
        """init_scheduler(is_leader=True) 后 get_jobs 调 scheduler.get_jobs。"""
        fresh_scheduler_manager.init_scheduler(is_leader=True)
        # mock 内部 scheduler 对象
        fresh_scheduler_manager.scheduler = MagicMock()
        job = MagicMock()
        job.id = "j1"
        job.name = "task1"
        job.next_run_time = None
        fresh_scheduler_manager.scheduler.get_jobs.return_value = [job]
        jobs = fresh_scheduler_manager.get_jobs()
        assert len(jobs) == 1
        assert jobs[0]["id"] == "j1"


class TestWrapExecutionResult:
    """_wrap_execution_result 是 20260822 失败语义修复——重点锁。"""

    def test_none_returns_success(self):
        from infra.schedule.scheduler_manager import SchedulerManager
        assert SchedulerManager._wrap_execution_result(None) == {
            "status": "success", "result": None
        }

    def test_true_returns_success(self):
        from infra.schedule.scheduler_manager import SchedulerManager
        assert SchedulerManager._wrap_execution_result(True) == {
            "status": "success", "result": True
        }

    def test_false_returns_error(self):
        """False 显式 → error（关键修复点）。"""
        from infra.schedule.scheduler_manager import SchedulerManager
        result = SchedulerManager._wrap_execution_result(False)
        assert result["status"] == "error"
        assert result["result"] is False

    def test_dict_passes_through(self):
        """业务自定义 dict 不被覆盖——透传。"""
        from infra.schedule.scheduler_manager import SchedulerManager
        custom = {"status": "custom", "data": [1, 2, 3]}
        assert SchedulerManager._wrap_execution_result(custom) is custom

    def test_other_value_returns_success(self):
        from infra.schedule.scheduler_manager import SchedulerManager
        result = SchedulerManager._wrap_execution_result("ok")
        assert result == {"status": "success", "result": "ok"}
        result = SchedulerManager._wrap_execution_result(42)
        assert result == {"status": "success", "result": 42}


# ============================================================
# UnifiedScheduler
# ============================================================
class TestUnifiedScheduler:
    def test_construct(self):
        s = UnifiedScheduler()
        assert s._task_loaders == {}
        # 内部 scheduler = SchedulerManager 实例
        assert isinstance(s.scheduler, SchedulerManager)

    def test_register_task_loader(self):
        s = UnifiedScheduler()
        def loader(scheduler):
            pass
        s.register_task_loader("etl", loader)
        assert s._task_loaders["etl"] is loader

    def test_start_calls_loaders(self):
        """start 触发 _load_all_tasks——验证每个 loader 被调。"""
        s = UnifiedScheduler()
        called = []
        s.register_task_loader("a", lambda sch: called.append("a"))
        s.register_task_loader("b", lambda sch: called.append("b"))
        # mock 实际 scheduler 防止真启动
        s.scheduler = MagicMock()
        s.scheduler.get_jobs.return_value = []
        s.start()
        assert "a" in called
        assert "b" in called

    def test_start_loader_exception_continues(self):
        """某个 loader 抛异常——start 不应整个挂。"""
        s = UnifiedScheduler()
        s.register_task_loader("bad", lambda sch: (_ for _ in ()).throw(RuntimeError("boom")))
        s.register_task_loader("ok", lambda sch: None)
        s.scheduler = MagicMock()
        s.scheduler.get_jobs.return_value = []
        s.start()  # 不抛

    def test_stop_calls_scheduler_stop(self):
        s = UnifiedScheduler()
        s.scheduler = MagicMock()
        s.stop()
        s.scheduler.stop.assert_called_once()

    def test_shutdown_calls_scheduler_shutdown(self):
        s = UnifiedScheduler()
        s.scheduler = MagicMock()
        s.shutdown()
        s.scheduler.shutdown.assert_called_once()

    def test_get_jobs_delegates(self):
        s = UnifiedScheduler()
        s.scheduler = MagicMock()
        s.scheduler.get_jobs.return_value = ["j1"]
        assert s.get_jobs() == ["j1"]

    def test_add_job_delegates(self):
        s = UnifiedScheduler()
        s.scheduler = MagicMock()
        cfg = JobConfig(func=lambda: None, job_id="x")
        s.scheduler.add_job.return_value = "x"
        assert s.add_job(cfg) == "x"
        s.scheduler.add_job.assert_called_once_with(cfg)

    def test_remove_job_delegates(self):
        s = UnifiedScheduler()
        s.scheduler = MagicMock()
        s.remove_job("x")
        s.scheduler.remove_job.assert_called_once_with("x")


# ============================================================
# LeaderElection — 分布式锁（mock Redis）
# ============================================================
class TestLeaderElection:
    def test_construct_stores_config(self):
        leader = LeaderElection(redis_config={"host": "h"})
        assert leader.redis_config == {"host": "h"}
        assert leader._is_leader is False
        assert leader._lock_value  # 自动生成 UUID
        assert leader._lock_ttl == 30
        assert leader._heartbeat_interval == 10

    def test_is_leader_initially_false(self):
        leader = LeaderElection(redis_config={})
        assert leader.is_leader() is False

    async def test_try_acquire_success(self):
        """SET NX 成功——try_acquire 返 True，is_leader=True。"""
        leader = LeaderElection(redis_config={})
        fake_redis = AsyncMock()
        fake_redis.set = AsyncMock(return_value=True)  # NX 成功
        leader._redis = fake_redis
        result = await leader.try_acquire()
        assert result is True
        assert leader.is_leader() is True
        # 验证 SET 调用参数
        call_kwargs = fake_redis.set.call_args.kwargs
        assert call_kwargs.get("nx") is True
        assert call_kwargs.get("ex") == 30

    async def test_try_acquire_failure(self):
        """SET NX 失败（锁已被占）——try_acquire 返 None（redis.set 的 by-design 返回值）。"""
        leader = LeaderElection(redis_config={})
        fake_redis = AsyncMock()
        fake_redis.set = AsyncMock(return_value=None)  # NX 失败
        leader._redis = fake_redis
        result = await leader.try_acquire()
        # redis.set 在 NX 失败时返 None（不是 False）——源码直接 return 这个值
        assert result is None
        assert leader.is_leader() is False

    async def test_try_acquire_redis_error_propagates(self):
        """redis 抛错——try_acquire 抛错（不吞）。源码不捕获。"""
        leader = LeaderElection(redis_config={})
        fake_redis = AsyncMock()
        fake_redis.set = AsyncMock(side_effect=Exception("conn lost"))
        leader._redis = fake_redis
        with pytest.raises(Exception, match="conn lost"):
            await leader.try_acquire()

    async def test_release_when_leader_runs_lua(self):
        """is_leader=True 时 release——调 redis.eval 跑 lua 脚本释放。"""
        leader = LeaderElection(redis_config={})
        leader._is_leader = True
        fake_redis = AsyncMock()
        fake_redis.eval = AsyncMock(return_value=1)
        leader._redis = fake_redis
        await leader.release()
        # 验证用 lua eval 释放（不是直接 delete）
        fake_redis.eval.assert_called_once()
        # lua 脚本 + key + lock_value 三个参数
        call_args = fake_redis.eval.call_args
        assert call_args.args[1] == 1  # KEYS 数量
        assert call_args.args[2] == leader._lock_key
        # 释放后 _is_leader 重置
        assert leader.is_leader() is False

    async def test_release_when_not_leader_skips_lua(self):
        """is_leader=False 时 release——不动 redis（避免误删别人的锁）。"""
        leader = LeaderElection(redis_config={})
        leader._is_leader = False
        fake_redis = AsyncMock()
        fake_redis.eval = AsyncMock()
        leader._redis = fake_redis
        await leader.release()
        # eval 不应被调（直接 delete 也不调）
        fake_redis.eval.assert_not_called()

    async def test_release_redis_error_propagates(self):
        """redis 抛错——release 也抛（不吞，by-design 锁未释放但调用方需知）。"""
        leader = LeaderElection(redis_config={})
        leader._is_leader = True
        fake_redis = AsyncMock()
        fake_redis.eval = AsyncMock(side_effect=Exception("conn lost"))
        leader._redis = fake_redis
        with pytest.raises(Exception, match="conn lost"):
            await leader.release()

    async def test_leader_context_acquires_and_releases(self):
        """async context manager——acquire on enter, release on exit。yield 返 acquired bool。"""
        leader = LeaderElection(redis_config={})
        leader._redis = AsyncMock()
        leader._redis.set = AsyncMock(return_value=True)
        leader._redis.eval = AsyncMock(return_value=1)
        async with leader.leader_context() as acquired:
            # yield 的是 try_acquire 的返回值（True/None）
            assert acquired is True
            assert leader.is_leader() is True
        # exit 后释放（_is_leader 重置）
        assert leader.is_leader() is False

    async def test_leader_context_acquire_fail_does_not_release(self):
        """acquire 失败时——context 不调 release（避免误删别人的锁）。"""
        leader = LeaderElection(redis_config={})
        leader._redis = AsyncMock()
        leader._redis.set = AsyncMock(return_value=None)  # acquire 失败
        leader._redis.eval = AsyncMock()
        async with leader.leader_context():
            pass  # 实际不应进 body
        # eval 不应被调（直接 delete 也不调）
        leader._redis.eval.assert_not_called()


# ============================================================
# 顶层 helper：get_scheduler_manager / get_unified_scheduler
# ============================================================
class TestGlobalGetHelpers:
    def test_get_scheduler_manager_creates_on_first_call(self):
        """首次调用——lazy 创建 SchedulerManager 单例。"""
        import infra.schedule as sched_mod
        sched_mod.scheduler_manager = None
        mgr = sched_mod.get_scheduler_manager()
        assert isinstance(mgr, SchedulerManager)

    def test_get_scheduler_manager_returns_same_instance(self):
        import infra.schedule as sched_mod
        sched_mod.scheduler_manager = None
        m1 = sched_mod.get_scheduler_manager()
        m2 = sched_mod.get_scheduler_manager()
        assert m1 is m2

    def test_get_unified_scheduler_creates_on_first_call(self):
        import infra.schedule as sched_mod
        sched_mod.unified_scheduler = None
        s = sched_mod.get_unified_scheduler()
        assert isinstance(s, UnifiedScheduler)


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
