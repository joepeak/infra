#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.schedule 模块 import 与核心类构造测试

搬运自 macro_monitor.core.schedule。
只验证能 import + 关键 dataclass/class 能构造——深度覆盖留待后续。
"""

from __future__ import annotations

import pytest


# ============================================================
# 1. infra.schedule 导入与核心符号
# ============================================================
class TestScheduleImport:
    def test_schedule_module_importable(self):
        """infra.schedule 能 import + 核心符号都在。"""
        from infra.schedule import (
            SchedulerManager,
            JobConfig,
            UnifiedScheduler,
            TaskRegistry,
            task_registry,
        )
        assert SchedulerManager is not None
        assert JobConfig is not None
        assert UnifiedScheduler is not None
        assert TaskRegistry is not None
        # 全局 task_registry 实例存在
        assert task_registry is not None

    def test_submodule_imports(self):
        """各子模块单独 import。"""
        # 子模块用 importlib 触发加载（__init__ 只 re-export 类，不 import 模块对象）
        import importlib
        helpers = importlib.import_module("infra.schedule.helpers")
        scheduler_manager = importlib.import_module("infra.schedule.scheduler_manager")
        tr = importlib.import_module("infra.schedule.task_registry")
        unified_scheduler = importlib.import_module("infra.schedule.unified_scheduler")
        leader = importlib.import_module("infra.schedule.leader")

        assert helpers is not None
        assert scheduler_manager is not None
        assert tr is not None
        assert unified_scheduler is not None
        assert leader is not None


# ============================================================
# 2. JobConfig dataclass 构造
# ============================================================
class TestJobConfig:
    def test_job_config_construct(self):
        """JobConfig dataclass 能构造（最小参数）。"""
        from infra.schedule import JobConfig

        def sample_func():
            pass

        cfg = JobConfig(func=sample_func)
        assert cfg.func is sample_func
        assert cfg.max_instances == 1
        assert cfg.misfire_grace_time == 300
        assert cfg.coalesce is True
        assert cfg.replace_existing is False
        assert cfg.run_immediately is False
        assert cfg.enabled is True

    def test_job_config_with_args(self):
        """JobConfig 带 args/kwargs 构造。"""
        from infra.schedule import JobConfig

        def sample_func(x, y=0):
            return x + y

        cfg = JobConfig(
            func=sample_func,
            trigger="cron",
            trigger_args={"hour": 8},
            job_id="job-1",
            name="test_job",
            args=(1,),
            kwargs={"y": 2},
        )
        assert cfg.trigger == "cron"
        assert cfg.trigger_args == {"hour": 8}
        assert cfg.job_id == "job-1"
        assert cfg.name == "test_job"
        assert cfg.args == (1,)
        assert cfg.kwargs == {"y": 2}


# ============================================================
# 3. TaskRegistry 类构造 + 增删
# ============================================================
class TestTaskRegistry:
    def test_task_registry_construct(self):
        """TaskRegistry 能实例化。"""
        from infra.schedule import TaskRegistry
        reg = TaskRegistry()
        assert reg is not None
        # 内部 dict 初始化
        assert reg._task_factories == {}
        assert reg._task_classes == {}

    def test_global_task_registry_exists(self):
        """模块级 task_registry 全局实例存在且是 TaskRegistry 类型。"""
        from infra.schedule import task_registry, TaskRegistry
        assert isinstance(task_registry, TaskRegistry)


# ============================================================
# 4. SchedulerManager 单例
# ============================================================
class TestSchedulerManager:
    def test_scheduler_manager_construct(self):
        """SchedulerManager 类能实例化（不启 APScheduler）。"""
        from infra.schedule import SchedulerManager
        # 注：每次实例化 _instance 都被覆盖为新实例（源码实现）
        mgr = SchedulerManager()
        assert mgr is not None


# ============================================================
# 5. UnifiedScheduler 构造
# ============================================================
class TestUnifiedScheduler:
    def test_unified_scheduler_construct(self):
        """UnifiedScheduler 类能实例化（不启动 scheduler）。"""
        from infra.schedule import UnifiedScheduler
        scheduler = UnifiedScheduler()
        assert scheduler is not None
        # 内部 task_loaders dict 初始化
        assert scheduler._task_loaders == {}


# ============================================================
# 6. LeaderElection 构造
# ============================================================
class TestLeaderElection:
    def test_leader_election_construct(self):
        """LeaderElection 类能实例化（不连真 redis）。"""
        from infra.schedule.leader import LeaderElection
        leader = LeaderElection(redis_config={"host": "localhost", "port": 6379})
        assert leader is not None
        assert leader.redis_config == {"host": "localhost", "port": 6379}
        assert leader._is_leader is False


# ============================================================
# 7. helpers 工具函数
# ============================================================
class TestHelpers:
    def test_init_task_queue(self):
        """init_task_queue / get_task_queue / get_main_loop 能调。"""
        import asyncio
        from infra.schedule.helpers import init_task_queue, get_task_queue, get_main_loop

        loop = asyncio.new_event_loop()
        try:
            queue = asyncio.Queue()
            init_task_queue(loop, queue)
            # 二次 get 应返同一对象
            assert get_task_queue() is queue
            assert get_main_loop() is loop
        finally:
            loop.close()

    def test_get_task_queue_without_init_raises(self):
        """未 init 时 get_task_queue 抛 RuntimeError。"""
        from infra.schedule import helpers
        # 强制重置 module-level 状态
        helpers._task_queue = None
        from infra.schedule.helpers import get_task_queue
        with pytest.raises(RuntimeError, match="任务队列未初始化"):
            get_task_queue()


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
