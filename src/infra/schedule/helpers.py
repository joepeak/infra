# infra/schedule/helpers.py

import asyncio
from typing import Optional

from infra.logger import get_logger

logger = get_logger(__name__)

# 使用全局变量（在同一个事件循环中共享）
_task_queue: Optional[asyncio.Queue] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def init_task_queue(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
    """初始化任务队列（必须在 worker 启动前调用）"""
    global _task_queue, _main_loop
    _task_queue = queue
    _main_loop = loop
    logger.info(f"任务队列已初始化, loop_id={id(loop)}, queue_id={id(queue)}")


def get_task_queue() -> asyncio.Queue:
    if _task_queue is None:
        raise RuntimeError("任务队列未初始化，请先调用 init_task_queue")
    return _task_queue


def get_main_loop() -> asyncio.AbstractEventLoop:
    if _main_loop is None:
        raise RuntimeError("主事件循环未初始化")
    return _main_loop


def submit_task(coro) -> asyncio.Future:
    """线程安全地提交协程到主事件循环的任务队列"""
    logger.debug(f"📤 submit_task 被调用, coro={coro}")
    queue = get_task_queue()
    loop = get_main_loop()

    async def _enqueue():
        await queue.put(coro)

    future = asyncio.run_coroutine_threadsafe(_enqueue(), loop)
    future.add_done_callback(lambda f: f.exception() if f.done() else None)
    return future


def is_task_queue_ready() -> bool:
    """检查任务队列是否已就绪"""
    return _task_queue is not None and _main_loop is not None