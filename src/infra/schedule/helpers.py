# infra/schedule/helpers.py

import asyncio
import concurrent.futures
from typing import Any, Optional

from infra.logger import get_logger

logger = get_logger(__name__)

# 使用全局变量（在同一个事件循环中共享）
_task_queue: Optional[asyncio.Queue] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def init_task_queue(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
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


def submit_task(coro: Any) -> "concurrent.futures.Future[Any]":
    """线程安全地提交协程到主事件循环的任务队列。

    Args:
        coro: 要执行的协程对象

    Returns:
        future——enqueue 完成后的 future（用于等待入队结果）

    Note:
        **调用方负责保证协程最终被 await**——本函数只把协程放进队列，
        队列的消费者负责取出并 await。若没有消费者，协程会一直悬在
        队列里（产生 "coroutine was never awaited" 警告且任务静默丢失）。

        这里不吞异常：enqueue 失败（如主 loop 已关闭）时**关闭协程并抛出**，
        避免「协程既没执行也没 await」的悬空状态。
    """
    logger.debug(f"📤 submit_task 被调用, coro={coro}")
    queue = get_task_queue()
    loop = get_main_loop()

    async def _enqueue() -> None:
        await queue.put(coro)

    try:
        future = asyncio.run_coroutine_threadsafe(_enqueue(), loop)
    except Exception:
        # 主 loop 已关闭等情况——协程永远不会被 await，
        # 必须显式关闭它，否则产生 never-awaited 警告
        _close_coro(coro)
        raise

    def _on_done(f: "concurrent.futures.Future[Any]") -> None:
        """enqueue 失败时关闭协程——避免悬空。"""
        try:
            exc = f.exception()
        except Exception:
            exc = None
        if exc is not None:
            logger.error(f"任务入队失败: {type(exc).__name__}: {exc}")
            _close_coro(coro)

    future.add_done_callback(_on_done)
    return future


def _close_coro(coro: Any) -> None:
    """关闭一个未被 await 的协程——避免 "never awaited" 警告。"""
    closer = getattr(coro, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def is_task_queue_ready() -> bool:
    """检查任务队列是否已就绪"""
    return _task_queue is not None and _main_loop is not None