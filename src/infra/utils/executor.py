#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的异步任务执行器，支持重试、并发、超时控制
"""

import asyncio
from functools import partial
from typing import List, Optional, Callable, Any, Dict, TypeVar, Tuple

from infra.utils.retry import RetryExecutor, RetryConfig
from infra.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class TaskExecutor:
    """统一的异步任务执行器"""
    
    def __init__(self, default_retry_config: Optional[RetryConfig] = None) -> None:
        self.default_retry_config = default_retry_config or RetryConfig.default()
    
    async def execute(self, func: Callable, *args: Any, retry_config: Optional[RetryConfig] = None, **kwargs: Any) -> Any:
        retry_config = retry_config or self.default_retry_config
        retry_executor = RetryExecutor(retry_config)
        
        if asyncio.iscoroutinefunction(func):
            return await retry_executor.execute_async(func, *args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, partial(retry_executor.execute_sync, func, *args, **kwargs)
            )


async def execute_task(func: Callable, *args: Any, retry_config: Optional[RetryConfig] = None, **kwargs: Any) -> Any:
    executor = TaskExecutor(retry_config)
    return await executor.execute(func, *args, **kwargs)