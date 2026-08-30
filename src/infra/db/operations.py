#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异步数据库操作装饰器和工具函数
"""

import time
import functools
from typing import Callable, Optional
from sqlalchemy.exc import SQLAlchemyError

from infra.db.monitor import record_db_query
from infra.logger import get_logger
from infra.exceptions import DatabaseError

logger = get_logger(__name__)


def db_operation(
    operation_name: str = "database_operation",
    log_performance: bool = True,
    db_key: Optional[str] = None,
):
    """
    异步数据库操作装饰器

    Args:
        operation_name: 操作名称（日志/异常里用）
        log_performance: 是否记录性能指标
        db_key: db manager key（业务库 "db" / 时序库 "timescaledb" 等），
                None 时从实例属性 self.db_key 动态获取
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> None:
            start_time = time.time()
            success = False

            # 动态获取 db_key：优先装饰器参数，其次实例属性
            actual_db_key = db_key if db_key is not None else getattr(args[0], "db_key", "db")

            try:
                # 直接执行异步函数
                result = await func(*args, **kwargs)
                success = True
                return result  # type: ignore[no-any-return]

            except SQLAlchemyError as e:
                logger.error(f"数据库操作失败 [{operation_name}]: {e}")
                raise DatabaseError(
                    f"数据库操作失败: {e}",
                    operation=operation_name,
                    details={'function': func.__name__, 'db_key': actual_db_key}
                )
            except Exception as e:
                logger.error(f"未知错误 [{operation_name}]: {e}")
                raise
            finally:
                if log_performance:
                    response_time = time.time() - start_time
                    record_db_query(response_time, success)

                    if response_time > 1.0:
                        logger.warning(f"数据库操作耗时较长 [{operation_name}]: {response_time:.2f}s")

        return wrapper
    return decorator
