#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异步数据库操作装饰器和工具函数
"""

import time
import functools
from typing import Callable, Optional
from sqlalchemy.exc import SQLAlchemyError

from infra.db.connection_manager import (
    get_business_session,
    get_timeseries_session,
)
from infra.db.monitor import record_db_query
from infra.logger import get_logger
from infra.exceptions import DatabaseError

logger = get_logger(__name__)


def db_operation(
    operation_name: str = "database_operation",
    log_performance: bool = True,
    db_type: Optional[str] = None,
):
    """
    异步数据库操作装饰器
    
    Args:
        operation_name: 操作名称
        log_performance: 是否记录性能指标
        db_type: 数据库类型，"business" 或 "timeseries"
                 为 None 时从实例属性 self.db_type 动态获取
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            
            # 动态获取 db_type：优先使用装饰器参数，其次从实例属性获取
            actual_db_type = db_type if db_type is not None else args[0].db_type
            
            try:
                # 直接执行异步函数
                result = await func(*args, **kwargs)
                success = True
                return result
                
            except SQLAlchemyError as e:
                logger.error(f"数据库操作失败 [{operation_name}]: {e}")
                raise DatabaseError(
                    f"数据库操作失败: {e}",
                    operation=operation_name,
                    details={'function': func.__name__, 'db_type': actual_db_type}
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

