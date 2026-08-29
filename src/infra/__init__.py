"""infra - 通用基础设施类库（自用，不公开）。

模块：
- infra.db：连接管理/ORM 基类/Repository 基类
- infra.config：YAML 配置加载
- infra.logger：日志
- infra.exceptions：异常类
- infra.time_util：时区转换工具
- infra.data_quality：DataQuality 枚举
- infra.bootstrap：bootstrap_all() 一键启动所有基础设施
- infra.utils：retry/lark/wechat/async_helpers/executor/trend_analyzer/common
- infra.redis：Redis 客户端
- infra.mq：消息队列（占位）
"""
# 重新导出常用符号（向后兼容）
from infra.time_util import TimeUtil
from infra.data_quality import DataQuality
from infra.logger import get_logger
from infra.exceptions import DatabaseError
from infra.config import init_config, get_config
from infra.db import (
    init_db_manager, init_timeseries_db_manager,
    get_business_db_manager, get_business_session, get_business_health,
    get_timeseries_db_manager, get_timeseries_session, get_timeseries_health,
    close_all_db_connections,
    DatabaseRepository, Base, BaseModel, TimeSeriesBaseModel,
)
from infra.redis import init_redis, close_redis, get_redis
from infra.bootstrap import bootstrap_all, shutdown_bootstrap, load_config

__all__ = [
    # time_util
    "TimeUtil",
    # data_quality
    "DataQuality",
    # logger
    "get_logger",
    # exceptions
    "DatabaseError",
    # config
    "init_config", "get_config",
    # db
    "init_db_manager", "init_timeseries_db_manager",
    "get_business_db_manager", "get_business_session", "get_business_health",
    "get_timeseries_db_manager", "get_timeseries_session", "get_timeseries_health",
    "close_all_db_connections",
    "DatabaseRepository", "Base", "BaseModel", "TimeSeriesBaseModel",
    # redis
    "init_redis", "close_redis", "get_redis",
    # bootstrap
    "bootstrap_all", "shutdown_bootstrap", "load_config",
]
