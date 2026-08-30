"""infra - 通用基础设施类库（自用，不公开）。

模块：
- infra.db：连接管理/ORM 基类/Repository 基类（按 key Map 化）
- infra.config：YAML 配置加载（set_config deep merge + replace_config）
- infra.logger：日志
- infra.exceptions：异常类
- infra.time_util：时区转换工具（@deprecated，请用 infra.utils.time_util）
- infra.data_quality：DataQuality 枚举
- infra.bootstrap：bootstrap_all() 一键启动所有基础设施
- infra.utils：retry/lark/wechat/async_helpers/executor/trend_analyzer/common/time_util
- infra.redis：Redis 客户端
- infra.mq：消息队列（基于 Redis Stream）
- infra.messages：通用消息定义（TaskMessage dataclass）
- infra.schedule：分布式任务调度（APScheduler + Redis 分布式锁）
- infra.script_logger：独立脚本日志（控制台+文件）
"""
# 重新导出常用符号（向后兼容）
# 20260829 refactor: TimeUtil 类改函数式——优先用 infra.utils.time_util.to_utc_event_time
from infra.utils.time_util import to_utc_event_time
# 旧路径保留（@deprecated）——通过 shim
from infra.time_util import TimeUtil  # noqa: F401  # 兼容旧 import
from infra.data_quality import DataQuality
from infra.logger import get_logger
from infra.exceptions import DatabaseError
from infra.config import init_config, get_config
from infra.db import (
    init_db_manager,
    get_db_manager,
    get_db_session,
    get_db_health,
    list_db_keys,
    close_all_db_connections,
    DatabaseRepository, Base, BaseModel, TimeSeriesBaseModel,
)
from infra.redis import init_redis, close_redis, get_redis
# 消息定义 + MQ + 调度器（最近迁移自 crypto-watcher/core，深度测试覆盖）
from infra.messages import TaskMessage, create_message
from infra.mq import (
    init_mq,
    get_mq,
    RedisStreamMQ,
    MessagePriority,
    QueueConfig,
    QueueName,
)
from infra.schedule import (
    SchedulerManager,
    UnifiedScheduler,
    TaskRegistry,
    get_scheduler_manager,
    get_unified_scheduler,
    JobConfig,
)
from infra.bootstrap import bootstrap_all, shutdown_bootstrap, load_config
from infra.script_logger import get_script_logger

__all__ = [
    # time_util（20260829 改函数式——旧类仍可访问但弃用）
    "to_utc_event_time",
    "TimeUtil",  # 兼容
    # data_quality
    "DataQuality",
    # logger
    "get_logger",
    # exceptions
    "DatabaseError",
    # config
    "init_config", "get_config",
    # db
    "init_db_manager",
    "get_db_manager", "get_db_session", "get_db_health",
    "list_db_keys", "close_all_db_connections",
    "DatabaseRepository", "Base", "BaseModel", "TimeSeriesBaseModel",
    # redis
    "init_redis", "close_redis", "get_redis",
    # messages（任务消息定义）
    "TaskMessage", "create_message",
    # mq（消息队列）
    "init_mq", "get_mq", "RedisStreamMQ",
    "MessagePriority", "QueueConfig", "QueueName",
    # schedule（分布式调度）
    "SchedulerManager", "UnifiedScheduler", "TaskRegistry",
    "get_scheduler_manager", "get_unified_scheduler", "JobConfig",
    # bootstrap
    "bootstrap_all", "shutdown_bootstrap", "load_config",
    # script_logger
    "get_script_logger",
]
