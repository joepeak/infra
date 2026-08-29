from sqlalchemy import String, Numeric, BigInteger, DateTime, func, Index, Date, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.types import TypeDecorator
from datetime import datetime, timezone, timedelta, date
from typing import Optional
from zoneinfo import ZoneInfo
from infra.data_quality import DataQuality

class JSONBCompatible(TypeDecorator):
    """支持 PostgreSQL JSONB 和 SQLite JSON 的兼容类型"""
    impl = JSON
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())
    
    def _coerce_compared_value(self, op, value):
        return self.impl._coerce_compared_value(op, value)

# 1. ORM 根基类
class Base(DeclarativeBase):
    pass

# 2. 通用基础模型 (抽象类：不生成实际物理表)
class BaseModel(Base):
    __abstract__ = True
    
    # 所有业务表都可以继承的元数据字段
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        comment="记录创建时间 (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="记录更新时间 (UTC)"
    )

# 3. 观测数据公共基类 (抽象类)
class ObservationBaseModel(BaseModel):
    __abstract__ = True
    
    # 统一业务日期字段，所有时序表共用
    stat_date: Mapped[date] = mapped_column(
        Date, nullable=False,
        comment="业务统计日期 (UTC)"
    )
    # 数据质量字段
    data_quality: Mapped[DataQuality] = mapped_column(
        String(20), default=DataQuality.UNKNOWN,
        comment="数据质量状态 (UNKNOWN/GOOD/DELAYED/STALE/PARTIAL/SUSPICIOUS/INVALID/MISSING)"
    )
    quality_reason: Mapped[Optional[str]] = mapped_column(
        String(512), nullable=True,
        comment="质量问题简要原因"
    )
    quality_checked_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True,
        comment="最近一次质量检查时间"
    )
    quality_info: Mapped[Optional[dict]] = mapped_column(
        JSONBCompatible, nullable=True,
        comment="质量诊断详情 (JSONB)"
    )

# 4. 统一时间序列基础模型 (抽象类)
class TimeSeriesBaseModel(ObservationBaseModel):
    __abstract__ = True
    
    # 核心设计：统一所有时序表的时间字段名为 event_time
    # 各模型自行定义复合主键，基类不声明 primary_key
    event_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        comment="统一事件发生时间戳 (UTC)"
    )

# 5. 统一观测数据抽象基类 (抽象类，不生成物理表)
class ObservationBase(TimeSeriesBaseModel):
    __abstract__ = True
    
    # 系统统一指标身份
    indicator_id: Mapped[str] = mapped_column(
        String(64), index=True,
        comment="系统统一指标 ID"
    )
    data_source: Mapped[str] = mapped_column(
        String(32),
        comment="数据来源: fred/binance/mt5/calculated"
    )
    source_identifier: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
        comment="外部数据源中的指标身份"
    )

class TimeUtil:
    @staticmethod
    def to_utc_event_time(val: str | datetime | date, source_tz: str = "UTC") -> datetime:
        """
        统一将各种输入格式（字符串、datetime、date）和各种源头时区转为标准 UTC datetime
        """
        if isinstance(val, str):
            # 处理仅有年月日的情况（如 FRED 宏观数据的 "2026-08-01"）
            if len(val) == 10:
                dt = datetime.strptime(val, "%Y-%m-%d")
            else:
                dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
        elif isinstance(val, date) and not isinstance(val, datetime):
            # date 对象转为 datetime
            dt = datetime.combine(val, datetime.min.time())
        else:
            dt = val

        # 如果没有时区信息，加上源头时区信息
        if dt.tzinfo is None:
            if source_tz == "UTC":
                dt = dt.replace(tzinfo=timezone.utc)
            elif source_tz == "Asia/Shanghai": # A股/中国宏观
                dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
            else: # 美股等 (America/New_York)
                dt = dt.replace(tzinfo=ZoneInfo(source_tz))

        # 统一转换为 UTC 时区
        return dt.astimezone(timezone.utc)


__all__ = [
    'Base',
    'BaseModel',
    'ObservationBaseModel',
    'TimeSeriesBaseModel',
    'ObservationBase',
    'TimeUtil',
    'JSONBCompatible',
]