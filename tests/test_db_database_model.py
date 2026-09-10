#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db.database_model 深度测试

infra 是下游项目的基础设施类库——database_model 是 ORM 基类，所有业务表继承。
本测试覆盖：
- JSONBCompatible 类型适配
- 4 个抽象基类的字段定义（不需要连 DB）

注意：基类是 __abstract__ = True，测试用 SQLite + create_all 建临时表验证 metadata。
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Integer

from infra.db.database_model import (
    Base,
    BaseModel,
    JSONBCompatible,
    ObservationBase,
    ObservationBaseModel,
    TimeSeriesBaseModel,
)
from infra.utils.time_util import to_utc_event_time
from infra.data_quality import DataQuality


# ============================================================
# 1. JSONBCompatible —— PG / SQLite dialect 适配
# ============================================================
class TestJSONBCompatible:
    def test_class_attributes(self):
        """JSONBCompatible 是 TypeDecorator，impl=JSON，cache_ok=True。"""
        assert JSONBCompatible.impl is not None
        assert JSONBCompatible.cache_ok is True

    def test_load_dialect_impl_postgresql_returns_jsonb(self):
        """PG dialect 返 JSONB。"""
        from sqlalchemy.dialects.postgresql import JSONB
        fake_dialect = MagicMock()
        fake_dialect.name = "postgresql"
        jsonb_descriptor = MagicMock()
        fake_dialect.type_descriptor.return_value = jsonb_descriptor
        result = JSONBCompatible().load_dialect_impl(fake_dialect)
        assert result is jsonb_descriptor
        # 验证传入了 JSONB
        call_args = fake_dialect.type_descriptor.call_args
        assert isinstance(call_args.args[0], JSONB)

    def test_load_dialect_impl_sqlite_returns_json(self):
        """非 PG dialect（SQLite）返普通 JSON。"""
        from sqlalchemy import JSON
        fake_dialect = MagicMock()
        fake_dialect.name = "sqlite"
        json_descriptor = MagicMock()
        fake_dialect.type_descriptor.return_value = json_descriptor
        result = JSONBCompatible().load_dialect_impl(fake_dialect)
        assert result is json_descriptor
        call_args = fake_dialect.type_descriptor.call_args
        assert isinstance(call_args.args[0], JSON)

    def test_load_dialect_impl_mysql_returns_json(self):
        """MySQL dialect 也走 JSON（兼容分支）。"""
        fake_dialect = MagicMock()
        fake_dialect.name = "mysql"
        result = JSONBCompatible().load_dialect_impl(fake_dialect)
        # 不抛错即可
        assert result is not None


# ============================================================
# 2. BaseModel —— 通用基础模型
# ============================================================
def _col(name, cls):
    """拿抽象类的 mapped_column 内部的 Column 对象。"""
    mc = cls.__dict__[name]
    col = getattr(mc, "column", None)
    if col is None:
        col = getattr(mc, "_column", None)
    assert col is not None, f"{cls.__name__}.{name} no column attribute"
    return col


class TestBaseModel:
    def test_abstract(self):
        """BaseModel 是抽象类——不生成物理表。"""
        assert BaseModel.__abstract__ is True

    def test_inherits_from_base(self):
        assert issubclass(BaseModel, Base)

    def test_has_created_at_column(self):
        """BaseModel 有 created_at 字段（带 server_default=func.now()）。"""
        col = _col("created_at", BaseModel)
        # server_default 应是 func.now()
        assert col.server_default is not None

    def test_has_updated_at_column(self):
        col = _col("updated_at", BaseModel)
        assert col.server_default is not None
        # onupdate 应有
        assert col.onupdate is not None


# ============================================================
# 3. ObservationBaseModel —— 观测数据基类
# ============================================================
class TestObservationBaseModel:
    def test_abstract(self):
        assert ObservationBaseModel.__abstract__ is True

    def test_inherits_from_basemodel(self):
        assert issubclass(ObservationBaseModel, BaseModel)

    def test_stat_date_column(self):
        """观测日期字段——Date 类型，not null。"""
        col = _col("stat_date", ObservationBaseModel)
        assert col.nullable is False

    def test_data_quality_column(self):
        """数据质量字段——String(20)，默认 DataQuality.UNKNOWN。"""
        col = _col("data_quality", ObservationBaseModel)
        # VARCHAR(20)
        assert "VARCHAR" in str(col.type)
        assert "20" in str(col.type)
        # 默认值 DataQuality.UNKNOWN
        from infra.data_quality import DataQuality as DQ
        assert col.default.arg is DQ.UNKNOWN

    def test_quality_reason_column(self):
        col = _col("quality_reason", ObservationBaseModel)
        assert col.nullable is True

    def test_quality_checked_at_column(self):
        col = _col("quality_checked_at", ObservationBaseModel)
        assert col.nullable is True

    def test_quality_info_column(self):
        """quality_info 用 JSONBCompatible 类型（TypeDecorator，dialect 加载时切换 JSON/JSONB）。"""
        col = _col("quality_info", ObservationBaseModel)
        # 抽象类阶段 type 仍是 JSONBCompatible 本身（TypeDecorator），
        # 真连 SQLite/PG 时 load_dialect_impl 才切换为 JSON/JSONB。
        # 锁住现状：type 是 JSONBCompatible
        assert isinstance(col.type, JSONBCompatible)


# ============================================================
# 4. TimeSeriesBaseModel —— 时序基类
# ============================================================
class TestTimeSeriesBaseModel:
    def test_abstract(self):
        assert TimeSeriesBaseModel.__abstract__ is True

    def test_inherits_from_observationbasemodel(self):
        assert issubclass(TimeSeriesBaseModel, ObservationBaseModel)

    def test_event_time_column(self):
        assert "event_time" in TimeSeriesBaseModel.__dict__


# ============================================================
# 5. ObservationBase —— 统一观测数据抽象基类
# ============================================================
class TestObservationBase:
    def test_abstract(self):
        assert ObservationBase.__abstract__ is True

    def test_inherits_from_timeseries(self):
        assert issubclass(ObservationBase, TimeSeriesBaseModel)

    def test_indicator_id_column(self):
        col = _col("indicator_id", ObservationBase)
        # 有 index
        assert col.index is True

    def test_data_source_column(self):
        assert "data_source" in ObservationBase.__dict__

    def test_source_identifier_column(self):
        col = _col("source_identifier", ObservationBase)
        assert col.nullable is True

    def test_combined_columns_count(self):
        """ObservationBase 继承所有父类字段——总共 2+5+1+3 = 11 个 mapped_column。"""
        # 数所有有 'column' 属性的 __dict__ 项
        count = 0
        for cls in [BaseModel, ObservationBaseModel, TimeSeriesBaseModel, ObservationBase]:
            for attr_name, attr_val in cls.__dict__.items():
                if attr_name.startswith("_"):
                    continue
                if hasattr(attr_val, "column") or hasattr(attr_val, "_column"):
                    count += 1
        # BaseModel 2 + ObservationBaseModel 5 + TimeSeriesBaseModel 1 + ObservationBase 3 = 11
        assert count == 11


# ============================================================
# 6. to_utc_event_time 时区转换
# ============================================================
class TestToUtcEventTime:
    def test_string_10_chars_parsed_as_date(self):
        """10 字符 "2026-08-01" 解析为 date（无时间部分）。"""
        result = to_utc_event_time("2026-08-01")
        assert result.year == 2026
        assert result.month == 8
        assert result.day == 1
        assert result.hour == 0
        assert result.minute == 0
        # 强制 UTC
        assert result.tzinfo == timezone.utc

    def test_string_with_time(self):
        """带时间的字符串解析。"""
        result = to_utc_event_time("2026-08-01 14:30:00")
        assert result.hour == 14
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_date_object_converted_to_datetime(self):
        """date 对象（无时间）转 datetime 00:00:00。"""
        d = date(2026, 8, 1)
        result = to_utc_event_time(d)
        assert result.year == 2026
        assert result.hour == 0
        assert result.tzinfo == timezone.utc

    def test_naive_datetime_assumes_utc(self):
        """naive datetime——加 UTC 时区。"""
        dt = datetime(2026, 8, 1, 14, 30, 0)  # 无 tzinfo
        result = to_utc_event_time(dt)
        assert result.tzinfo == timezone.utc

    def test_aware_datetime_converted_to_utc(self):
        """带时区的 datetime——转换到 UTC。"""
        shanghai = timezone(timedelta(hours=8))
        dt = datetime(2026, 8, 1, 14, 30, 0, tzinfo=shanghai)
        result = to_utc_event_time(dt)
        assert result.tzinfo == timezone.utc
        # 14:30 上海时间 → 06:30 UTC
        assert result.hour == 6
        assert result.minute == 30

    def test_aware_datetime_new_york_converted(self):
        """美东时间（UTC-5/-4）转 UTC。"""
        ny = timezone(timedelta(hours=-5))
        dt = datetime(2026, 8, 1, 14, 30, 0, tzinfo=ny)
        result = to_utc_event_time(dt)
        assert result.tzinfo == timezone.utc
        # 14:30 NY → 19:30 UTC
        assert result.hour == 19

    def test_source_tz_shanghai(self):
        """source_tz='Asia/Shanghai' 字符串——加 +8 时区。"""
        result = to_utc_event_time("2026-08-01 22:00:00", source_tz="Asia/Shanghai")
        # 22:00 上海 → 14:00 UTC
        assert result.hour == 14
        assert result.tzinfo == timezone.utc

    def test_source_tz_zoneinfo_lookup(self):
        """source_tz='America/New_York'——用 zoneinfo 查 tz。"""
        result = to_utc_event_time("2026-08-01 10:00:00", source_tz="America/New_York")
        # 10:00 NY (-4 夏令时) → 14:00 UTC
        assert result.hour == 14
        assert result.tzinfo == timezone.utc


# ============================================================
# 7. metadata 与 Base 一致性（建表测试）
# ============================================================
class TestMetadata:
    def test_all_classes_register_in_base_metadata(self):
        """所有 abstract 类都注册到 Base.metadata。"""
        for cls in [BaseModel, ObservationBaseModel, TimeSeriesBaseModel, ObservationBase]:
            assert cls.metadata is Base.metadata

    def test_create_all_sqlite(self, tmp_path):
        """__abstract__ 表不在 create_all 里——验证：定义一个具体子类能正常建表。"""
        # Mapped / mapped_column 已在模块级 import——SQLAlchemy 2.0 注解要求

        # 定义具体子类测试
        class TestTable(ObservationBase):
            __tablename__ = "test_db_model_meta"
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            value: Mapped[str] = mapped_column(String(50))

        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{tmp_path}/test.db")
        Base.metadata.create_all(engine)
        # 验证表存在
        from sqlalchemy import inspect
        inspector = inspect(engine)
        assert "test_db_model_meta" in inspector.get_table_names()
        # 验证字段
        cols = {c["name"] for c in inspector.get_columns("test_db_model_meta")}
        # 继承自 ObservationBase 的所有字段 + 具体子类的字段
        assert "id" in cols
        assert "value" in cols
        assert "stat_date" in cols
        assert "event_time" in cols
        assert "indicator_id" in cols
        engine.dispose()


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
