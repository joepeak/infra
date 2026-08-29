#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DatabaseRepository 基类完整测试

覆盖范围：
1. 基础 CRUD: create, get_by_id, update, delete, get_all, count
2. query 方法: where, conditions, 所有操作符, None 处理, 逻辑, 排序, 分页
3. 分页: paginate 边界情况
4. 批量操作: bulk_create, bulk_update, bulk_delete
5. 存在性: exists
6. 会话管理: get_session
7. 异常处理: 无效操作符, 无效字段
8. 综合场景: 模拟 FredRepository 使用

依赖：tests/conftest.py 已把 infra 包加进 sys.path，并提供 session 级 PG 临时库
+ business engine（DATABASE_URL 由 conftest 注入）。
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from infra.db.database_repository import DatabaseRepository, PageResult
from infra.db import get_business_db_manager
from infra.db.database_model import Base as _InfraBase


# ============================================================
# 继承 infra.db.database_model.Base 的测试模型
# ============================================================
class TestModel(_InfraBase):
    """测试 Repository 用的数据模型（直接挂在 infra 的 Base 上）。"""

    # 告诉 pytest：这不是测试类（只是类名以 Test* 开头）
    __test__ = False

    __tablename__ = "test_repository_model"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(64), index=True)
    stat_date: Mapped[date] = mapped_column(Date)
    value: Mapped[Optional[float]] = mapped_column(Numeric(18, 8), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)


# ============================================================
# Fixture
# ============================================================
@pytest_asyncio.fixture
async def repo(engine):
    """
    准备一个干净的 TestModel 表 + 完整测试数据，yield 一个可用 Repository。

    - engine 来自 conftest 的 function-scoped fixture（业务 DB engine）
    - 用 conftest 的临时库，session 末会 DROP 库
    """
    # 建表（幂等：每次 fixture 都先 drop 再 create，避免残留状态）
    async with engine.begin() as conn:
        await conn.run_sync(_InfraBase.metadata.drop_all)
        await conn.run_sync(_InfraBase.metadata.create_all)

    repo = DatabaseRepository(TestModel, db_type="business")

    # 清理旧数据（防 drop_all 后还有残留）
    async with repo.get_session() as session:
        await session.execute(text(f"DELETE FROM {TestModel.__tablename__}"))
        await session.commit()

    # 插入测试数据
    now = date.today()
    test_records = [
        # VIXCLS: 6 条，全部 active
        {"series_id": "VIXCLS", "stat_date": now - timedelta(days=5), "value": 15.2, "category": "fred", "is_active": True},
        {"series_id": "VIXCLS", "stat_date": now - timedelta(days=4), "value": 16.1, "category": "fred", "is_active": True},
        {"series_id": "VIXCLS", "stat_date": now - timedelta(days=3), "value": 14.8, "category": "fred", "is_active": True},
        {"series_id": "VIXCLS", "stat_date": now - timedelta(days=2), "value": 17.3, "category": "fred", "is_active": True},
        {"series_id": "VIXCLS", "stat_date": now - timedelta(days=1), "value": 16.5, "category": "fred", "is_active": True},
        {"series_id": "VIXCLS", "stat_date": now, "value": 15.8, "category": "fred", "is_active": True},
        # RRPONTTLD: 4 条
        {"series_id": "RRPONTTLD", "stat_date": now - timedelta(days=3), "value": 2000.0, "category": "fred", "is_active": True},
        {"series_id": "RRPONTTLD", "stat_date": now - timedelta(days=2), "value": 2100.0, "category": "fred", "is_active": True},
        {"series_id": "RRPONTTLD", "stat_date": now - timedelta(days=1), "value": 2050.0, "category": "fred", "is_active": True},
        {"series_id": "RRPONTTLD", "stat_date": now, "value": 2080.0, "category": "fred", "is_active": True},
        # WALCL: 4 条，其中 1 条 inactive，1 条 value=None
        {"series_id": "WALCL", "stat_date": now - timedelta(days=5), "value": 7150.0, "category": "fred", "is_active": False},
        {"series_id": "WALCL", "stat_date": now - timedelta(days=3), "value": 7180.0, "category": "fred", "is_active": True},
        {"series_id": "WALCL", "stat_date": now - timedelta(days=1), "value": 7200.0, "category": "fred", "is_active": True},
        {"series_id": "WALCL", "stat_date": now, "value": 7220.0, "category": "fred", "is_active": True},
        # 带 None 值
        {"series_id": "WALCL", "stat_date": now, "value": None, "category": "fred", "is_active": True},
        {"series_id": "TEST_NULL", "stat_date": now, "value": None, "category": "test", "is_active": True},
        {"series_id": "TEST_NULL", "stat_date": now - timedelta(days=1), "value": None, "category": "test", "is_active": True},
    ]

    for record in test_records:
        await repo.create(**record)

    yield repo


# ============================================================
# 1. 基础 CRUD 测试
# ============================================================

async def test_create(repo):
    """测试创建记录"""
    now = date.today()
    record = await repo.create(
        series_id="TEST_CREATE",
        stat_date=now,
        value=99.99,
        category="test",
        is_active=True,
        description="test description"
    )

    assert record.id is not None
    assert record.series_id == "TEST_CREATE"
    assert float(record.value) == 99.99
    assert record.category == "test"
    assert record.is_active is True
    assert record.description == "test description"

    # 验证数据库中存在
    found = await repo.get_by_id(record.id)
    assert found is not None
    assert found.id == record.id


async def test_get_by_id(repo):
    """测试根据 ID 获取记录"""
    now = date.today()
    record = await repo.create(
        series_id="TEST_GET",
        stat_date=now,
        value=42.0
    )

    found = await repo.get_by_id(record.id)
    assert found is not None
    assert found.series_id == "TEST_GET"

    # 不存在的 ID
    not_found = await repo.get_by_id(999999)
    assert not_found is None


async def test_update(repo):
    """测试更新记录"""
    now = date.today()
    record = await repo.create(
        series_id="TEST_UPDATE",
        stat_date=now,
        value=50.0,
        category="test",
        is_active=True
    )

    updated = await repo.update(
        record.id,
        value=75.5,
        category="updated",
        is_active=False
    )
    assert updated is not None
    assert float(updated.value) == 75.5
    assert updated.category == "updated"
    assert updated.is_active is False

    # 验证更新生效
    found = await repo.get_by_id(record.id)
    assert float(found.value) == 75.5


async def test_update_nonexistent(repo):
    """测试更新不存在的记录"""
    updated = await repo.update(999999, value=100.0)
    assert updated is None


async def test_delete(repo):
    """测试删除记录"""
    now = date.today()
    record = await repo.create(
        series_id="TEST_DELETE",
        stat_date=now,
        value=100.0
    )

    deleted = await repo.delete(record.id)
    assert deleted is True

    found = await repo.get_by_id(record.id)
    assert found is None


async def test_delete_nonexistent(repo):
    """测试删除不存在的记录"""
    deleted = await repo.delete(999999)
    assert deleted is False


async def test_get_all(repo):
    """测试获取所有记录"""
    all_records = await repo.get_all()
    assert len(all_records) >= 17  # 测试数据数量

    # 带过滤
    filtered = await repo.get_all(series_id="VIXCLS")
    assert len(filtered) == 6
    for r in filtered:
        assert r.series_id == "VIXCLS"

    # 多个过滤条件
    filtered = await repo.get_all(series_id="VIXCLS", is_active=True)
    assert len(filtered) == 6

    # limit
    limited = await repo.get_all(limit=5)
    assert len(limited) == 5


async def test_count(repo):
    """测试统计记录数"""
    total = await repo.count()
    assert total >= 17

    # 带过滤
    count = await repo.count(series_id="VIXCLS")
    assert count == 6

    count = await repo.count(series_id="WALCL", is_active=True)
    assert count == 4


# ============================================================
# 2. query 方法 - where 参数
# ============================================================

async def test_query_where_simple(repo):
    """测试 query where 简单等值条件"""
    # 单个条件
    records = await repo.query(where={"series_id": "VIXCLS"})
    assert len(records) == 6

    # 多个条件 (AND)
    records = await repo.query(where={"series_id": "VIXCLS", "is_active": True})
    assert len(records) == 6

    # 多个条件，其中一个是 False
    records = await repo.query(where={"series_id": "WALCL", "is_active": True})
    assert len(records) == 4  # 4 条 WALCL 中 1 条 inactive, 1 条 value=None (active)


async def test_query_where_advanced(repo):
    """测试 query where 高级用法"""
    # where 中使用 in
    records = await repo.query(
        where={"series_id": {"in": ["VIXCLS", "WALCL"]}}
    )
    assert len(records) >= 9  # 6 + 4 = 10，但 WALCL 有 4 条

    # where 中使用 between
    now = date.today()
    records = await repo.query(
        where={"stat_date": {"between": [now - timedelta(days=3), now]}}
    )
    assert len(records) >= 5

    # where 中使用 gt
    records = await repo.query(
        where={"value": {"gt": 16.0}}
    )
    assert len(records) >= 3

    # where 中使用 gte
    records = await repo.query(
        where={"value": {"gte": 16.0}}
    )
    assert len(records) >= 3

    # where 中使用 lt
    records = await repo.query(
        where={"value": {"lt": 16.0}}
    )
    assert len(records) >= 2

    # where 中使用 lte
    records = await repo.query(
        where={"value": {"lte": 16.0}}
    )
    assert len(records) >= 3

    # where 中使用 like
    records = await repo.query(
        where={"series_id": {"like": "VIX%"}}
    )
    assert len(records) == 6


async def test_query_where_empty(repo):
    """测试空 where"""
    records = await repo.query(where={})
    assert len(records) >= 17

    records = await repo.query()
    assert len(records) >= 17


# ============================================================
# 3. query 方法 - conditions 操作符
# ============================================================

async def test_query_conditions_eq_ne(repo):
    """测试 eq/ne 操作符"""
    # eq
    records = await repo.query(conditions=[("series_id", "eq", "VIXCLS")])
    assert len(records) == 6

    # ne
    records = await repo.query(conditions=[("series_id", "ne", "VIXCLS")])
    assert len(records) >= 10


async def test_query_conditions_comparison(repo):
    """测试比较操作符 gt/gte/lt/lte"""
    # gt
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS"), ("value", "gt", 16.0)]
    )
    assert len(records) >= 2
    for r in records:
        assert float(r.value) > 16.0

    # gte
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS"), ("value", "gte", 16.0)]
    )
    assert len(records) >= 2

    # lt
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS"), ("value", "lt", 16.0)]
    )
    assert len(records) >= 2
    for r in records:
        assert float(r.value) < 16.0

    # lte
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS"), ("value", "lte", 16.0)]
    )
    assert len(records) >= 3


async def test_query_conditions_null_handling(repo):
    """⭐ 核心测试：query 处理 None 值"""
    # is_not None
    records = await repo.query(
        conditions=[("series_id", "eq", "WALCL"), ("value", "is_not", None)]
    )
    assert len(records) == 4  # 4 条 WALCL 中 1 条 value=None (active)


    # is_null
    records = await repo.query(
        conditions=[("series_id", "eq", "TEST_NULL"), ("value", "is_null", None)]
    )
    assert len(records) >= 1

    # is_not_null
    records = await repo.query(
        conditions=[("series_id", "eq", "WALCL"), ("value", "is_not_null", None)]
    )
    assert len(records) == 4

    # lte date + is_not None 组合（之前报错的场景）
    now = date.today()
    records = await repo.query(
        conditions=[
            ("series_id", "eq", "VIXCLS"),
            ("stat_date", "lte", now),
            ("value", "is_not", None)
        ]
    )
    assert len(records) == 6

    # between + is_not None 组合
    now = date.today()
    records = await repo.query(
        conditions=[
            ("series_id", "eq", "RRPONTTLD"),
            ("stat_date", "between", [now - timedelta(days=3), now]),
            ("value", "is_not", None)
        ]
    )
    assert len(records) == 4


async def test_query_conditions_like(repo):
    """测试 like/ilike 操作符"""
    # like
    records = await repo.query(conditions=[("series_id", "like", "VIX%")])
    assert len(records) == 6

    records = await repo.query(conditions=[("series_id", "like", "%NULL%")])
    assert len(records) >= 1

    # ilike (case insensitive)
    records = await repo.query(conditions=[("series_id", "ilike", "vix%")])
    assert len(records) == 6


async def test_query_conditions_in(repo):
    """测试 in/not_in 操作符"""
    records = await repo.query(conditions=[("series_id", "in", ["VIXCLS", "WALCL"])])
    assert len(records) >= 9

    records = await repo.query(conditions=[("series_id", "not_in", ["VIXCLS", "WALCL"])])
    assert len(records) >= 1


async def test_query_conditions_between(repo):
    """测试 between 操作符"""
    now = date.today()
    records = await repo.query(
        conditions=[("stat_date", "between", [now - timedelta(days=3), now])]
    )
    assert len(records) >= 5

    # between 无效参数（应返回空或忽略）
    records = await repo.query(
        conditions=[("stat_date", "between", [now])]  # 只有一个值
    )
    # 应该返回所有记录（无效条件被忽略）
    assert len(records) >= 17


async def test_query_conditions_is_operator(repo):
    """测试 is_ 操作符（用于布尔值比较）"""
    records = await repo.query(conditions=[("is_active", "is", True)])
    assert len(records) >= 10

    records = await repo.query(conditions=[("is_active", "is", False)])
    assert len(records) == 1  # WALCL 中 1 条 inactive


async def test_query_conditions_is_not_operator(repo):
    """测试 is_not 操作符"""
    records = await repo.query(conditions=[("is_active", "is_not", True)])
    assert len(records) == 1

    records = await repo.query(conditions=[("is_active", "is_not", False)])
    assert len(records) >= 10


# ============================================================
# 4. query 方法 - 逻辑、排序、分页
# ============================================================

async def test_query_logic_or(repo):
    """测试 OR 逻辑"""
    records_and = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS"), ("series_id", "eq", "WALCL")],
        logic="AND"
    )
    assert len(records_and) == 0  # AND 同时满足两个条件

    records_or = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS"), ("series_id", "eq", "WALCL")],
        logic="OR"
    )
    assert len(records_or) >= 10


async def test_query_limit_offset(repo):
    """测试 limit 和 offset"""
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS")],
        limit=3
    )
    assert len(records) == 3

    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS")],
        limit=3,
        offset=2
    )
    assert len(records) == 3


async def test_query_order_by(repo):
    """测试 order_by"""
    # 字符串格式
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS")],
        order_by="stat_date desc"
    )
    assert len(records) == 6
    if len(records) >= 2:
        assert records[0].stat_date >= records[1].stat_date

    # 字典格式
    records = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS")],
        order_by={"stat_date": "asc"}
    )
    if len(records) >= 2:
        assert records[0].stat_date <= records[1].stat_date

    # 多字段排序
    records = await repo.query(
        order_by=["category asc", "stat_date desc"]
    )
    assert len(records) >= 10


async def test_query_select_fields(repo):
    """测试 select_fields"""
    results = await repo.query(
        conditions=[("series_id", "eq", "VIXCLS")],
        select_fields=["series_id", "value"],
        limit=2
    )
    # SQLAlchemy 返回 Row 对象，可通过下标访问
    assert len(results) == 2
    assert results[0][0] == "VIXCLS"  # series_id
    assert results[0][1] is not None   # value


async def test_query_distinct(repo):
    """测试 distinct"""
    # 不带 distinct
    results = await repo.query(
        select_fields=["category"],
        limit=10
    )
    # 可能有重复

    # 带 distinct
    distinct_results = await repo.query(
        select_fields=["category"],
        distinct=True
    )
    # 去重后 category 数量
    assert len(distinct_results) >= 1


# ============================================================
# 5. 分页测试
# ============================================================

async def test_paginate_basic(repo):
    """测试基本分页"""
    result = await repo.paginate(
        page=1,
        page_size=5,
        where={"series_id": "VIXCLS"}
    )

    assert isinstance(result, PageResult)
    assert result.page == 1
    assert result.page_size == 5
    assert result.total == 6
    assert result.total_pages == 2
    assert len(result.items) == 5

    # 第二页
    result2 = await repo.paginate(
        page=2,
        page_size=5,
        where={"series_id": "VIXCLS"}
    )
    assert len(result2.items) == 1


async def test_paginate_with_conditions(repo):
    """测试带 conditions 的分页"""
    result = await repo.paginate(
        page=1,
        page_size=3,
        conditions=[
            ("series_id", "eq", "VIXCLS"),
            ("value", "gt", 16.0)
        ]
    )
    assert result.total >= 2
    assert len(result.items) >= 1


async def test_paginate_empty(repo):
    """测试空结果分页"""
    result = await repo.paginate(
        page=1,
        page_size=10,
        where={"series_id": "NOT_EXISTS"}
    )
    assert result.total == 0
    assert len(result.items) == 0
    assert result.total_pages == 1


async def test_paginate_out_of_range(repo):
    """测试超出总页数"""
    result = await repo.paginate(
        page=999,
        page_size=5,
        where={"series_id": "VIXCLS"}
    )
    # 超出总页数返回空列表
    assert len(result.items) == 0


async def test_paginate_page_zero(repo):
    """测试页码为0（应自动修正为1）"""
    result = await repo.paginate(
        page=0,
        page_size=5,
        where={"series_id": "VIXCLS"}
    )
    assert result.page == 1


# ============================================================
# 6. 批量操作测试
# ============================================================

async def test_bulk_create(repo):
    """测试批量创建"""
    now = date.today()
    items = [
        {"series_id": "BULK_1", "stat_date": now, "value": 1.0, "category": "bulk"},
        {"series_id": "BULK_2", "stat_date": now, "value": 2.0, "category": "bulk"},
        {"series_id": "BULK_3", "stat_date": now, "value": 3.0, "category": "bulk"},
    ]

    instances = await repo.bulk_create(items)
    assert len(instances) == 3

    records = await repo.query(where={"category": "bulk"})
    assert len(records) == 3


async def test_bulk_create_empty(repo):
    """测试空批量创建"""
    instances = await repo.bulk_create([])
    assert len(instances) == 0


async def test_bulk_update(repo):
    """测试批量更新"""
    now = date.today()
    items = [
        {"series_id": "BULK_UPDATE_1", "stat_date": now, "value": 10.0, "category": "bulk_update"},
        {"series_id": "BULK_UPDATE_2", "stat_date": now, "value": 20.0, "category": "bulk_update"},
    ]
    instances = await repo.bulk_create(items)
    ids = [inst.id for inst in instances]

    count = await repo.bulk_update(ids, {"value": 99.0, "category": "updated"})
    assert count == 2

    updated = await repo.query(where={"category": "updated"})
    assert len(updated) == 2
    for r in updated:
        assert float(r.value) == 99.0


async def test_bulk_update_empty(repo):
    """测试空批量更新"""
    count = await repo.bulk_update([], {"value": 100.0})
    assert count == 0


async def test_bulk_update_batch_size(repo):
    """测试分批批量更新"""
    now = date.today()
    items = []
    for i in range(15):
        items.append({
            "series_id": f"BATCH_{i}",
            "stat_date": now,
            "value": float(i),
            "category": "batch_test"
        })
    instances = await repo.bulk_create(items)
    ids = [inst.id for inst in instances]

    # batch_size=5，应该分 3 批
    count = await repo.bulk_update(ids, {"category": "batch_updated"}, batch_size=5)
    assert count == 15

    updated = await repo.query(where={"category": "batch_updated"})
    assert len(updated) == 15


async def test_bulk_delete(repo):
    """测试批量删除"""
    now = date.today()
    items = [
        {"series_id": "BULK_DEL_1", "stat_date": now, "value": 1.0},
        {"series_id": "BULK_DEL_2", "stat_date": now, "value": 2.0},
    ]
    instances = await repo.bulk_create(items)
    ids = [inst.id for inst in instances]

    count = await repo.bulk_delete(ids)
    assert count == 2

    remaining = await repo.query(where={"series_id": {"in": ["BULK_DEL_1", "BULK_DEL_2"]}})
    assert len(remaining) == 0


async def test_bulk_delete_empty(repo):
    """测试空批量删除"""
    count = await repo.bulk_delete([])
    assert count == 0


# ============================================================
# 7. 存在性测试
# ============================================================

async def test_exists(repo):
    """测试 exists 方法"""
    exists = await repo.exists(series_id="VIXCLS")
    assert exists is True

    exists = await repo.exists(series_id="NOT_EXISTS")
    assert exists is False

    exists = await repo.exists(series_id="VIXCLS", value=15.8)
    assert exists is True

    exists = await repo.exists(series_id="VIXCLS", value=999.0)
    assert exists is False


# ============================================================
# 8. 会话管理测试
# ============================================================

async def test_session_context_manager(repo):
    """测试会话上下文管理器"""
    async with repo.get_session() as session:
        assert session is not None
        # 执行简单查询验证会话有效
        result = await session.execute(text("SELECT 1"))
        row = result.scalar()
        assert row == 1


# ============================================================
# 9. 异常处理测试
# ============================================================

async def test_query_invalid_operator(repo):
    """测试无效操作符（应忽略）"""
    # 无效操作符应该被忽略，返回所有记录
    records = await repo.query(
        conditions=[("series_id", "unknown_op", "VIXCLS")]
    )
    assert len(records) >= 17


async def test_query_invalid_field(repo):
    """测试无效字段名（应忽略）"""
    # 无效字段名应该被忽略
    records = await repo.query(
        where={"invalid_field": "value"}
    )
    assert len(records) >= 17

    records = await repo.query(
        conditions=[("invalid_field", "eq", "value")]
    )
    assert len(records) >= 17


# ============================================================
# 10. 综合场景测试
# ============================================================

async def test_fred_repository_scenario(repo):
    """模拟 FredRepository 的典型使用场景"""
    now = date.today()

    # 场景1: get_raw_value_as_of - 获取指定日期或之前的最新值
    records = await repo.query(
        conditions=[
            ("series_id", "eq", "VIXCLS"),
            ("stat_date", "lte", now),
            ("value", "is_not", None)
        ],
        order_by={"stat_date": "desc"},
        limit=1
    )
    assert len(records) == 1
    assert records[0].series_id == "VIXCLS"

    # 场景2: get_raw_series - 获取历史序列（用于 percentile/zscore/trend）
    records = await repo.query(
        conditions=[
            ("series_id", "eq", "VIXCLS"),
            ("stat_date", "lte", now),
            ("value", "is_not", None)
        ],
        order_by={"stat_date": "asc"},
        limit=5
    )
    assert len(records) == 5
    if len(records) >= 2:
        assert records[0].stat_date <= records[1].stat_date

    # 场景3: 获取最近5条
    records = await repo.query(
        conditions=[
            ("series_id", "eq", "VIXCLS"),
            ("value", "is_not", None)
        ],
        order_by={"stat_date": "desc"},
        limit=5
    )
    assert len(records) == 5

    # 场景4: 带日期的 range 查询（用于 delta 计算）
    start_date = now - timedelta(days=30)
    records = await repo.query(
        conditions=[
            ("series_id", "eq", "RRPONTTLD"),
            ("stat_date", "gte", start_date),
            ("stat_date", "lte", now),
            ("value", "is_not", None)
        ],
        order_by={"stat_date": "asc"}
    )
    assert len(records) == 4  # RRPONTTLD 有 4 条

    # 场景5: 分页查询历史数据
    result = await repo.paginate(
        page=1,
        page_size=3,
        conditions=[
            ("series_id", "eq", "VIXCLS"),
            ("value", "is_not", None)
        ],
        order_by={"stat_date": "desc"}
    )
    assert result.total == 6
    assert len(result.items) == 3


# ============================================================
# 11. 类型转换测试
# ============================================================

async def test_decimal_to_float_conversion(repo):
    """测试 Decimal 到 float 的转换"""
    now = date.today()
    record = await repo.create(
        series_id="TEST_DECIMAL",
        stat_date=now,
        value=123.456789,
        category="test"
    )

    found = await repo.get_by_id(record.id)
    assert found is not None
    assert isinstance(float(found.value), float)
    assert round(float(found.value), 6) == 123.456789


# ============================================================
# 运行说明
# ============================================================

if __name__ == "__main__":
    print("请使用 pytest 运行此测试文件:")
    print("  pytest tests/test_database_repository.py -v")
