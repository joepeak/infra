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
from infra.db import get_db_manager
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

    repo = DatabaseRepository(TestModel, db_key="db")

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
# 12. 分组取最新记录测试 (get_latest_per_group)
# ============================================================

class TestGetLatestPerGroup:
    """测试 get_latest_per_group 方法"""

    async def test_get_latest_per_group_basic(self, repo):
        """测试基本的分组取最新记录"""
        # 每个 series_id 取最新的 1 条
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1
        )
        
        # VIXCLS: 最新 1 条 (stat_date = today)
        # RRPONTTLD: 最新 1 条 (stat_date = today)
        # WALCL: 最新 1 条 (stat_date = today, value=7220)
        # TEST_NULL: 最新 1 条 (stat_date = today, value=None)
        # 应该有 4 个分组
        assert len(records) == 4
        
        # 验证每个分组最新日期是 today
        now = date.today()
        for r in records:
            assert r.stat_date == now

    async def test_get_latest_per_group_limit_2(self, repo):
        """测试每个分组取最新的 2 条"""
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=2
        )
        
        # 验证 VIXCLS 有 2 条（今天和昨天）
        vix_records = [r for r in records if r.series_id == "VIXCLS"]
        assert len(vix_records) == 2
        # 排序应该是 descending
        if len(vix_records) >= 2:
            assert vix_records[0].stat_date >= vix_records[1].stat_date
        
        # RRPONTTLD 有 2 条
        rrp_records = [r for r in records if r.series_id == "RRPONTTLD"]
        assert len(rrp_records) == 2

    async def test_get_latest_per_group_with_where(self, repo):
        """测试带 where 条件的分组取最新"""
        # 只取 active 的记录
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=2,
            where={"is_active": True}
        )
        
        # WALCL 有 4 条 active（其中 1 条 value=None）
        walcl_records = [r for r in records if r.series_id == "WALCL"]
        # WALCL 有 4 条 active，取最新 2 条
        assert len(walcl_records) == 2
        for r in walcl_records:
            assert r.is_active is True

    async def test_get_latest_per_group_with_conditions(self, repo):
        """测试带 conditions 条件的分组取最新"""
        # 只取 value > 16 的记录
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=2,
            conditions=[("value", "gt", 16.0)]
        )
        
        # VIXCLS: 值 > 16 的有 3 条，取最新 2 条
        vix_records = [r for r in records if r.series_id == "VIXCLS"]
        assert len(vix_records) == 2
        for r in vix_records:
            assert float(r.value) > 16.0

    async def test_get_latest_per_group_with_logic_or(self, repo):
        """测试 OR 逻辑的分组取最新"""
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1,
            conditions=[
                ("series_id", "eq", "VIXCLS"),
                ("series_id", "eq", "WALCL")
            ],
            logic="OR"
        )
        
        # 应该返回 VIXCLS 和 WALCL 各最新 1 条
        series_ids = {r.series_id for r in records}
        assert "VIXCLS" in series_ids
        assert "WALCL" in series_ids
        assert len(records) == 2

    async def test_get_latest_per_group_with_extra_order(self, repo):
        """测试带额外排序的分组取最新"""
        # 在同一天的数据中，按 value 降序排列
        now = date.today()
        await repo.create(
            series_id="TEST_EXTRA_ORDER",
            stat_date=now,
            value=10.0,
            category="test"
        )
        await repo.create(
            series_id="TEST_EXTRA_ORDER",
            stat_date=now,
            value=30.0,
            category="test"
        )
        await repo.create(
            series_id="TEST_EXTRA_ORDER",
            stat_date=now,
            value=20.0,
            category="test"
        )
        
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=3,
            extra_order={"value": "desc"},
            where={"series_id": "TEST_EXTRA_ORDER"}
        )
        
        # 应该按 value 降序排列: 30, 20, 10
        assert len(records) == 3
        values = [float(r.value) for r in records]
        assert values == [30.0, 20.0, 10.0]

    async def test_get_latest_per_group_handle_null_values(self, repo):
        """测试处理 null 值的情况"""
        # TEST_NULL 分组有 2 条，都是 value=None
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1,
            where={"series_id": "TEST_NULL"}
        )
        
        assert len(records) == 1
        assert records[0].series_id == "TEST_NULL"
        assert records[0].stat_date == date.today()
        assert records[0].value is None

    async def test_get_latest_per_group_with_filters_kwargs(self, repo):
        """测试使用 **filters 参数"""
        # 使用关键字参数过滤
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1,
            category="fred",  # 只取 category='fred' 的记录
            is_active=True
        )
        
        # 应该返回所有 active 的 fred 分组的最新记录
        # VIXCLS, RRPONTTLD, WALCL (active 的)
        assert len(records) >= 3
        for r in records:
            assert r.category == "fred"
            assert r.is_active is True

    async def test_get_latest_per_group_nonexistent_column(self, repo):
        """测试不存在的列名"""
        with pytest.raises(ValueError, match="字段 'nonexistent_column' 不存在"):
            await repo.get_latest_per_group(
                group_column="nonexistent_column",
                order_column="stat_date",
                limit=1
            )

    async def test_get_latest_per_group_all_series(self, repo):
        """测试所有分组各取最新 1 条"""
        # 获取所有 series_id 的最新记录
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1
        )
        
        # 验证每个 series_id 只出现一次
        series_ids = [r.series_id for r in records]
        assert len(series_ids) == len(set(series_ids))
        
        # 验证包含所有预期的分组
        expected_series = {"VIXCLS", "RRPONTTLD", "WALCL", "TEST_NULL"}
        assert set(series_ids) == expected_series

    # ==================== 综合场景测试（在类内部） ====================

    async def test_get_latest_per_group_comprehensive(self, repo):
        """综合测试：模拟 FredRepository 中所有 series 的最新值获取"""
        now = date.today()
        
        # 场景：获取所有 FRED 系列的最新值（类似 dashboard 展示）
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1,
            where={"category": "fred"},
            is_active=True
        )
        
        # 验证结果
        # 应该返回 3 个分组：VIXCLS, RRPONTTLD, WALCL
        assert len(records) == 3
        
        # 构建 series_id 到值的映射
        value_map = {r.series_id: float(r.value) for r in records}
        
        # VIXCLS 最新值应该是 15.8
        assert "VIXCLS" in value_map
        assert value_map["VIXCLS"] == 15.8
        
        # RRPONTTLD 最新值应该是 2080.0
        assert "RRPONTTLD" in value_map
        assert value_map["RRPONTTLD"] == 2080.0
        
        # WALCL 最新值应该是 7220.0
        assert "WALCL" in value_map
        assert value_map["WALCL"] == 7220.0
        
        # 验证所有记录都是最新日期
        for r in records:
            assert r.stat_date == now

    async def test_get_latest_per_group_with_date_filter(self, repo):
        """测试带日期过滤的分组取最新"""
        now = date.today()
        # 只取 3 天前的数据（取每个分组在该日期或之前的最新值）
        
        # 这里我们直接用 conditions 过滤日期范围
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=1,
            conditions=[
                ("stat_date", "lte", now - timedelta(days=1)),
                ("value", "is_not", None)
            ]
        )
        
        # 对于 VIXCLS，日期 <= yesterday 的最新数据是 16.5
        vix_records = [r for r in records if r.series_id == "VIXCLS"]
        if vix_records:
            assert len(vix_records) == 1
            assert float(vix_records[0].value) == 16.5

    async def test_get_latest_per_group_large_dataset(self, repo):
        """测试大数据量下的分组取最新（使用批量创建）"""
        now = date.today()
        # 创建多个分组，每个分组多条数据
        items = []
        for i in range(10):  # 10个分组
            series_id = f"PERF_{i}"
            for day in range(30):  # 每个分组 30 条
                items.append({
                    "series_id": series_id,
                    "stat_date": now - timedelta(days=day),
                    "value": float(day * 10 + i),  # 使用 i 作为分组标识
                    "category": "perf_test",
                    "is_active": True
                })
        
        await repo.bulk_create(items)
        
        # 获取所有分组的最新 5 条
        records = await repo.get_latest_per_group(
            group_column="series_id",
            order_column="stat_date",
            limit=5,
            where={"category": "perf_test"}
        )
        
        # 应该有 10 * 5 = 50 条记录
        assert len(records) == 50
        
        # 验证每个分组有 5 条
        for i in range(10):
            series_id = f"PERF_{i}"
            group_records = [r for r in records if r.series_id == series_id]
            assert len(group_records) == 5
            # 验证日期是最近的 5 天
            dates = sorted([r.stat_date for r in group_records], reverse=True)
            assert dates[0] == now  # 最新的是今天
            assert len(dates) == 5



# ============================================================
# query_raw 原生 SQL 查询
# ============================================================
class TestQueryRaw:
    """测试 query_raw 方法"""

    async def test_simple_select(self, repo):
        """测试基本 SELECT 查询返回 dict 列表"""
        rows = await repo.query_raw(
            "SELECT series_id, value FROM test_repository_model WHERE series_id = 'VIXCLS'"
        )
        assert len(rows) == 6
        assert all(isinstance(r, dict) for r in rows)
        assert all("series_id" in r and "value" in r for r in rows)
        assert rows[0]["series_id"] == "VIXCLS"

    async def test_params_binding(self, repo):
        """测试参数绑定防止 SQL 注入"""
        rows = await repo.query_raw(
            "SELECT value FROM test_repository_model WHERE series_id = :sid AND value > :threshold",
            {"sid": "VIXCLS", "threshold": 16.0}
        )
        assert len(rows) == 3  # 16.1, 17.3, 16.5

    async def test_aggregate_query(self, repo):
        """测试聚合查询（GROUP BY + 聚合函数）"""
        rows = await repo.query_raw(
            "SELECT series_id, COUNT(*) as cnt, MAX(value) as max_val FROM test_repository_model "
            "WHERE category = 'fred' GROUP BY series_id ORDER BY series_id"
        )
        assert len(rows) == 3
        cnts = sorted(int(r["cnt"]) for r in rows)
        assert cnts == [4, 5, 6]

    async def test_pivot_with_case_when(self, repo):
        """测试 CASE WHEN 条件聚合透视（长表转宽表）"""
        now = date.today()
        rows = await repo.query_raw(
            "SELECT "
            "MAX(CASE WHEN stat_date = :d1 THEN value END) AS v1, "
            "MAX(CASE WHEN stat_date = :d2 THEN value END) AS v2 "
            "FROM test_repository_model WHERE series_id = :sid",
            {"sid": "VIXCLS", "d1": (now - timedelta(days=1)), "d2": (now - timedelta(days=2))}
        )
        assert len(rows) == 1
        assert "v1" in rows[0]
        assert "v2" in rows[0]

    async def test_empty_result(self, repo):
        """测试无结果返回空列表"""
        rows = await repo.query_raw(
            "SELECT value FROM test_repository_model WHERE series_id = 'NONEXISTENT'"
        )
        assert rows == []

    async def test_insert_via_raw_sql(self, repo):
        """测试通过原生 SQL 插入数据"""
        now = date.today()
        await repo.query_raw(
            "INSERT INTO test_repository_model (series_id, stat_date, value, category, is_active) "
            "VALUES (:sid, :sd, :val, :cat, :active)",
            {"sid": "RAW_INSERT", "sd": now, "val": 999.0, "cat": "test", "active": True}
        )
        rows = await repo.query_raw(
            "SELECT value FROM test_repository_model WHERE series_id = 'RAW_INSERT'"
        )
        assert len(rows) == 1
        assert float(rows[0]["value"]) == 999.0
