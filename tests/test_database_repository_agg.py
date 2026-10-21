#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DatabaseRepository 增强测试（20261021）

覆盖三类内容：

1. **Bug 回归**——修复前会「静默返回错误结果」的问题：
   - ``where`` 字典形式丢弃 ``is_null`` / ``is_not_null`` / ``eq`` / ``ne`` 条件
   - ``in`` 空 list 在 conditions 形式下被丢弃 → 返回全表
   - ``paginate`` 的 where 只支持等值（与 query 行为不一致）
   - ``where`` 字典形式与 conditions 形式能力不对等
   - ``count`` / ``exists`` 不支持 where / conditions

2. **能力对等性**——三种过滤入参形式（where 等值 / where 操作符 dict /
   conditions 列表）现在行为一致。

3. **新增能力**：``query_aggregate()``——GROUP BY / HAVING / 聚合函数。
   独立方法，不改 ``query()`` 签名，向后兼容。

依赖：tests/conftest.py 的 sqlite 临时库 + engine fixture。
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional

import pytest
import pytest_asyncio
from sqlalchemy import (
    Boolean,
    Date,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from infra.db.database_model import Base as _InfraBase
from infra.db.database_repository import DatabaseRepository


# ============================================================
# 测试模型
# ============================================================
class AggModel(_InfraBase):
    """聚合测试用的模型。"""

    __test__ = False

    __tablename__ = "test_agg_model"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[str] = mapped_column(String(64), index=True)
    stat_date: Mapped[date] = mapped_column(Date)
    value: Mapped[Optional[float]] = mapped_column(Numeric(18, 8), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@pytest_asyncio.fixture
async def agg_repo(engine):
    """干净的 AggModel 表 + 测试数据。"""
    async with engine.begin() as conn:
        await conn.run_sync(_InfraBase.metadata.drop_all)
        await conn.run_sync(_InfraBase.metadata.create_all)

    repo = DatabaseRepository(AggModel, db_key="db")
    async with repo.get_session() as session:
        await session.execute(text(f"DELETE FROM {AggModel.__tablename__}"))
        await session.commit()

    now = date.today()
    records = [
        # A 组：3 条，值为 10 / 20 / 30
        {"series_id": "A", "stat_date": now - timedelta(days=2), "value": 10.0, "category": "x", "is_active": True},
        {"series_id": "A", "stat_date": now - timedelta(days=1), "value": 20.0, "category": "x", "is_active": True},
        {"series_id": "A", "stat_date": now, "value": 30.0, "category": "x", "is_active": True},
        # B 组：2 条，值为 100 / 200
        {"series_id": "B", "stat_date": now - timedelta(days=1), "value": 100.0, "category": "y", "is_active": True},
        {"series_id": "B", "stat_date": now, "value": 200.0, "category": "y", "is_active": True},
        # C 组：1 条，value 为 NULL
        {"series_id": "C", "stat_date": now, "value": None, "category": "z", "is_active": True},
        # D 组：inactive
        {"series_id": "D", "stat_date": now, "value": 999.0, "category": "x", "is_active": False},
    ]
    for r in records:
        await repo.create(**r)

    yield repo


# ============================================================
# 1. Bug 回归：where 字典形式静默丢条件
# ============================================================
class TestWhereDictBugFix:
    """修复前：where 的 dict 操作符形式只认 7 个操作符，其余被静默丢弃。

    丢弃意味着**查询条件丢失 → 返回比预期更多的行**——比报错更危险。
    """

    async def test_where_dict_is_null_not_dropped(self, agg_repo):
        """where={"value": {"is_null": None}} 必须生效。"""
        rows = await agg_repo.query(where={"value": {"is_null": None}})
        # C 组 1 条 + （无其他 NULL）→ 修复前会返回全部 7 条
        assert len(rows) == 1
        assert rows[0].series_id == "C"

    async def test_where_dict_is_not_null_not_dropped(self, agg_repo):
        """where={"value": {"is_not_null": None}} 必须生效。"""
        rows = await agg_repo.query(where={"value": {"is_not_null": None}})
        assert len(rows) == 6
        assert all(r.value is not None for r in rows)

    async def test_where_dict_eq_not_dropped(self, agg_repo):
        """where={"category": {"eq": "x"}} 必须生效。"""
        rows = await agg_repo.query(where={"category": {"eq": "x"}})
        assert len(rows) == 4
        assert all(r.category == "x" for r in rows)

    async def test_where_dict_ne_not_dropped(self, agg_repo):
        """where={"category": {"ne": "x"}} 必须生效。"""
        rows = await agg_repo.query(where={"category": {"ne": "x"}})
        assert len(rows) == 3
        assert all(r.category != "x" for r in rows)

    async def test_where_dict_gt_still_works(self, agg_repo):
        """修复不能破坏原本能用的操作符。"""
        rows = await agg_repo.query(where={"value": {"gt": 50}})
        assert len(rows) == 3
        assert all(r.value > 50 for r in rows)

    async def test_where_dict_between_still_works(self, agg_repo):
        rows = await agg_repo.query(where={"value": {"between": [50, 250]}})
        assert len(rows) == 2

    async def test_where_dict_in_still_works(self, agg_repo):
        rows = await agg_repo.query(where={"series_id": {"in": ["A", "B"]}})
        assert len(rows) == 5

    async def test_where_dict_multiple_operators_combined(self, agg_repo):
        """同一字段多个操作符——AND 组合。"""
        rows = await agg_repo.query(where={"value": {"gt": 10, "lt": 200}})
        # 20, 30, 100 → 3 条
        assert len(rows) == 3


# ============================================================
# 2. Bug 回归：in 空 list
# ============================================================
class TestInEmptyListBugFix:
    """修复前：conditions 形式的 in 空 list 返回 None → 条件被丢弃 → 返回全表。"""

    async def test_conditions_in_empty_list_returns_empty(self, agg_repo):
        """conditions=[("series_id", "in", [])] 应返回 0 条。"""
        rows = await agg_repo.query(conditions=[("series_id", "in", [])])
        assert rows == []

    async def test_where_dict_in_empty_list_returns_empty(self, agg_repo):
        """where={"series_id": {"in": []}} 也应返回 0 条（两处语义一致）。"""
        rows = await agg_repo.query(where={"series_id": {"in": []}})
        assert rows == []

    async def test_conditions_not_in_empty_list_returns_all(self, agg_repo):
        """not_in 空 list → 匹配所有行（不是丢弃条件）。"""
        rows = await agg_repo.query(conditions=[("series_id", "not_in", [])])
        assert len(rows) == 7

    async def test_conditions_in_empty_list_with_other_condition(self, agg_repo):
        """in 空 list 与其他条件组合——仍是空集。"""
        rows = await agg_repo.query(
            conditions=[("series_id", "in", []), ("category", "eq", "x")],
        )
        assert rows == []


# ============================================================
# 3. Bug 回归：paginate / count / exists 与 query 对齐
# ============================================================
class TestPaginateCountExistsAlignment:
    async def test_paginate_where_dict_operator(self, agg_repo):
        """修复前 paginate 的 where 只支持等值，操作符 dict 会被当成等值比较。"""
        result = await agg_repo.paginate(page=1, page_size=10, where={"value": {"gt": 50}})
        assert result.total == 3
        assert len(result.items) == 3
        assert all(item.value > 50 for item in result.items)

    async def test_paginate_where_dict_is_null(self, agg_repo):
        result = await agg_repo.paginate(page=1, page_size=10, where={"value": {"is_null": None}})
        assert result.total == 1
        assert result.items[0].series_id == "C"

    async def test_paginate_where_dict_in_empty(self, agg_repo):
        result = await agg_repo.paginate(page=1, page_size=10, where={"series_id": {"in": []}})
        assert result.total == 0
        assert result.items == []

    async def test_paginate_equals_query_result(self, agg_repo):
        """paginate 与 query 对同一条件应给出一致的总数。"""
        p = await agg_repo.paginate(page=1, page_size=100, where={"category": "x"})
        q = await agg_repo.query(where={"category": "x"})
        assert p.total == len(q)

    async def test_count_supports_where_operator(self, agg_repo):
        """count 支持 where 的操作符形式。"""
        assert await agg_repo.count(where={"value": {"gt": 50}}) == 3
        assert await agg_repo.count(where={"value": {"is_null": None}}) == 1
        assert await agg_repo.count(where={"series_id": {"in": ["A", "B"]}}) == 5

    async def test_count_supports_conditions(self, agg_repo):
        assert await agg_repo.count(conditions=[("value", "gt", 50)]) == 3
        assert await agg_repo.count(conditions=[("series_id", "in", ["A"])]) == 3

    async def test_count_legacy_filters_still_work(self, agg_repo):
        """向后兼容：原来的 count(category="x") 仍可用。"""
        assert await agg_repo.count(category="x") == 4

    async def test_exists_supports_where_operator(self, agg_repo):
        assert await agg_repo.exists(where={"value": {"gt": 1000}}) is False
        assert await agg_repo.exists(where={"value": {"gt": 50}}) is True
        assert await agg_repo.exists(where={"value": {"is_null": None}}) is True

    async def test_exists_supports_conditions(self, agg_repo):
        assert await agg_repo.exists(conditions=[("series_id", "eq", "A")]) is True
        assert await agg_repo.exists(conditions=[("series_id", "eq", "ZZZ")]) is False

    async def test_exists_legacy_filters_still_work(self, agg_repo):
        assert await agg_repo.exists(category="x") is True
        assert await agg_repo.exists(category="nope") is False


# ============================================================
# 4. 能力对等性：三种过滤形式结果一致
# ============================================================
class TestFilterFormEquivalence:
    async def test_where_dict_vs_conditions_equivalent(self, agg_repo):
        """where 操作符形式与 conditions 列表形式应完全等价。"""
        a = await agg_repo.query(where={"value": {"gt": 50}})
        b = await agg_repo.query(conditions=[("value", "gt", 50)])
        assert len(a) == len(b) == 3
        assert {r.id for r in a} == {r.id for r in b}

    async def test_where_dict_is_null_vs_conditions_equivalent(self, agg_repo):
        a = await agg_repo.query(where={"value": {"is_null": None}})
        b = await agg_repo.query(conditions=[("value", "is_null", None)])
        assert len(a) == len(b) == 1

    async def test_where_equality_vs_conditions_equivalent(self, agg_repo):
        a = await agg_repo.query(where={"series_id": "A"})
        b = await agg_repo.query(conditions=[("series_id", "eq", "A")])
        assert len(a) == len(b) == 3

    async def test_filters_kwarg_vs_where_equivalent(self, agg_repo):
        a = await agg_repo.query(where={"category": "x"})
        b = await agg_repo.query(category="x")
        assert len(a) == len(b) == 4

    async def test_where_none_value_becomes_is_null(self, agg_repo):
        """where={"value": None} 应等价于 IS NULL。"""
        rows = await agg_repo.query(where={"value": None})
        assert len(rows) == 1
        assert rows[0].series_id == "C"

    async def test_or_logic_with_where_dict(self, agg_repo):
        """logic="OR" 时 where 操作符形式也生效。"""
        rows = await agg_repo.query(
            where={"series_id": "A", "value": {"gt": 100}},
            logic="OR",
        )
        # A 组 3 条 + value>100 的 (200, 999) 2 条 → 5 条
        assert len(rows) == 5


# ============================================================
# 5. 新增：query_aggregate（GROUP BY / HAVING / 聚合）
# ============================================================
class TestQueryAggregate:
    async def test_count_with_group_by(self, agg_repo):
        """每组计数。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
        )
        by_series = {r["series_id"]: r["cnt"] for r in rows}
        assert by_series == {"A": 3, "B": 2, "C": 1, "D": 1}

    async def test_sum_avg_min_max(self, agg_repo):
        """多个聚合函数。"""
        rows = await agg_repo.query_aggregate(
            aggregates={
                "total": "sum:value",
                "avg_val": "avg:value",
                "min_val": "min:value",
                "max_val": "max:value",
            },
            group_by="series_id",
        )
        a = next(r for r in rows if r["series_id"] == "A")
        assert float(a["total"]) == 60.0
        assert float(a["avg_val"]) == 20.0
        assert float(a["min_val"]) == 10.0
        assert float(a["max_val"]) == 30.0

    async def test_aggregate_without_group_by(self, agg_repo):
        """不分组——全表聚合。"""
        rows = await agg_repo.query_aggregate(aggregates={"cnt": "count"})
        assert len(rows) == 1
        assert rows[0]["cnt"] == 7

    async def test_aggregate_with_where_filter(self, agg_repo):
        """聚合前的行级过滤。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
            where={"value": {"gt": 50}},
        )
        by_series = {r["series_id"]: r["cnt"] for r in rows}
        assert by_series == {"B": 2, "D": 1}

    async def test_aggregate_with_conditions(self, agg_repo):
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
            conditions=[("is_active", "eq", True)],
        )
        by_series = {r["series_id"]: r["cnt"] for r in rows}
        assert by_series == {"A": 3, "B": 2, "C": 1}

    async def test_aggregate_with_having(self, agg_repo):
        """HAVING —— 聚合后过滤。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
            having={"cnt": {"gt": 1}},
        )
        by_series = {r["series_id"]: r["cnt"] for r in rows}
        assert by_series == {"A": 3, "B": 2}

    async def test_aggregate_with_having_on_real_column(self, agg_repo):
        """HAVING 用真实列名——自动转成 sum(col) 比较。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"total": "sum:value"},
            group_by="series_id",
            having={"value": {"gt": 50}},
        )
        by_series = {r["series_id"]: float(r["total"]) for r in rows}
        # A 组 sum=60（>50），B 组 sum=300，D 组 sum=999 → C 组 value 全 NULL，sum=NULL 不满足
        assert by_series == {"A": 60.0, "B": 300.0, "D": 999.0}

    async def test_aggregate_order_by_alias(self, agg_repo):
        """按聚合别名排序。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
            order_by="cnt desc",
        )
        counts = [r["cnt"] for r in rows]
        assert counts == sorted(counts, reverse=True)

    async def test_aggregate_order_by_dict(self, agg_repo):
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
            order_by={"cnt": "asc"},
        )
        counts = [r["cnt"] for r in rows]
        assert counts == sorted(counts)

    async def test_aggregate_limit_offset(self, agg_repo):
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="series_id",
            order_by="series_id asc",
            limit=2,
            offset=1,
        )
        assert len(rows) == 2
        assert rows[0]["series_id"] == "B"

    async def test_aggregate_multiple_group_by(self, agg_repo):
        """多列分组。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by=["series_id", "category"],
        )
        pairs = {(r["series_id"], r["category"]) for r in rows}
        assert ("A", "x") in pairs
        assert ("B", "y") in pairs

    async def test_aggregate_func_expression(self, agg_repo):
        """func.xxx() 形式。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "func.count()", "total": "func.sum(value)"},
            group_by="series_id",
        )
        a = next(r for r in rows if r["series_id"] == "A")
        assert a["cnt"] == 3
        assert float(a["total"]) == 60.0

    async def test_aggregate_no_aggregates_raises(self, agg_repo):
        """空 aggregates 抛 DatabaseError。"""
        from infra.exceptions import DatabaseError

        with pytest.raises(DatabaseError):
            await agg_repo.query_aggregate(aggregates={})

    async def test_aggregate_unknown_column_warns_and_skips(self, agg_repo):
        """未知列——跳过并继续（不整体失败）。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count", "bad": "sum:nonexistent_col"},
            group_by="series_id",
        )
        # bad 被忽略，cnt 正常
        assert all("bad" not in r for r in rows)
        assert sum(r["cnt"] for r in rows) == 7

    async def test_aggregate_unknown_group_by_warns(self, agg_repo):
        """未知分组列——跳过。"""
        rows = await agg_repo.query_aggregate(
            aggregates={"cnt": "count"},
            group_by="nonexistent_col",
        )
        assert len(rows) == 1
        assert rows[0]["cnt"] == 7

    async def test_aggregate_does_not_break_query(self, agg_repo):
        """新增方法不影响原有 query() 的行为。"""
        rows = await agg_repo.query(where={"series_id": "A"})
        assert len(rows) == 3


# ============================================================
# 6. _get_column 白名单校验
# ============================================================
class TestGetColumn:
    def test_returns_real_column(self, agg_repo):
        col = agg_repo._get_column("series_id")
        assert col is not None
        assert col.name == "series_id"

    def test_rejects_non_column_attributes(self, agg_repo):
        """非列属性（metadata / registry / __table__）应返回 None。

        旧实现用 hasattr()——这些属性都是 True，会被误当列使用。
        """
        assert agg_repo._get_column("metadata") is None
        assert agg_repo._get_column("registry") is None
        assert agg_repo._get_column("__table__") is None

    def test_rejects_nonexistent_column(self, agg_repo):
        assert agg_repo._get_column("definitely_not_a_column") is None
