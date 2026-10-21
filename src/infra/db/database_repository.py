
from typing import Any, AsyncGenerator, Callable, Dict, Generic, List, Optional, TypeVar, Union, cast
from dataclasses import dataclass
from contextlib import asynccontextmanager

from sqlalchemy import select, delete, and_, or_, func, text, false, true
from sqlalchemy.ext.asyncio import AsyncSession
import asyncpg

from infra.db.connection_manager import get_db_session, get_db_manager_or_raise
from infra.db.operations import db_operation
from infra.logger import get_logger
from infra.exceptions import DatabaseError

ModelType = TypeVar('ModelType')


@dataclass
class PageResult(Generic[ModelType]):
    """分页查询结果"""
    items: List[ModelType]
    total: int
    page: int
    page_size: int
    total_pages: int


class DatabaseRepository(Generic[ModelType]):
    """
    异步数据库仓库基类，提供通用的CRUD操作

    Args:
        model_class: ORM 模型类
        db_key: db manager key（业务库 "db" / 时序库 "timescaledb" 等）
    """

    def __init__(self, model_class: ModelType, db_key: str = "db"):
        self.model_class = model_class
        self.db_key = db_key
        self.logger = get_logger(self.__class__.__name__)

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[Any, None]:
        """获取异步数据库会话。

        优先复用 ambient session（session_scope 内的 Unit of Work 事务），
        此时不在本调用 commit/rollback——由 session_scope 统一控制；
        无 ambient session 时回退到原行为（独立 session + commit-on-exit）。
        """
        from infra.db.connection_manager import get_current_session
        cur = get_current_session()
        if cur is not None:
            yield cur
            return
        async with get_db_session(self.db_key) as session:
            yield session

    @asynccontextmanager
    async def get_direct_connection(self) -> AsyncGenerator[Any, None]:
        """
        获取一个直接的数据库连接 (AsyncConnection)，用于执行原生SQL
        """
        db_manager = get_db_manager_or_raise(self.db_key)
        if not db_manager or not db_manager.engine:
            raise DatabaseError(
                f"数据库引擎 (key='{self.db_key}') 未初始化",
                operation="get_direct_connection",
            )

        async with db_manager.engine.connect() as conn:
            yield conn
    
    # ==================== 原生 SQL 查询 ====================
    
    @db_operation("query_raw")
    async def query_raw(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行原生 SQL，返回 dict 列表（列名 -> 值）。
        
        用于复杂查询（如透视/聚合）绕过 ORM 逐字段查询的限制。
        对于 SELECT 返回行数据；对于 INSERT/UPDATE/DELETE 提交事务并返回空列表。
        """
        async with self.get_session() as session:
            result = await session.execute(text(sql), params or {})
            # INSERT/UPDATE/DELETE 不返回行 —— 提交并返回空列表
            if not result.returns_rows:
                await session.commit()
                return []
            rows = [dict(row._mapping) for row in result]
            await session.commit()
            return rows
    
    # ==================== 基础 CRUD 操作 ====================
    
    @db_operation("create_record")
    async def create(self, model_instance: Optional[Any] = None, **kwargs: Any) -> Any:
        """创建记录，支持对象或字典参数"""
        async with self.get_session() as session:
            if model_instance is not None:
                if isinstance(model_instance, self.model_class):  # type: ignore[arg-type]
                    instance = model_instance
                else:
                    kwargs = {**model_instance, **kwargs}
                    instance = self.model_class(**kwargs)  # type: ignore[operator]
            else:
                instance = self.model_class(**kwargs)  # type: ignore[operator]
            
            session.add(instance)
            await session.flush()
            return instance
    
    @db_operation("get_by_id")
    async def get_by_id(self, record_id: Any) -> Optional[Any]:
        """根据ID获取记录"""
        from uuid import UUID
        # 兼容字符串 UUID：SQLite 的 UUID 列需要 UUID 对象而非字符串
        if isinstance(record_id, str):
            try:
                record_id = UUID(record_id)
            except ValueError:
                pass
        async with self.get_session() as session:
            result = await session.execute(
                select(self.model_class).where(  # type: ignore[call-overload]
                    self.model_class.id == record_id  # type: ignore[attr-defined]
                )
            )
            return result.scalar_one_or_none()
    
    @db_operation("get_all")
    async def get_all(self, limit: Optional[int] = None, **filters: Any) -> List[Any]:
        """获取所有记录"""
        async with self.get_session() as session:
            stmt = select(self.model_class)  # type: ignore[call-overload]
            
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            return result.scalars().all()  # type: ignore[no-any-return]
    
    @db_operation("update_record")
    async def update(self, record_id: Any, **kwargs: Any) -> Optional[Any]:
        """更新记录"""
        async with self.get_session() as session:
            instance = await session.get(self.model_class, record_id)
            
            if instance:
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                await session.flush()
                return instance
            
            return None
    
    @db_operation("delete_record")
    async def delete(self, record_id: Any) -> bool:
        """删除记录"""
        async with self.get_session() as session:
            instance = await session.get(self.model_class, record_id)
            
            if instance:
                await session.delete(instance)
                await session.flush()
                return True
            
            return False
    
    @db_operation("count_records")
    async def count(
        self,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        logic: str = "AND",
        **filters: Any
    ) -> int:
        """统计记录数。

        Args:
            where: 等值 / 带操作符的条件字典（与 query() 语义一致）
            conditions: 操作符条件列表 ``[(field, op, value), ...]``
            logic: 条件逻辑，"AND" 或 "OR"
            **filters: 简单等值过滤条件
        """
        async with self.get_session() as session:
            stmt = select(func.count()).select_from(self.model_class)  # type: ignore[arg-type]

            filter_exprs = self._build_filters(
                where=where,
                conditions=conditions,
                filters=filters,
            )
            stmt = self._apply_filters(stmt, filter_exprs, logic)

            result = await session.execute(stmt)
            return result.scalar() or 0
    
    # ==================== 灵活查询方法 ====================
    
    @db_operation("query_records")
    async def query(
        self,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        logic: str = "AND",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[Union[str, List[str], Dict[str, str]]] = None,
        select_fields: Optional[List[str]] = None,
        distinct: bool = False,
        **filters: Any
    ) -> List[Any]:
        """
        统一查询方法（合并原 query + find_by）

        Args:
            where: 等值条件字典，如 {"status": "active", "age": 18}
            conditions: 操作符条件列表，格式为 [(field, operator, value), ...]
                支持的 operator: eq, ne, gt, gte, lt, lte, like, ilike, in, not_in,
                is_null, is_not_null, between
            logic: conditions 逻辑，"AND" 或 "OR"
            limit: 限制数量
            offset: 偏移量
            order_by: 排序，支持多种格式：
                - str: "created_at desc"
                - List[str]: ["created_at desc", "id asc"]
                - Dict[str, str]: {"created_at": "desc", "id": "asc"}
            select_fields: 选择的字段列表
            distinct: 是否去重
            **filters: 简单等值过滤条件（向后兼容）
        """
        async with self.get_session() as session:
            # 构建查询
            if select_fields:
                cols = [getattr(self.model_class, f) for f in select_fields if hasattr(self.model_class, f)]
                if not cols:
                    cols = [self.model_class]
                stmt = select(*cols)
            else:
                stmt = select(self.model_class)  # type: ignore[call-overload]

            filter_exprs = self._build_filters(
                where=where,
                conditions=conditions,
                filters=filters,
            )

            # 应用条件
            if filter_exprs:
                if logic.upper() == "OR":
                    stmt = stmt.where(or_(*filter_exprs))
                else:
                    stmt = stmt.where(and_(*filter_exprs))

            # 处理排序
            if order_by:
                stmt = self._apply_order_by(stmt, order_by)

            # 处理去重
            if distinct:
                stmt = stmt.distinct()

            # 处理 limit 和 offset
            if limit is not None:
                stmt = stmt.limit(limit)
            if offset is not None:
                stmt = stmt.offset(offset)

            result = await session.execute(stmt)

            if select_fields:
                return result.all()  # type: ignore[no-any-return]  # 返回元组列表
            return result.scalars().all()  # type: ignore[no-any-return]

    # ==================== 聚合 / GROUP BY ====================

    @db_operation("query_aggregate")
    async def query_aggregate(
        self,
        aggregates: Dict[str, str],
        group_by: Optional[Union[str, List[str]]] = None,
        having: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        logic: str = "AND",
        order_by: Optional[Union[str, List[str], Dict[str, str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **filters: Any
    ) -> List[Dict[str, Any]]:
        """聚合查询（GROUP BY / HAVING）——返回 dict 列表。

        **独立于 :meth:`query`**：``query()`` 的签名与返回类型完全不变，
        因此这是纯新增能力，100% 向后兼容。

        Args:
            aggregates: 聚合表达式字典，``{别名: 表达式}``。
                表达式可以是：
                - ``"count"`` / ``"count:col"`` —— 计数
                - ``"sum:col"`` / ``"avg:col"`` / ``"min:col"`` / ``"max:col"``
                - ``"avg:col"`` 等价于 ``"mean:col"``
                也接受**原始 SQL 片段**（以 ``func.`` 开头）或 ``sqlalchemy`` 表达式对象。
            group_by: 分组列名（str 或 list）
            having: 分组后的过滤条件，``{列名: 值或操作符 dict}``
                —— 语义与 ``where`` 相同（如 ``{"total": {"gt": 100}}``）
            where: 行级过滤条件（聚合前）——语义与 ``query()`` 相同
            conditions: 操作符条件列表（聚合前）
            logic: 条件逻辑，"AND" 或 "OR"
            order_by: 排序（按聚合结果的别名或分组列）
            limit: 限制数量
            offset: 偏移量
            **filters: 简单等值过滤条件（聚合前）

        Returns:
            dict 列表——每行含分组列与聚合别名

        Examples:
            >>> # 每个 series 的最新值
            >>> await repo.query_aggregate(
            ...     aggregates={"latest": "max:created_at"},
            ...     group_by="series",
            ... )
            [{'series': 'btc', 'latest': datetime(...)}, ...]

            >>> # 每个 series 的条数 + 平均值，并过滤
            >>> await repo.query_aggregate(
            ...     aggregates={"cnt": "count", "avg_price": "avg:price"},
            ...     group_by="series",
            ...     having={"cnt": {"gt": 10}},
            ... )
        """
        if not aggregates:
            raise DatabaseError(
                "query_aggregate 需要至少一个聚合表达式",
                operation="query_aggregate",
            )

        async with self.get_session() as session:
            # ===== 1. 构造 select 列表 =====
            select_cols: List[Any] = []
            alias_map: Dict[str, Any] = {}

            # 分组列先放进 select（顺序即返回 dict 的列序）
            group_cols: List[Any] = []
            group_names: List[str] = []
            if group_by:
                names = [group_by] if isinstance(group_by, str) else list(group_by)
                for name in names:
                    col = self._get_column(name)
                    if col is None:
                        self.logger.warning(f"group_by 的列不存在，已忽略: {name}")
                        continue
                    group_cols.append(col)
                    group_names.append(name)
                    select_cols.append(col)

            # 聚合列
            for alias, expr in aggregates.items():
                agg_expr = self._build_aggregate_expr(expr)
                if agg_expr is None:
                    self.logger.warning(f"无法解析的聚合表达式，已忽略: {alias}={expr}")
                    continue
                labeled = agg_expr.label(alias)
                select_cols.append(labeled)
                alias_map[alias] = labeled

            if not select_cols:
                raise DatabaseError(
                    "query_aggregate 没有可用的列（group_by 与 aggregates 都无效）",
                    operation="query_aggregate",
                )

            stmt = select(*select_cols)
            # 没有分组列时（如全表 count），select(func.count()) 无法自动推断
            # FROM 哪张表——必须显式 select_from，否则 SQL 缺少 FROM 子句
            if not group_cols:
                stmt = stmt.select_from(cast(Any, self.model_class))

            # ===== 2. 行级过滤（聚合前）=====
            filter_exprs = self._build_filters(
                where=where,
                conditions=conditions,
                filters=filters,
            )
            stmt = self._apply_filters(stmt, filter_exprs, logic)

            # ===== 3. GROUP BY =====
            if group_cols:
                stmt = stmt.group_by(*group_cols)

            # ===== 4. HAVING（聚合后过滤）=====
            if having:
                having_exprs = self._build_having_exprs(having, alias_map)
                if having_exprs:
                    stmt = stmt.having(and_(*having_exprs))

            # ===== 5. 排序 / 分页 =====
            if order_by:
                stmt = self._apply_aggregate_order_by(stmt, order_by, alias_map, group_names)
            if limit is not None:
                stmt = stmt.limit(limit)
            if offset is not None:
                stmt = stmt.offset(offset)

            result = await session.execute(stmt)
            rows = result.mappings().all()
            return [dict(row) for row in rows]

    def _build_aggregate_expr(self, expr: Any) -> Any:  # noqa: ANN401
        """解析单个聚合表达式。

        支持的写法：
          - ``"count"`` / ``"count:col"``
          - ``"sum:col"`` / ``"avg:col"`` / ``"mean:col"`` / ``"min:col"`` / ``"max:col"``
          - 原始 SQL 片段（以 ``func.`` 开头）——直接使用
          - SQLAlchemy 表达式对象——直接使用

        Returns:
            SQLAlchemy 表达式；无法解析时 None
        """
        # 已经是表达式对象
        if not isinstance(expr, str):
            return expr

        raw = expr.strip()
        if not raw:
            return None

        # 原始 SQL 片段：func.count() / func.sum(Model.col) 等
        if raw.startswith("func."):
            try:
                # 只允许 func.xxx(...) 这种受限形式——避免 eval 的任意代码执行风险
                return self._parse_func_expression(raw)
            except Exception as e:
                self.logger.warning(f"聚合表达式解析失败: {expr}（{e}）")
                return None

        # "agg:col" 形式
        if ":" in raw:
            agg_name, col_name = raw.split(":", 1)
            agg_name = agg_name.strip().lower()
            col_name = col_name.strip()
            col = self._get_column(col_name) if col_name else None
            if col is None:
                self.logger.warning(f"聚合表达式的列不存在: {expr}")
                return None
            agg_func = cast(
                Optional[Callable[..., Any]],
                {
                    "count": func.count,
                    "sum": func.sum,
                    "avg": func.avg,
                    "mean": func.avg,
                    "min": func.min,
                    "max": func.max,
                }.get(agg_name),
            )
            if agg_func is None:
                self.logger.warning(f"不支持的聚合函数: {agg_name}")
                return None
            return agg_func(col)

        # 裸 "count" —— 计数
        if raw.lower() == "count":
            return func.count()

        self.logger.warning(f"无法解析的聚合表达式: {expr}")
        return None

    def _parse_func_expression(self, raw: str) -> Any:  # noqa: ANN401
        """解析 ``func.xxx(...)`` 形式的聚合表达式——**不用 eval**。

        只支持受限语法，避免任意代码执行：

            func.count()
            func.count(col)
            func.sum(col)
            func.avg(col)

        Args:
            raw: 形如 ``func.sum(price)`` 的字符串

        Returns:
            SQLAlchemy 表达式

        Raises:
            ValueError: 语法不合法或引用了未知列
        """
        text = raw.strip()
        if not text.startswith("func.") or not text.endswith(")"):
            raise ValueError(f"不支持的聚合表达式格式: {raw}")

        inner = text[len("func."):-1]
        func_name, _, arg = inner.partition("(")
        func_name = func_name.strip()
        arg = arg.strip()

        agg_func = cast(
            Optional[Callable[..., Any]],
            {
                "count": func.count,
                "sum": func.sum,
                "avg": func.avg,
                "mean": func.avg,
                "min": func.min,
                "max": func.max,
            }.get(func_name),
        )
        if agg_func is None:
            raise ValueError(f"不支持的聚合函数: {func_name}")

        if not arg:
            return agg_func()

        # 只允许纯列名（不含括号 / 引号 / 空格）——防止注入任意表达式
        if not arg.replace("_", "").isalnum():
            raise ValueError(f"聚合表达式的参数只能是列名，收到: {arg}")

        col = self._get_column(arg)
        if col is None:
            raise ValueError(f"聚合表达式的列不存在: {arg}")
        return agg_func(col)

    def _build_having_exprs(
        self,
        having: Dict[str, Any],
        alias_map: Dict[str, Any],
    ) -> List[Any]:
        """构造 HAVING 条件。

        key 可以是聚合别名（``{"cnt": {"gt": 10}}``）或真实列名
        （``{"price": {"gt": 100}}``）——真实列名会转成 ``func.sum(col)``。
        """
        exprs: List[Any] = []
        for key, value in having.items():
            target = alias_map.get(key)
            if target is None:
                # 不是聚合别名——按「对该列求和后比较」处理
                col = self._get_column(key)
                if col is None:
                    self.logger.warning(f"having 的列不存在，已忽略: {key}")
                    continue
                target = func.sum(col)

            if isinstance(value, dict):
                for op, val in value.items():
                    expr = self._build_filter_expr(target, op, val)
                    if expr is not None:
                        exprs.append(expr)
            else:
                expr = self._build_filter_expr(target, "eq", value)
                if expr is not None:
                    exprs.append(expr)
        return exprs

    def _apply_aggregate_order_by(
        self,
        stmt: Any,
        order_by: Any,
        alias_map: Dict[str, Any],
        group_names: List[str],
    ) -> Any:
        """聚合查询的排序——支持按聚合别名或分组列。"""
        items: List[Any]
        if isinstance(order_by, str):
            items = [order_by]
        elif isinstance(order_by, dict):
            items = [f"{k} {v}" for k, v in order_by.items()]
        elif isinstance(order_by, list):
            items = list(order_by)
        else:
            return stmt

        for item in items:
            if not isinstance(item, str):
                continue
            parts = item.strip().split()
            if not parts:
                continue
            name = parts[0]
            desc = len(parts) > 1 and parts[1].upper() == "DESC"

            target = alias_map.get(name)
            if target is None and name in group_names:
                target = self._get_column(name)
            if target is None:
                self.logger.warning(f"order_by 的列不存在，已忽略: {name}")
                continue
            stmt = stmt.order_by(target.desc() if desc else target)
        return stmt

    # ==================== 分组取最新记录 ====================
    @db_operation("get_latest_per_group")
    async def get_latest_per_group(
        self,
        group_column: str,
        order_column: str,
        limit: int = 1,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        logic: str = "AND",
        extra_order: Optional[Union[str, List[str], Dict[str, str]]] = None,
        **filters: Any
    ) -> List[Any]:
        """
        按分组取最新（或最旧）的 N 条记录
        
        Args:
            group_column: 分组字段名
            order_column: 排序字段名（通常为 created_at 等时间字段）
            limit: 每个分组取几条（默认 1）
            where: 等值条件字典
            conditions: 操作符条件列表
            logic: 条件逻辑
            extra_order: 额外的排序（在窗口函数内使用）
            **filters: 简单等值过滤条件
            
        Returns:
            每个分组最新的 N 条记录列表
        """
        from sqlalchemy import func, select, and_, or_
        
        # 验证字段是否存在
        if not hasattr(self.model_class, group_column):
            raise ValueError(f"字段 '{group_column}' 不存在于模型 {self.model_class.__name__}")
        if not hasattr(self.model_class, order_column):
            raise ValueError(f"字段 '{order_column}' 不存在于模型 {self.model_class.__name__}")
        
        async with self.get_session() as session:
            # 构建过滤条件
            filter_exprs = []
            
            if where:
                for key, value in where.items():
                    if hasattr(self.model_class, key):
                        filter_exprs.append(getattr(self.model_class, key) == value)
            
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    filter_exprs.append(getattr(self.model_class, key) == value)
            
            if conditions:
                for field, operator, value in conditions:
                    if hasattr(self.model_class, field):
                        column = getattr(self.model_class, field)
                        expr = self._build_filter_expr(column, operator, value)
                        if expr is not None:
                            filter_exprs.append(expr)
            
            # 构建窗口函数
            group_attr = getattr(self.model_class, group_column)
            order_attr = getattr(self.model_class, order_column)
            
            # 构建排序表达式
            order_exprs = [order_attr.desc()]
            
            # 如果有额外排序
            if extra_order:
                if isinstance(extra_order, str):
                    parts = extra_order.strip().split()
                    if len(parts) == 2 and parts[1].upper() in ["ASC", "DESC"]:
                        if hasattr(self.model_class, parts[0]):
                            col = getattr(self.model_class, parts[0])
                            order_exprs.append(col.desc() if parts[1].upper() == "DESC" else col)
                    else:
                        if hasattr(self.model_class, extra_order):
                            order_exprs.append(getattr(self.model_class, extra_order))
                elif isinstance(extra_order, dict):
                    for field, direction in extra_order.items():
                        if hasattr(self.model_class, field):
                            col = getattr(self.model_class, field)
                            order_exprs.append(col.desc() if direction.upper() == "DESC" else col)
                elif isinstance(extra_order, list):
                    for item in extra_order:
                        if isinstance(item, str):
                            parts = item.strip().split()
                            if len(parts) == 2 and parts[1].upper() in ["ASC", "DESC"]:
                                if hasattr(self.model_class, parts[0]):
                                    col = getattr(self.model_class, parts[0])
                                    order_exprs.append(col.desc() if parts[1].upper() == "DESC" else col)
                            else:
                                if hasattr(self.model_class, item):
                                    order_exprs.append(getattr(self.model_class, item))
                        elif isinstance(item, dict):
                            for field, direction in item.items():
                                if hasattr(self.model_class, field):
                                    col = getattr(self.model_class, field)
                                    order_exprs.append(col.desc() if direction.upper() == "DESC" else col)
            
            # 关键修复：使用 CTE 方式，并确保 SELECT 子句正确
            # 第一步：创建带窗口函数的子查询
            # 选择所有列 + 窗口函数
            cols = [getattr(self.model_class, c.name) for c in self.model_class.__table__.columns]
            
            # 构建子查询
            subq = select(
                *cols,
                func.row_number().over(
                    partition_by=group_attr,
                    order_by=order_exprs
                ).label("rn")
            ).select_from(self.model_class)
            
            # 应用过滤条件
            if filter_exprs:
                if logic.upper() == "OR":
                    subq = subq.where(or_(*filter_exprs))
                else:
                    subq = subq.where(and_(*filter_exprs))
            
            subq = subq.subquery("ranked")
            
            # 第二步：从子查询中选择 rn <= limit 的记录
            # 使用子查询的所有列，并映射回 ORM 对象
            stmt = select(self.model_class).from_statement(
                select(*[subq.c[col.name] for col in self.model_class.__table__.columns])
                .select_from(subq)
                .where(subq.c.rn <= limit)
            )
            
            result = await session.execute(stmt)
            return result.scalars().all()  # type: ignore[no-any-return]

    # ==================== 分页查询 ====================
    
    @db_operation("paginate")
    async def paginate(
        self,
        page: int = 1,
        page_size: int = 20,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        logic: str = "AND",
        order_by: Optional[Union[str, List[str], Dict[str, str]]] = None,
        **filters: Any
    ) -> PageResult[ModelType]:
        """
        分页查询

        Args:
            page: 页码，从1开始
            page_size: 每页数量
            where: 等值条件字典
            conditions: 操作符条件列表 [(field, op, value), ...]
            logic: 条件逻辑，"AND" 或 "OR"
            order_by: 排序
            **filters: 简单等值过滤条件
        """
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 20

        async with self.get_session() as session:
            # 构建过滤条件——与 query() 共用 _build_filters，
            # 因此 where 的三种形式（等值 / 操作符 dict / conditions 列表）能力对等
            filter_exprs = self._build_filters(
                where=where,
                conditions=conditions,
                filters=filters,
            )

            # 查询总数
            count_stmt = select(func.count()).select_from(self.model_class)  # type: ignore[arg-type]
            count_stmt = self._apply_filters(count_stmt, filter_exprs, logic)

            total_result = await session.execute(count_stmt)
            total = total_result.scalar() or 0

            # 查询分页数据
            stmt = select(self.model_class)  # type: ignore[call-overload]
            stmt = self._apply_filters(stmt, filter_exprs, logic)

            if order_by:
                stmt = self._apply_order_by(stmt, order_by)

            stmt = stmt.offset((page - 1) * page_size).limit(page_size)
            
            result = await session.execute(stmt)
            items = result.scalars().all()
            
            return PageResult(
                items=items,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=(total + page_size - 1) // page_size if total > 0 else 1
            )
    
    # ==================== 批量操作 ====================
    
    @db_operation("bulk_create")
    async def bulk_create(self, items: List[Dict[str, Any]]) -> List[Any]:
        """批量创建"""
        if not items:
            return []
        
        async with self.get_session() as session:
            instances = [self.model_class(**item) for item in items]  # type: ignore[operator]
            session.add_all(instances)
            await session.flush()
            return instances
    
    @db_operation("bulk_update")
    async def bulk_update(
        self,
        ids: List[Any],
        updates: Dict[str, Any],
        batch_size: int = 100
    ) -> int:
        """
        批量更新
        
        Args:
            ids: ID列表
            updates: 更新字段字典
            batch_size: 批次大小
        """
        if not ids or not updates:
            return 0
        
        async with self.get_session() as session:
            updated_count = 0
            
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                
                stmt = select(self.model_class).where(  # type: ignore[call-overload]
                    self.model_class.id.in_(batch_ids)  # type: ignore[attr-defined]
                )
                result = await session.execute(stmt)
                instances = result.scalars().all()
                
                for instance in instances:
                    for key, value in updates.items():
                        if hasattr(instance, key):
                            setattr(instance, key, value)
                    updated_count += 1
                
                await session.flush()
            
            return updated_count
    
    @db_operation("bulk_delete")
    async def bulk_delete(self, ids: List[Any]) -> int:
        """批量删除"""
        if not ids:
            return 0
        
        async with self.get_session() as session:
            stmt = delete(self.model_class).where(  # type: ignore[arg-type]
                self.model_class.id.in_(ids)  # type: ignore[attr-defined]
            )
            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount  # type: ignore[no-any-return]
    
    # ==================== 存在性检查 ====================
    
    @db_operation("exists")
    async def exists(
        self,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        logic: str = "AND",
        **filters: Any
    ) -> bool:
        """检查记录是否存在。

        Args:
            where: 等值 / 带操作符的条件字典（与 query() 语义一致）
            conditions: 操作符条件列表 ``[(field, op, value), ...]``
            logic: 条件逻辑，"AND" 或 "OR"
            **filters: 简单等值过滤条件
        """
        async with self.get_session() as session:
            stmt = select(self.model_class)  # type: ignore[call-overload]

            filter_exprs = self._build_filters(
                where=where,
                conditions=conditions,
                filters=filters,
            )
            stmt = self._apply_filters(stmt, filter_exprs, logic)

            stmt = stmt.limit(1)
            result = await session.execute(stmt)
            return result.first() is not None
    
    # ==================== 工具方法 ====================
    
    def _build_filters(
        self,
        where: Optional[Dict[str, Any]] = None,
        conditions: Optional[List[tuple]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """统一构造过滤表达式——``query`` / ``paginate`` / ``count`` 共用。

        三种入参形式**能力对等**（这是修复前最大的 API 陷阱）：

        1. ``where={"status": "active"}`` —— 等值
        2. ``where={"age": {"gt": 18}}`` —— 带操作符的 dict（值是 dict 时）
        3. ``conditions=[("age", "gt", 18)]`` —— 操作符列表
        4. ``filters={"status": "active"}`` —— 关键字参数形式的等值

        Args:
            where: 等值 / 带操作符的条件字典
            conditions: ``(field, operator, value)`` 三元组列表
            filters: 简单等值条件

        Returns:
            SQLAlchemy 表达式列表（已按 ``logic`` 由调用方合并）
        """
        exprs: List[Any] = []

        # where：值是 dict → 操作符形式；否则等值
        if where:
            for key, value in where.items():
                column = self._get_column(key)
                if column is None:
                    continue
                if isinstance(value, dict):
                    # {"gt": 18, "lt": 65} —— 逐个操作符构造
                    for op, val in value.items():
                        expr = self._build_filter_expr(column, op, val)
                        if expr is not None:
                            exprs.append(expr)
                else:
                    # 等值——走 _build_filter_expr 以统一 NULL 语义
                    # （value=None → IS NULL，而不是 == None）
                    expr = self._build_filter_expr(column, "eq", value)
                    if expr is not None:
                        exprs.append(expr)

        # filters：简单等值
        if filters:
            for key, value in filters.items():
                column = self._get_column(key)
                if column is None:
                    continue
                expr = self._build_filter_expr(column, "eq", value)
                if expr is not None:
                    exprs.append(expr)

        # conditions：显式操作符列表
        if conditions:
            for field, operator, value in conditions:
                column = self._get_column(field)
                if column is None:
                    continue
                expr = self._build_filter_expr(column, operator, value)
                if expr is not None:
                    exprs.append(expr)

        return exprs

    def _get_column(self, key: str) -> Any:  # noqa: ANN401
        """取模型列——不是列就返回 None。

        用 ``__table__.columns`` 白名单校验，而不是 ``hasattr(model, key)``：
          - ``hasattr`` 会命中 ``metadata`` / ``registry`` / ``__table__`` 等
            非列属性，拿它们做比较会抛 ``ArgumentError``
          - ``__table__.columns`` 是字典查找，O(1) 且精确

        Args:
            key: 列名

        Returns:
            SQLAlchemy 列对象；不是有效列则 None
        """
        table = getattr(self.model_class, "__table__", None)
        if table is not None:
            columns = getattr(table, "columns", None)
            if columns is not None:
                return columns.get(key)
            return None
        # 没有 __table__（非 ORM 模型）——退回 getattr
        return getattr(self.model_class, key, None)

    def _apply_filters(self, stmt: Any, exprs: List[Any], logic: str = "AND") -> Any:
        """把过滤表达式应用到语句上（按 AND / OR 合并）。"""
        if not exprs:
            return stmt
        if logic.upper() == "OR":
            return stmt.where(or_(*exprs))
        return stmt.where(and_(*exprs))

    def _apply_order_by(self, stmt: Any, order_by: Any) -> Any:
        """应用排序条件"""
        if isinstance(order_by, str):
            # 支持 "created_at desc" 格式
            parts = order_by.strip().split()
            if len(parts) == 2 and parts[1].upper() in ["ASC", "DESC"]:
                if hasattr(self.model_class, parts[0]):
                    column = getattr(self.model_class, parts[0])
                    if parts[1].upper() == "DESC":
                        stmt = stmt.order_by(column.desc())
                    else:
                        stmt = stmt.order_by(column)
            else:
                if hasattr(self.model_class, order_by):
                    stmt = stmt.order_by(getattr(self.model_class, order_by))
        elif isinstance(order_by, dict):
            # {"created_at": "desc", "id": "asc"}
            for field, direction in order_by.items():
                if hasattr(self.model_class, field):
                    column = getattr(self.model_class, field)
                    if direction.upper() == "DESC":
                        stmt = stmt.order_by(column.desc())
                    else:
                        stmt = stmt.order_by(column)
        elif isinstance(order_by, list):
            # ["created_at desc", "id asc"]
            for item in order_by:
                stmt = self._apply_order_by(stmt, item)
        return stmt
    
    def _build_filter_expr(self, column: Any, operator: str, value: Any) -> Any:
        # ===== 1. 处理 value is None =====
        if value is None:
            if operator in ("eq", "is"):
                return column.is_(None)
            if operator in ("ne", "is_not"):
                return column.is_not(None)
            if operator == "is_null":
                return column.is_(None)
            if operator == "is_not_null":
                return column.is_not(None)
            # 对于 >, <, >=, <=, like, ilike, in, not_in, between
            # None 比较永远返回 False——用 sqlalchemy.false() 明确表达
            # （旧写法 `IS NULL AND IS NOT NULL` 依赖三值逻辑，脆弱且难读）
            if operator in ("gt", "gte", "lt", "lte", "like", "ilike", "in", "not_in", "between"):
                return false()
            return None

        # ===== 2. 处理布尔值 =====
        if isinstance(value, bool):
            if operator in ("eq", "is"):
                return column.is_(value)
            if operator in ("ne", "is_not"):
                return column.is_not(value)

        # ===== 3. 处理 in/not_in 的嵌套字典格式 =====
        if operator in ("in", "not_in") and isinstance(value, dict):
            # 格式: {"in": [1,2,3]} 或 {"not_in": [1,2,3]}
            for op, vals in value.items():
                if op == "in" and isinstance(vals, (list, tuple)):
                    return column.in_(vals)
                if op == "not_in" and isinstance(vals, (list, tuple)):
                    return column.not_in(vals)
            # 如果没匹配，回退
            return column.in_(value.get("in", [])) if "in" in value else None

        # ===== 4. 正常处理其他值 =====
        # 注意 in / not_in 的空 list 语义：
        #   空 list 意味着「匹配不到任何行」——必须生成恒 False 条件，
        #   而**不是**返回 None（返回 None 会让条件被静默丢弃 → 返回全表！）。
        if operator == "in":
            if isinstance(value, (list, tuple, set, frozenset)):
                return column.in_(list(value)) if value else false()
            self.logger.warning(f"in 操作符需要 list/tuple，收到 {type(value).__name__}")
            return None
        if operator == "not_in":
            if isinstance(value, (list, tuple, set, frozenset)):
                return column.not_in(list(value)) if value else true()
            self.logger.warning(f"not_in 操作符需要 list/tuple，收到 {type(value).__name__}")
            return None

        operators = {
            "eq": column == value,
            "ne": column != value,
            "gt": column > value,
            "gte": column >= value,
            "lt": column < value,
            "lte": column <= value,
            "like": column.like(value),
            "ilike": column.ilike(value),
            "is_null": column.is_(None),
            "is_": column.is_(value),
            "is_not": column.is_not(value),
            "is_not_null": column.is_not(None),
            "between": column.between(value[0], value[1]) if isinstance(value, (list, tuple)) and len(value) == 2 else None,
        }
        result = operators.get(operator)
        if result is None:
            self.logger.warning(f"未知的操作符: {operator}")
        return result
    
    # ==================== 原生连接支持 ====================
    
    async def get_raw_connection(self, session: AsyncSession) -> asyncpg.Connection:
        """获取原始的 asyncpg 连接"""
        async_conn = await session.connection()
        # 优先使用官方 API，兼容不同 SQLAlchemy 版本
        try:
            raw_conn = await async_conn.get_raw_connection()
            return raw_conn.driver_connection
        except (AttributeError, NotImplementedError):
            sync_conn = async_conn.sync_connection
            if sync_conn is None:
                raise RuntimeError("无法获取同步连接")
            return sync_conn.connection.driver_connection
    
    async def copy_records_to_table(
        self,
        session: AsyncSession,
        table_name: str,
        records: List[tuple],
        columns: List[str]
    ) -> int:
        """
        通用 COPY 方法，用于大批量数据导入
        
        Args:
            session: 数据库会话
            table_name: 表名
            records: 记录列表
            columns: 列名列表
            
        Returns:
            导入的记录数
        """
        if not records:
            return 0
        
        raw_conn = await self.get_raw_connection(session)
        
        await raw_conn.copy_records_to_table(
            table_name,
            records=records,
            columns=columns
        )
        return len(records)
    

async def execute_query(
    query_func: Callable,
    operation_name: str = "custom_query",
    db_key: Optional[str] = None,
) -> Any:
    """执行自定义查询函数"""
    @db_operation(operation_name, db_key=db_key)
    async def wrapper() -> None:
        return await query_func()  # type: ignore[no-any-return]

    return await wrapper()

