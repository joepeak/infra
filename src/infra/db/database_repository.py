
from typing import Any, AsyncGenerator, Callable, Dict, Generic, List, Optional, TypeVar, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

from sqlalchemy import select, delete, and_, or_, func, text
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
    async def count(self, **filters: Any) -> int:
        """统计记录数"""
        async with self.get_session() as session:
            stmt = select(func.count()).select_from(self.model_class)  # type: ignore[arg-type]
            
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
            
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

            filter_exprs = []

            # 处理 where 等值条件
            if where:
                for key, value in where.items():
                    if hasattr(self.model_class, key):
                        # 如果 value 是 dict，可能是 in/not_in/between 等
                        if isinstance(value, dict):
                            for op, val in value.items():
                                if op in ("in", "not_in") and isinstance(val, (list, tuple)):
                                    filter_exprs.append(getattr(self.model_class, key).in_(val) if op == "in" else getattr(self.model_class, key).not_in(val))
                                elif op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                                    filter_exprs.append(getattr(self.model_class, key).between(val[0], val[1]))
                                elif op in ("gt", "gte", "lt", "lte"):
                                    if val is not None:
                                        col = getattr(self.model_class, key)
                                        if op == "gt": filter_exprs.append(col > val)
                                        elif op == "gte": filter_exprs.append(col >= val)
                                        elif op == "lt": filter_exprs.append(col < val)
                                        elif op == "lte": filter_exprs.append(col <= val)
                                elif op == "like":
                                    filter_exprs.append(getattr(self.model_class, key).like(val))
                                elif op == "ilike":
                                    filter_exprs.append(getattr(self.model_class, key).ilike(val))
                        else:
                            filter_exprs.append(getattr(self.model_class, key) == value)

            # 处理简单 filters（保持兼容）
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    filter_exprs.append(getattr(self.model_class, key) == value)

            # 处理操作符条件
            if conditions:
                for field, operator, value in conditions:
                    if hasattr(self.model_class, field):
                        column = getattr(self.model_class, field)
                        expr = self._build_filter_expr(column, operator, value)
                        if expr is not None:
                            filter_exprs.append(expr)

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

            # 查询总数
            count_stmt = select(func.count()).select_from(self.model_class)  # type: ignore[arg-type]
            if filter_exprs:
                if logic.upper() == "OR":
                    count_stmt = count_stmt.where(or_(*filter_exprs))
                else:
                    count_stmt = count_stmt.where(and_(*filter_exprs))

            total_result = await session.execute(count_stmt)
            total = total_result.scalar() or 0

            # 查询分页数据
            stmt = select(self.model_class)  # type: ignore[call-overload]
            if filter_exprs:
                if logic.upper() == "OR":
                    stmt = stmt.where(or_(*filter_exprs))
                else:
                    stmt = stmt.where(and_(*filter_exprs))

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
    async def exists(self, **filters: Any) -> bool:
        """检查记录是否存在"""
        async with self.get_session() as session:
            stmt = select(self.model_class)  # type: ignore[call-overload]
            
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    stmt = stmt.where(getattr(self.model_class, key) == value)
            
            stmt = stmt.limit(1)
            result = await session.execute(stmt)
            return result.first() is not None
    
    # ==================== 工具方法 ====================
    
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
            # None 比较永远返回 False
            if operator in ("gt", "gte", "lt", "lte", "like", "ilike", "in", "not_in", "between"):
                return column.is_(None) & column.is_not(None)  # 永远为 False
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
        operators = {
            "eq": column == value,
            "ne": column != value,
            "gt": column > value,
            "gte": column >= value,
            "lt": column < value,
            "lte": column <= value,
            "like": column.like(value),
            "ilike": column.ilike(value),
            "in": column.in_(value) if isinstance(value, (list, tuple)) and value else None,
            "not_in": column.not_in(value) if isinstance(value, (list, tuple)) and value else None,
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

