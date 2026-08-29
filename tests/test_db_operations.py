#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.db.operations.db_operation 装饰器测试

覆盖：
1. 正常返回：成功路径透传结果
2. SQLAlchemyError → DatabaseError 包装
3. 普通 Exception 透传（不包装）
4. 无论成败都 record_db_query（finally）
5. log_performance=False 时不调用 record_db_query
6. > 1.0s 时记 WARNING
7. db_type 参数固定 vs 动态从 self.db_type 取
8. operation_name 出现在 DatabaseError 的 details

注意：装饰器内部用 args[0].db_type——测试需要一个有 db_type 属性的对象当 self。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from infra.db.operations import db_operation
from infra.exceptions import DatabaseError


# ============================================================
# 辅助：一个有 db_type 属性的简单类
# ============================================================
class FakeRepo:
    """假装是 repository 实例。"""

    def __init__(self, db_type: str = "business"):
        self.db_type = db_type


# ============================================================
# 1. 正常路径
# ============================================================
class TestNormalPath:
    async def test_returns_func_result(self):
        @db_operation("get_user")
        async def op(self):
            return {"id": 1, "name": "alice"}

        repo = FakeRepo()
        result = await op(repo)
        assert result == {"id": 1, "name": "alice"}

    async def test_passes_args_and_kwargs(self):
        @db_operation("echo")
        async def op(self, x, y=0):
            return x + y

        repo = FakeRepo()
        result = await op(repo, 3, y=4)
        assert result == 7


# ============================================================
# 2. SQLAlchemyError → DatabaseError 包装
# ============================================================
class TestSQLAlchemyError:
    async def test_sqlalchemy_error_raises_database_error(self):
        @db_operation("insert")
        async def op(self):
            raise SQLAlchemyError("connection lost")

        repo = FakeRepo()
        with pytest.raises(DatabaseError) as exc_info:
            await op(repo)
        assert "connection lost" in str(exc_info.value)
        assert exc_info.value.error_code == "DB_ERROR"
        assert exc_info.value.operation == "insert"
        # details 应含 function name + db_type
        assert exc_info.value.details["function"] == "op"
        assert exc_info.value.details["db_type"] == "business"

    async def test_operational_error_wrapped(self):
        """OperationalError 是 SQLAlchemyError 子类。"""
        @db_operation("query")
        async def op(self):
            raise OperationalError("SELECT 1", {}, Exception("boom"))

        repo = FakeRepo(db_type="timeseries")
        with pytest.raises(DatabaseError) as exc_info:
            await op(repo)
        assert exc_info.value.operation == "query"
        assert exc_info.value.details["db_type"] == "timeseries"


# ============================================================
# 3. 普通 Exception 透传
# ============================================================
class TestGenericException:
    async def test_value_error_passes_through(self):
        @db_operation("op")
        async def fn(self):
            raise ValueError("not a db error")

        repo = FakeRepo()
        with pytest.raises(ValueError, match="not a db error"):
            await fn(repo)

    async def test_runtime_error_passes_through(self):
        @db_operation("op")
        async def fn(self):
            raise RuntimeError("oops")

        repo = FakeRepo()
        with pytest.raises(RuntimeError, match="oops"):
            await fn(repo)


# ============================================================
# 4. 无论成败都 record_db_query
# ============================================================
class TestRecordQuery:
    async def test_success_records_query(self):
        with patch("infra.db.operations.record_db_query") as mock_record:
            @db_operation("op")
            async def fn(self):
                return "ok"

            await fn(FakeRepo())
            mock_record.assert_called_once()
            # 第一个参数是 response_time（float），第二个是 success=True
            args = mock_record.call_args[0]
            assert isinstance(args[0], float)
            assert args[1] is True

    async def test_failure_records_unsuccessful_query(self):
        with patch("infra.db.operations.record_db_query") as mock_record:
            @db_operation("op")
            async def fn(self):
                raise SQLAlchemyError("x")

            with pytest.raises(DatabaseError):
                await fn(FakeRepo())
            # 即使抛错，finally 仍记录
            mock_record.assert_called_once()
            assert mock_record.call_args[0][1] is False

    async def test_passes_dynamic_db_type(self):
        with patch("infra.db.operations.record_db_query") as mock_record:
            @db_operation("op")
            async def fn(self):
                return "ok"

            await fn(FakeRepo(db_type="timeseries"))
            # success 仍记录（db_type 不影响 record）
            mock_record.assert_called_once()


# ============================================================
# 5. log_performance=False
# ============================================================
class TestLogPerformanceDisabled:
    async def test_no_record_when_disabled(self):
        with patch("infra.db.operations.record_db_query") as mock_record:
            @db_operation("op", log_performance=False)
            async def fn(self):
                return "ok"

            await fn(FakeRepo())
            mock_record.assert_not_called()

    async def test_no_warning_when_disabled(self, caplog):
        """log_performance=False 也不应记录 WARNING（即便 > 1s）。"""
        import time as _time
        import logging

        with patch("infra.db.operations.record_db_query"):
            with caplog.at_level(logging.WARNING):
                @db_operation("slow_op", log_performance=False)
                async def fn(self):
                    _time.sleep(1.2)  # 触发 > 1s
                    return "ok"

                await fn(FakeRepo())
            # 没有 "耗时较长" 警告
            slow_warnings = [r for r in caplog.records if "耗时较长" in r.message]
            assert slow_warnings == []


# ============================================================
# 6. > 1.0s 警告
# ============================================================
class TestSlowOperationWarning:
    async def test_slow_op_warning(self, caplog):
        import time as _time
        import logging

        with patch("infra.db.operations.record_db_query"):
            with caplog.at_level(logging.WARNING, logger="infra.db.operations"):
                @db_operation("slow_op")
                async def fn(self):
                    _time.sleep(1.1)  # > 1.0s 触发
                    return "ok"

                await fn(FakeRepo())
            slow_warnings = [r for r in caplog.records if "耗时较长" in r.message]
            assert len(slow_warnings) >= 1
            assert "slow_op" in slow_warnings[0].message

    async def test_fast_op_no_warning(self, caplog):
        import logging
        with patch("infra.db.operations.record_db_query"):
            with caplog.at_level(logging.WARNING):
                @db_operation("fast_op")
                async def fn(self):
                    return "ok"

                await fn(FakeRepo())
            slow_warnings = [r for r in caplog.records if "耗时较长" in r.message]
            assert slow_warnings == []


# ============================================================
# 7. db_type 参数
# ============================================================
class TestDbType:
    async def test_explicit_db_type_overrides_self(self):
        """装饰器 db_type 参数优先于 self.db_type。"""
        with patch("infra.db.operations.record_db_query") as mock_record:
            @db_operation("op", db_type="timeseries")
            async def fn(self):
                return "ok"

            # self.db_type='business' 但装饰器传 'timeseries'
            await fn(FakeRepo(db_type="business"))
            # record 调用不依赖 db_type，只验证调用发生
            mock_record.assert_called_once()

    async def test_db_type_in_database_error_details(self):
        @db_operation("op", db_type="timeseries")
        async def fn(self):
            raise SQLAlchemyError("x")

        with pytest.raises(DatabaseError) as exc_info:
            await fn(FakeRepo(db_type="business"))  # self 的是 business
        # 装饰器参数优先
        assert exc_info.value.details["db_type"] == "timeseries"

    async def test_dynamic_db_type_from_self(self):
        """不传 db_type → 从 self.db_type 取。"""
        @db_operation("op")  # 不传 db_type
        async def fn(self):
            raise SQLAlchemyError("x")

        with pytest.raises(DatabaseError) as exc_info:
            await fn(FakeRepo(db_type="timeseries"))
        assert exc_info.value.details["db_type"] == "timeseries"

    async def test_dynamic_db_type_missing_raises(self):
        """不传 db_type 且 self 没 db_type → AttributeError。"""
        @db_operation("op")
        async def fn(self):
            return "ok"

        class NoDbType:
            pass

        with pytest.raises(AttributeError):
            await fn(NoDbType())


# ============================================================
# 8. functools.wraps 保留原函数元信息
# ============================================================
class TestWraps:
    def test_func_name_preserved(self):
        @db_operation("op")
        async def my_special_op(self):
            """My docstring."""
            return "ok"

        assert my_special_op.__name__ == "my_special_op"
        assert "My docstring." in (my_special_op.__doc__ or "")


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
