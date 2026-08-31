"""infra.utils.error_handler：通用异常处理工具（20260829 升级）。

采纳 dramacraft 设计：
- 装饰器：handle_exceptions（同步）/ async_handle_exceptions（异步）
- 上下文管理器：ExceptionContext（with 语句局部异常处理）
- 可配置：log_level / default_return / reraise / exception_types

设计要点：
- 保留原异常（不 return None）——调用方 try/except 清晰
- exception_types 过滤——只捕获指定类型（防 KeyError/ValueError 等程序错误被重试）
- AppException 扩展——记录 error_code 和 details
"""
import functools
import traceback
from typing import (Callable, Any, Dict, Optional, Type, Union)

from infra.logger import get_logger
from infra.exceptions import AppException


def _build_exc_info(func_name: str, module_name: str, exc: BaseException) -> Dict[str, Any]:
    """构造异常信息字典。"""
    exc_info: Dict[str, Any] = {
        'function': func_name,
        'module_name': module_name,
        'exception_type': type(exc).__name__,
        'exception_message': str(exc),
        'traceback': traceback.format_exc(),
    }
    if isinstance(exc, AppException):
        # 类型：error_code 一定是 str，details 是 Dict[str, Any]
        details: Dict[str, Any] = dict(exc.details) if exc.details else {}
        exc_info['error_code'] = str(exc.error_code)
        exc_info['details'] = details
    return exc_info


def _log_exception(logger: Any, log_level: str, func_name: str, exc: BaseException, exc_info: Dict[str, Any]) -> None:
    """按 log_level 等级记录异常。"""
    log_method = getattr(logger, log_level.lower(), logger.error)
    log_method(f"Exception in {func_name}: {str(exc)}", extra=exc_info)


def _should_capture(exc: BaseException, exception_types: Optional[Union[Type[Exception], tuple]]) -> bool:
    """判断异常是否应被捕获。"""
    if exception_types is None:
        return True
    return isinstance(exc, exception_types)


def handle_exceptions(
    default_return: Any = None,
    log_level: str = "ERROR",
    reraise: bool = False,
    exception_types: Optional[Union[Type[Exception], tuple]] = None,
) -> Callable:  # noqa: F811
    """
    同步异常处理装饰器。

    Args:
        default_return: 异常发生时的默认返回值
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        reraise: 是否重新抛出原异常（即使捕获后）
        exception_types: 要捕获的异常类型——None=全部；或 Type 或 tuple

    Example:
        @handle_exceptions(default_return="fallback", reraise=True, exception_types=ValueError)
        def my_func(): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 判断是否捕获此异常
                if not _should_capture(e, exception_types):
                    if reraise:
                        raise
                    return default_return

                exc_info = _build_exc_info(func.__name__, func.__module__, e)
                _log_exception(logger, log_level, func.__name__, e, exc_info)

                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def async_handle_exceptions(
    default_return: Any = None,
    log_level: str = "ERROR",
    reraise: bool = False,
    exception_types: Optional[Union[Type[Exception], tuple]] = None,
) -> Callable:  # noqa: F811
    """
    异步异常处理装饰器。

    Args: 同 handle_exceptions。
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not _should_capture(e, exception_types):
                    if reraise:
                        raise
                    return default_return

                exc_info = _build_exc_info(func.__name__, func.__module__, e)
                _log_exception(logger, log_level, func.__name__, e, exc_info)

                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


class ExceptionContext:
    """异常上下文管理器（with 语句局部异常处理）。

    用法：
        with ExceptionContext(operation_name="my_op", default_return=None) as ctx:
            do_something()
        # 异常被捕获、记录、抑制
    """

    def __init__(
        self,
        operation_name: str,
        default_return: Any = None,
        log_level: str = "ERROR",
        reraise: bool = False,
        exception_types: Optional[Union[Type[Exception], tuple]] = None,
    ) -> None:
        self.operation_name = operation_name
        self.default_return = default_return
        self.log_level = log_level
        self.reraise = reraise
        self.exception_types = exception_types
        self.logger = get_logger(self.__class__.__module__)

    def __enter__(self) -> "ErrorContext":  # type: ignore[name-defined]
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Any
    ) -> bool:
        if exc_type is None:
            return True  # 无异常

        # narrow Optional[BaseException] -> BaseException
        assert exc_val is not None
        # 判断是否捕获
        if not _should_capture(exc_val, self.exception_types):
            if self.reraise:
                return False  # 重新抛出
            return True  # 抑制

        # 记录异常
        exc_info: Dict[str, Any] = {
            'operation': self.operation_name,
            'exception_type': exc_type.__name__ if exc_type else "Unknown",
            'exception_message': str(exc_val),
            'traceback': ''.join(traceback.format_exception(exc_type, exc_val, exc_tb)),
        }
        if isinstance(exc_val, AppException):
            exc_info['error_code'] = str(exc_val.error_code)
            exc_info['details'] = dict(exc_val.details) if exc_val.details else {}
        _log_exception(self.logger, self.log_level, self.operation_name, exc_val, exc_info)

        if self.reraise:
            return False  # 重新抛出
        return True  # 抑制


__all__ = [
    "handle_exceptions",
    "async_handle_exceptions",
    "ExceptionContext",
]
