#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义异常类
"""

from typing import Optional, Any


class AppException(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        error_msg = f"[{self.error_code}] {self.message}" if self.error_code else self.message
        if self.details:
            error_msg += f" | Details: {self.details}"
        return error_msg


class ConfigurationError(AppException):
    """配置错误"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class DatabaseError(AppException):
    """数据库错误"""
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "DB_ERROR", details)
        self.operation = operation


class RedisError(AppException):
    """Redis错误"""
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "REDIS_ERROR", details)
        self.operation = operation


class MessageQueueError(AppException):
    """消息队列错误"""
    
    def __init__(self, message: str, queue_name: Optional[str] = None, operation: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "MQ_ERROR", details)
        self.queue_name = queue_name
        self.operation = operation


class SchedulerError(AppException):
    """调度器错误"""
    
    def __init__(self, message: str, job_id: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "SCHEDULER_ERROR", details)
        self.job_id = job_id


class DataProducerError(AppException):
    """数据生产者错误"""
    
    def __init__(self, message: str, source: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "PRODUCER_ERROR", details)
        self.source = source


class DataConsumerError(AppException):
    """数据消费者错误"""
    
    def __init__(self, message: str, task_type: Optional[str] = None, task_id: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "CONSUMER_ERROR", details)
        self.task_type = task_type
        self.task_id = task_id


class APIError(AppException):
    """API错误"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, endpoint: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message, "API_ERROR", details)
        self.status_code = status_code
        self.endpoint = endpoint


class BusinessException(AppException):
    """
    业务异常 - 用户可自行处理
    例如：知识点不存在、参数不合法、数据重复等
    错误码: BIZ_ERROR
    """
    def __init__(
        self, 
        message: str, 
        suggestion: Optional[str] = None,
        data: Optional[Any] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message, "BIZ_ERROR", details)
        self.suggestion = suggestion
        self.data = data


class SystemException(AppException):
    """
    系统异常 - 用户无法自行处理
    例如：数据库连接失败、代码bug、依赖服务不可用等
    错误码: SYS_ERROR
    """
    def __init__(
        self, 
        message: str = "系统异常，请稍后重试", 
        data: Optional[Any] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message, "SYS_ERROR", details)
        self.data = data


class AuthException(AppException):
    """
    权限异常 - 需要用户授权
    例如：未绑定微信号、权限不足等
    错误码: AUTH_ERROR
    """
    def __init__(
        self, 
        message: str = "请先绑定微信号", 
        suggestion: Optional[str] = None,
        data: Optional[Any] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message, "AUTH_ERROR", details)
        self.suggestion = suggestion
        self.data = data

