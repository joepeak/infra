#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消息队列模块
基于 Redis Stream 实现
"""

from .stream_mq import (
    RedisStreamMQ,
    init_mq,
    get_mq,
    MessagePriority,
    QueueConfig,
    QueueName,
)

__all__ = [
    'RedisStreamMQ',
    'init_mq',
    'get_mq',
    'MessagePriority',
    'QueueConfig',
    'QueueName',
]