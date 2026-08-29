#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置管理模块

使用方式:
    from infra.config import get_config, init_config
    
    # 初始化配置（应用启动时）
    config = init_config()
    
    # 获取配置（任何地方）
    config = get_config()
    redis_host = config.get('redis', {}).get('host')
"""

from .loader import ConfigLoader, get_config, get_config_path, init_config, reload_config

__all__ = [
    'ConfigLoader',
    'get_config',
    'get_config_path',
    'init_config',
    'reload_config',
]

