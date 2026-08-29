#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置加载器
配置目录由调用方传入，文件名固定
"""

import os
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

from infra.logger import get_logger

logger = get_logger(__name__)

# 固定的配置文件名
DEFAULT_CONFIG_FILES = ["config.yaml", "jobs.yaml"]

# _config: Optional[Dict[str, Any]] = None
_config_loaded = False


class ConfigLoader:
    """配置加载器（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._config_dir: Optional[Path] = None
        logger.info("配置加载器已创建")
    
    def configure(self, config_dir: Path):
        """
        配置配置目录（由调用方调用）

        Args:
            config_dir: 配置文件目录

        只设置 _config_dir，不主动 load。load 由 init_config()（force_reload）
        或 get_config() 首次调用时触发。
        """
        self._config_dir = Path(config_dir)
        logger.info(f"配置目录已设置: {self._config_dir}")
    
    def _get_env(self) -> str:
        """获取当前环境"""
        env = os.getenv('MACRO_MONITOR_ENV', '')
        if not env:
            env = os.getenv('ENVIRONMENT', '')
        if not env:
            env = 'dev'
        return env.lower()
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载 YAML 文件"""
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                logger.debug(f"已加载: {file_path.name}")
                return data
        except Exception as e:
            logger.error(f"加载失败 {file_path}: {e}")
            return {}
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """加载配置"""
        global  _config_loaded
        
        if _config_loaded and not force_reload:
            return self._config
        
        if self._config_dir is None:
            raise RuntimeError("请先调用 configure() 设置配置目录")
        
        env = self._get_env()
        logger.info(f"加载配置, 目录: {self._config_dir}, 环境: {env}")
        
        merged = {}
        
        # 加载固定配置文件
        for filename in DEFAULT_CONFIG_FILES:
            file_path = self._config_dir / filename
            if file_path.exists():
                merged = self._deep_merge(merged, self._load_yaml(file_path))
                logger.info(f"已加载: {filename}")
        
        # 加载环境配置
        env_file = self._config_dir / f"{env}.yaml"
        if env_file.exists():
            merged = self._deep_merge(merged, self._load_yaml(env_file))
            logger.info(f"已加载: {env}.yaml")
        
        # 加载本地覆盖
        local_file = self._config_dir / "local.yaml"
        if local_file.exists():
            merged = self._deep_merge(merged, self._load_yaml(local_file))
            logger.info("已加载: local.yaml")
        
        self._config = merged
        _config_loaded = True

        logger.info(f"配置加载完成: {list(merged.keys())}")
        return self._config
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取配置（返回深拷贝——调用方可任意修改不影响内部状态）。

        嵌套 dict 也完全隔离。
        """
        if not _config_loaded:
            return self.load_config()
        return copy.deepcopy(self._config)

    def set_config(self, conf: Dict[str, Any]) -> None:
        """
        浅覆盖设置配置（deep merge 语义）。

        与整 dict 替换不同——只覆盖传入的 key，未传入的 key 保留。
        嵌套 dict 递归合并，原值与新值都是 dict 时递归。

        例：
            当前 _config = {"a": 1, "b": {"x": 10, "y": 20}}
            set_config({"b": {"y": 999, "z": 30}})
            → _config = {"a": 1, "b": {"x": 10, "y": 999, "z": 30}}
        """
        if not _config_loaded:
            self.load_config()
        self._config = self._deep_merge(self._config, conf)

    def reload_config(self) -> Dict[str, Any]:
        """重新加载"""
        return self.load_config(force_reload=True)

    def get_config_path(self) -> Path:
        """获取配置文件目录"""
        return self._config_dir

# ========== 便捷函数 ==========

_loader: Optional[ConfigLoader] = None


def _get_loader() -> ConfigLoader:
    global _loader
    if _loader is None:
        _loader = ConfigLoader()
    return _loader


def init_config(config_dir: Path) -> Dict[str, Any]:
    """
    初始化配置（应用启动时调用）

    Args:
        config_dir: 配置文件目录（如 project_root / "conf"）

    总是 force_reload——保证 _config 与 _config_dir 一致，避免切目录后
    get_config() 仍返回旧 dict 的静默不一致。
    """
    _get_loader().configure(config_dir)
    return _get_loader().load_config(force_reload=True)


def get_config() -> Dict[str, Any]:
    """获取配置（深拷贝——可任意修改返回值不影响内部状态）"""
    return _get_loader().get_config()


def set_config(conf: Dict[str, Any]) -> None:
    """浅覆盖设置配置（deep merge 语义）"""
    _get_loader().set_config(conf)


def reload_config() -> Dict[str, Any]:
    """重新加载配置"""
    return _get_loader().reload_config()

def get_config_path() -> Path:
    """获取配置文件目录"""
    return _get_loader().get_config_path()
