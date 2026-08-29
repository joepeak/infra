#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infra.config 加载器测试

覆盖：
1. ConfigLoader 单例性
2. init_config + get_config 行为
3. configure 之前调用 load_config → RuntimeError
4. YAML 文件加载顺序：config.yaml → jobs.yaml → {env}.yaml → local.yaml
5. _deep_merge 的递归行为（覆盖、非 dict 不递归）
6. 环境变量优先级：MACRO_MONITOR_ENV > ENVIRONMENT > dev
7. reload_config 强制重读文件
8. 不存在的文件被静默跳过
9. YAML 解析失败时回退到 {}
10. get_config_path 返回配置的目录

修复后行为（init_config 总是 force_reload）：
   * init_config(A) → init_config(B) 后，get_config_path() == B，get_config() == B 的内容
   * reload_config() 仍然存在，用于不切换目录但想重读文件的场景

注意：ConfigLoader 是模块级单例 + 模块全局 _config/_config_loaded 状态。
每个测试**必须**通过 reset_config 还原，否则测试间会污染。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from infra.config import (
    ConfigLoader,
    get_config,
    get_config_path,
    init_config,
    reload_config,
)
import infra.config.loader as loader_module


# ============================================================
# Fixture：每个测试前重置 ConfigLoader 状态
# ============================================================
@pytest.fixture(autouse=True)
def _reset_config_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    每个测试前：
    1. 备份环境变量（涉及 MACRO_MONITOR_ENV / ENVIRONMENT）
    2. 重置 loader 模块的 _config / _config_loaded 状态
    3. 把 ConfigLoader._instance 替换为新实例（破坏单例性，让 configure() 重新生效）
    4. 还原工作目录到 tmp_path
    """
    # 备份环境变量
    saved_env = {}
    for k in ("MACRO_MONITOR_ENV", "ENVIRONMENT"):
        saved_env[k] = os.environ.pop(k, None)

    # 重置模块全局
    loader_module._config = None
    loader_module._config_loaded = False
    loader_module._loader = None
    # 破坏单例（下次 _get_loader() 会建新实例，避免 self._config 跨测试残留）
    ConfigLoader._instance = None

    yield

    # TEARDOWN：还原
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    loader_module._config = None
    loader_module._config_loaded = False
    loader_module._loader = None
    ConfigLoader._instance = None


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """返回 tmp_path 作为配置目录（测试自己写 yaml）。"""
    return tmp_path


def _write(dir_: Path, name: str, data: Dict[str, Any]) -> Path:
    """在 dir_ 下写一个 YAML 文件，返回路径。"""
    p = dir_ / name
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


# ============================================================
# 1. ConfigLoader 单例
# ============================================================
class TestConfigLoaderSingleton:
    def test_singleton_returns_same_instance(self):
        a = ConfigLoader()
        b = ConfigLoader()
        assert a is b

    def test_singleton_survives_reload(self):
        a = ConfigLoader()
        # 多次创建
        ConfigLoader()
        ConfigLoader()
        assert ConfigLoader() is a


# ============================================================
# 2. init_config + get_config 流程
# ============================================================
class TestInitAndGet:
    def test_init_config_loads_and_returns(self, config_dir):
        _write(config_dir, "config.yaml", {"a": 1, "b": "x"})

        result = init_config(config_dir)

        assert result == {"a": 1, "b": "x"}
        # get_config 应返回同一份
        assert get_config() == result

    def test_init_config_path_kept(self, config_dir):
        _write(config_dir, "config.yaml", {"a": 1})

        init_config(config_dir)

        assert get_config_path() == config_dir

    def test_get_config_path_before_init_returns_none(self):
        """configure() 之前调用 get_config_path() 返 None。"""
        assert get_config_path() is None

    def test_init_config_with_empty_dir(self, config_dir):
        """空目录：没有 yaml 文件时返回空 dict。"""
        result = init_config(config_dir)
        assert result == {}

    def test_get_config_lazy_loads(self, config_dir):
        """configure() 后未显式 init_config 时，get_config() 会触发 load。"""
        from infra.config.loader import _get_loader
        _get_loader().configure(config_dir)
        _write(config_dir, "config.yaml", {"lazy": True})

        result = get_config()
        assert result == {"lazy": True}


# ============================================================
# 3. configure() 之前调用 load_config
# ============================================================
class TestLoadBeforeConfigure:
    def test_load_without_configure_raises(self):
        """未 configure() 直接 load_config → RuntimeError。"""
        loader = ConfigLoader()
        # singleton 的 _config_dir 是 None
        with pytest.raises(RuntimeError, match="请先调用 configure"):
            loader.load_config()


# ============================================================
# 4. 文件加载顺序
# ============================================================
class TestLoadOrder:
    def test_jobs_yaml_overrides_config_yaml(self, config_dir):
        """加载顺序：config.yaml → jobs.yaml → {env}.yaml → local.yaml。

        即 jobs.yaml 后于 config.yaml 加载 → jobs.yaml 覆盖 config.yaml 同名 key。
        """
        _write(config_dir, "jobs.yaml", {"k": "from_jobs", "jobs_only": 1})
        _write(config_dir, "config.yaml", {"k": "from_config", "config_only": 2})

        result = init_config(config_dir)
        assert result["k"] == "from_jobs"  # 后加载的胜出
        assert result["jobs_only"] == 1
        assert result["config_only"] == 2

    def test_env_yaml_overrides_default(self, config_dir, monkeypatch):
        """{env}.yaml 覆盖 config.yaml 同名 key。"""
        _write(config_dir, "config.yaml", {"k": "default"})
        _write(config_dir, "test.yaml", {"k": "test_env"})

        monkeypatch.setenv("MACRO_MONITOR_ENV", "test")
        result = init_config(config_dir)
        assert result["k"] == "test_env"

    def test_local_overrides_env(self, config_dir, monkeypatch):
        """local.yaml 最后加载，覆盖所有。"""
        _write(config_dir, "config.yaml", {"k": "default"})
        _write(config_dir, "prod.yaml", {"k": "prod"})
        _write(config_dir, "local.yaml", {"k": "local"})

        monkeypatch.setenv("MACRO_MONITOR_ENV", "prod")
        result = init_config(config_dir)
        assert result["k"] == "local"

    def test_load_order_jobs_config_env_local(self, config_dir, monkeypatch):
        """完整顺序：jobs → config → env → local，全部 merge。"""
        _write(config_dir, "jobs.yaml", {"only_jobs": 1, "shared": "jobs"})
        _write(config_dir, "config.yaml", {"only_config": 2, "shared": "config"})
        _write(config_dir, "dev.yaml", {"only_env": 3, "shared": "env"})
        _write(config_dir, "local.yaml", {"only_local": 4, "shared": "local"})

        monkeypatch.setenv("MACRO_MONITOR_ENV", "dev")
        result = init_config(config_dir)
        # 全部 key 存在
        assert result == {
            "only_jobs": 1,
            "only_config": 2,
            "only_env": 3,
            "only_local": 4,
            "shared": "local",  # local 胜出
        }


# ============================================================
# 5. _deep_merge 递归
# ============================================================
class TestDeepMerge:
    def test_dict_keys_recursively_merged(self):
        loader = ConfigLoader()
        base = {"db": {"host": "localhost", "port": 5432, "user": "postgres"}}
        override = {"db": {"port": 3306, "password": "secret"}}
        result = loader._deep_merge(base, override)
        assert result == {
            "db": {
                "host": "localhost",  # 保留
                "port": 3306,  # 覆盖
                "user": "postgres",  # 保留
                "password": "secret",  # 新增
            }
        }

    def test_non_dict_values_replaced_entirely(self):
        """非 dict 值（包括 list、str、int）整体替换，不递归。"""
        loader = ConfigLoader()
        base = {"items": [1, 2, 3], "name": "old"}
        override = {"items": [9], "name": "new"}
        result = loader._deep_merge(base, override)
        assert result == {"items": [9], "name": "new"}

    def test_top_level_non_dict_added(self):
        loader = ConfigLoader()
        base = {"a": 1}
        override = {"b": 2}
        result = loader._deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        loader = ConfigLoader()
        base = {"db": {"host": "x"}}
        override = {"db": {"port": 1}}
        loader._deep_merge(base, override)
        # base 不应被修改
        assert base == {"db": {"host": "x"}}


# ============================================================
# 6. 环境变量优先级
# ============================================================
class TestEnvPriority:
    def test_macro_monitor_env_wins(self, config_dir, monkeypatch):
        _write(config_dir, "config.yaml", {"k": "base"})
        _write(config_dir, "prod.yaml", {"k": "prod"})
        _write(config_dir, "dev.yaml", {"k": "dev"})

        monkeypatch.setenv("MACRO_MONITOR_ENV", "prod")
        monkeypatch.setenv("ENVIRONMENT", "dev")
        result = init_config(config_dir)
        assert result["k"] == "prod"

    def test_environment_used_when_macro_monitor_unset(self, config_dir, monkeypatch):
        _write(config_dir, "config.yaml", {"k": "base"})
        _write(config_dir, "staging.yaml", {"k": "staging"})

        monkeypatch.delenv("MACRO_MONITOR_ENV", raising=False)
        monkeypatch.setenv("ENVIRONMENT", "staging")
        result = init_config(config_dir)
        assert result["k"] == "staging"

    def test_default_dev_when_no_env_vars(self, config_dir, monkeypatch):
        _write(config_dir, "config.yaml", {"k": "base"})
        _write(config_dir, "dev.yaml", {"k": "dev"})

        monkeypatch.delenv("MACRO_MONITOR_ENV", raising=False)
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        result = init_config(config_dir)
        assert result["k"] == "dev"

    def test_env_case_insensitive(self, config_dir, monkeypatch):
        """环境名被 .lower() 处理。"""
        _write(config_dir, "config.yaml", {"k": "base"})
        _write(config_dir, "prod.yaml", {"k": "prod"})

        monkeypatch.setenv("MACRO_MONITOR_ENV", "PROD")  # 大写
        result = init_config(config_dir)
        assert result["k"] == "prod"


# ============================================================
# 7. reload_config
# ============================================================
class TestReload:
    def test_reload_picks_up_file_changes(self, config_dir):
        cfg = _write(config_dir, "config.yaml", {"k": 1})
        init_config(config_dir)
        assert get_config() == {"k": 1}

        # 改文件
        cfg.write_text(yaml.safe_dump({"k": 2}), encoding="utf-8")
        # 没 reload 之前还是旧值
        assert get_config() == {"k": 1}
        # reload 后新值
        result = reload_config()
        assert result == {"k": 2}

    def test_get_config_after_reload_returns_new(self, config_dir):
        _write(config_dir, "config.yaml", {"k": 1})
        init_config(config_dir)
        reload_config()
        assert get_config() == {"k": 1}


# ============================================================
# 8. 不存在 / 损坏的文件
# ============================================================
class TestFileErrors:
    def test_missing_files_silently_skipped(self, config_dir):
        """空目录（无任何 yaml）→ 返空 dict，不抛异常。"""
        result = init_config(config_dir)
        assert result == {}

    def test_missing_env_yaml_ok(self, config_dir, monkeypatch):
        """环境 yaml 不存在：跳过，不影响其他文件。"""
        _write(config_dir, "config.yaml", {"k": 1})
        monkeypatch.setenv("MACRO_MONITOR_ENV", "nonexistent_env")
        result = init_config(config_dir)
        assert result == {"k": 1}

    def test_malformed_yaml_returns_empty_for_that_file(self, config_dir, monkeypatch):
        """单个 yaml 损坏 → 加载该文件返空 {}（不影响其他）。"""
        _write(config_dir, "config.yaml", {"k": 1})
        # jobs.yaml 是损坏的 yaml
        (config_dir / "jobs.yaml").write_text("not: valid: yaml: : :", encoding="utf-8")

        # 应该不抛：损坏文件被 logger.error + 返回 {}，其他文件正常
        result = init_config(config_dir)
        # config.yaml 正常加载，jobs.yaml 加载失败回退到 {}
        assert result.get("k") == 1


# ============================================================
# 9. 多 init 幂等
# ============================================================
class TestReconfigure:
    def test_re_init_with_new_dir_replaces_config(self, config_dir, tmp_path):
        """re-init 总是 force_reload：切到新目录后 get_config() 立即是新值。

        修复前：init_config 切目录后 get_config() 仍返旧 dict（懒缓存），
        需显式 reload_config() 才会读到新目录。
        修复后：init_config(B) 自动 force_reload，get_config() 立即是 B 的内容。
        """
        _write(config_dir, "config.yaml", {"a": 1})
        init_config(config_dir)
        assert get_config() == {"a": 1}

        _write(tmp_path, "config.yaml", {"b": 2})
        init_config(tmp_path)
        # 路径和内容都同步更新
        assert get_config_path() == tmp_path
        assert get_config() == {"b": 2}

    def test_re_init_same_dir_also_reloads(self, config_dir, tmp_path):
        """同目录重复 init_config 也会 force_reload——便于运行时改文件后重 init。"""
        cfg_path = config_dir / "config.yaml"
        _write(config_dir, "config.yaml", {"v": 1})
        init_config(config_dir)
        assert get_config() == {"v": 1}

        # 改文件
        cfg_path.write_text("v: 2", encoding="utf-8")
        # 不 reload，缓存仍是 1
        assert get_config() == {"v": 1}
        # 再次 init_config 同一目录 → force_reload → 新值
        init_config(config_dir)
        assert get_config() == {"v": 2}


# ============================================================
# 运行入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
