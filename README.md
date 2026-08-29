# infra

**通用基础设施类库**（自用，不公开）——DB/Redis/Config/Logger/异常/工具。

## 模块

| 模块 | 作用 |
|---|---|
| `infra.db` | 连接管理 + ORM 基类 + Repository 基类 + 监控 |
| `infra.config` | YAML 配置加载 |
| `infra.logger` | 日志系统（控制台+文件） |
| `infra.exceptions` | 异常类（DatabaseError 等）|
| `infra.time_util` | `TimeUtil` 时区转换工具 |
| `infra.data_quality` | `DataQuality` 枚举（8 状态）|
| `infra.bootstrap` | `bootstrap_all(config_dir)` 一键启动 |
| `infra.utils` | `retry` / `lark` / `wechat` / `async_helpers` / `executor` / `trend_analyzer` / `common` |
| `infra.redis` | Redis 客户端（同步/异步）|
| `infra.mq` | 消息队列（占位）|

## 安装

```bash
# 一次性（可编辑模式——改即生效）
cd ~/projects/infra
pip install -e .

# 或 uv
uv pip install -e .
```

## 使用

```python
import asyncio
from pathlib import Path
from infra import bootstrap_all, TimeUtil, DataQuality

async def main():
    # 一键启动（DB/Redis/Config/Logger）
    await bootstrap_all(config_dir=Path("/path/to/configs"))
    
    # 用时间工具
    utc_dt = TimeUtil.to_utc_event_time("2026-08-29", "Asia/Shanghai")
    
    # 用枚举
    print(DataQuality.GOOD.value)  # "good"

asyncio.run(main())
```

## 设计原则

- **零业务依赖**——`infra` 不引用任何业务包（macro_monitor 等）
- **单向上游**——业务项目可依赖 `infra`，反之不行
- **核心组件自包含**——DB/Redis/Config/Logger/TimeUtil 可独立使用
- **pytest 友好**——mock 点少（DI 模式）

## 测试

```bash
.venv/bin/python -m pytest tests/ -v
```

## 版本

0.1.0（20260829 init）
