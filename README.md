# infra

**通用基础设施类库**（自用，不公开）——DB/Redis/Config/Logger/异常/工具。

## 模块

| 模块 | 作用 |
|---|---|
| `infra.db` | 连接管理 + ORM 基类 + Repository 基类 + 监控 |
| `infra.config` | YAML 配置加载 |
| `infra.logger` | 日志系统（控制台+文件） |
| `infra.exceptions` | 异常类（DatabaseError 等）|
| `infra.data_quality` | `DataQuality` 枚举（8 状态）|
| `infra.llm` | 通用 LLM 客户端（OpenAI 兼容协议 + fallback + streaming） |
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
from infra import bootstrap_all, to_utc_event_time, DataQuality

async def main():
    # 一键启动（DB/Redis/Config/Logger）
    await bootstrap_all(config_dir=Path("/path/to/configs"))
    
    # 用时间工具
    utc_dt = to_utc_event_time("2026-08-29", "Asia/Shanghai")
    
    # 用枚举
    print(DataQuality.GOOD.value)  # "good"

asyncio.run(main())
```

### LLM 调用

```python
import asyncio
from infra.llm import init_llm_client, get_llm_client, LLMRequest

# 方式1：从环境变量/yaml .env 初始化
config = {
    "llm": {
        "api_key": "sk-...",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        # 可选 fallback（独立 endpoint/key）
        "fallback_model": "gpt-3.5-turbo",
        "fallback_base_url": "https://api.openai.com/v1",
    }
}
await init_llm_client(config)

client = get_llm_client()
if client is None:
    raise RuntimeError("LLM 未配置")

# 同步调用
resp = client.invoke(
    LLMRequest(messages=[{"role": "user", "content": "Hello"}])
)
print(resp.content)

# 异步调用
resp = await client.ainvoke(
    LLMRequest(messages=[{"role": "user", "content": "Hello"}])
)

# 流式调用（同步）
for chunk in client.stream(
    LLMRequest(messages=[{"role": "user", "content": "Hello"}], max_tokens=100)
):
    print(chunk.content, end="", flush=True)

# 流式调用（异步）
async for chunk in client.astream(
    LLMRequest(messages=[{"role": "user", "content": "Hello"}], max_tokens=100)
):
    print(chunk.content, end="", flush=True)
```

支持 OpenAI / DeepSeek / OpenRouter 等任何 OpenAI 兼容 API。

## 设计原则

- **零业务依赖**——`infra` 不引用任何业务包（macro_monitor 等）
- **单向上游**——业务项目可依赖 `infra`，反之不行
- **核心组件自包含**——DB/Redis/Config/Logger 可独立使用
- **pytest 友好**——mock 点少（DI 模式）

## 测试

```bash
.venv/bin/python -m pytest tests/ -v
```

## 版本

0.1.0（20260829 init）
