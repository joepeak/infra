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

#### Post-processor（链式思考泄露过滤）

通过 `openrouter/free` 等动态路由时，可能命中 `cohere/north-mini-code:free` 这样直接在
`content` 输出 CoT 的非推理模型。`infra.llm.postprocess` 提供**启发式**检测 + 清洗钩子：

```python
from infra.llm import LLMRequest, init_llm_client, get_llm_client
from infra.llm.postprocess import strip_cot_after_answer, is_likely_cot

client = get_llm_client()

# 方式1：在请求时绑定后处理器 —— 自动提取 "Final answer:" 后的内容
resp = client.invoke(
    LLMRequest(
        messages=[{"role": "user", "content": "Generate a post"}],
        post_processor=strip_cot_after_answer,
    )
)

# 方式2：手动检测
resp = client.invoke(
    LLMRequest(messages=[{"role": "user", "content": "Hello"}])
)
if is_likely_cot(resp.content, resp.reasoning_content):
    # 触发降级：重试备用模型，或使用确定性模板
    ...
```

可用工具：

| 工具 | 策略 | 说明 |
|---|---|---|
| `is_likely_cot(content, reasoning_content)` | 检测 | 启发式判断 content 是否含 CoT |
| `strip_cot_after_answer(content, reasoning)` | 保守 | 仅当检测到 CoT 时，提取 "final answer:" 后内容 |
| `strip_cot_with_fallback(content, reasoning)` | 激进 | CoT 但无 final answer 标记时，回退到 reasoning_content |
| `make_cot_aware_processor(aggressive=True, min_length=400)` | 工厂 | 创建自定义后处理器 |
| `default_cot_processor` | 默认 | `strip_cot_after_answer` 的别名 |
| `aggressive_cot_processor` | 激进 | `strip_cot_with_fallback` 的别名 |

> ⚠️ 这些检测为**启发式**，不能 100% 可靠。建议与业务层验证 + 降级链搭配使用。
> `ProviderCapabilityRegistry` 会自动记录 `leaks_cot=True` 的模型，供日志/诊断查看。

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
