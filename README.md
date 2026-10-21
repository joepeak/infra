# infra

**通用基础设施类库**（自用，不公开）——DB/Redis/MQ/Config/Logger/LLM/调度/异常/工具。

## 模块

| 模块 | 作用 |
|---|---|
| `infra.db` | 连接管理（多 key Map 化）+ ORM 基类 + Repository 基类 + 监控 |
| `infra.redis` | Redis 客户端（同步/异步）——String/Hash/List/Set/ZSet/Stream 全命令 + 分布式锁 |
| `infra.mq` | **消息队列**（基于 Redis Stream）——重试 / 死信 / 优先级 / 并发限流 |
| `infra.schedule` | 分布式任务调度（APScheduler + Redis leader 选举）|
| `infra.llm` | 通用 LLM 客户端（OpenAI 兼容协议 + fallback + streaming + 能力探测）|
| `infra.config` | YAML 配置加载（deep merge / 环境变量占位符解析）|
| `infra.logger` | 日志系统（控制台+文件） |
| `infra.script_logger` | 独立脚本日志（控制台+文件） |
| `infra.exceptions` | 异常类（DatabaseError 等）|
| `infra.messages` | 通用消息定义（`TaskMessage` dataclass）|
| `infra.data_quality` | `DataQuality` 枚举（8 状态）|
| `infra.bootstrap` | `bootstrap_all(config_dir)` 一键启动 |
| `infra.utils` | `retry` / `lark` / `wechat` / `async_helpers` / `executor` / `error_handler` / `common` / `time_util` |

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

### 消息队列（infra.mq）

基于 **Redis Stream**——不是 SAQ/内存队列。核心能力：

| 能力 | 说明 |
|---|---|
| **失败重试** | handler 抛异常 → 消息留在 PEL → 认领协程按 `claim_min_idle_ms` 认领后重试 |
| **死信队列** | 重试次数超 `max_retries` → 移入 `<stream>:dead_letter`，可查可恢复 |
| **优先级消费** | 一个逻辑队列 = 多个 stream（`q:critical` / `q:high` / ...），按优先级顺序读 |
| **每优先级独立配置** | 各自的并发上限 / 重试次数 / 时效 / 死信队列 |
| **并发限流** | `max_concurrency` 真正限制同时执行的 handler 数（≠ worker 数 × batch_size）|
| **幂等支持** | handler 拿到 `msg.msg_id`——投递语义是「至少一次」，用它做幂等键 |
| **消费者清理** | `stop()` 清理消费者组成员——不残留僵尸 consumer |

```python
from infra.mq import init_mq, get_mq, MessagePriority

mq = await init_mq(
    config={"redis": {"host": "127.0.0.1", "port": 6379}},
    routing={"image_gen": "media_queue"},
    queues=[{
        "name": "media_queue",
        "concurrency": 3,
        "priorities": ["critical", "high", "normal", "low"],
        "per_priority": {
            # critical 独占 10 个并发额度，重试 5 次，独立死信队列
            "critical": {
                "max_concurrency": 10,
                "max_retries": 5,
                "separate_dead_letter": True,
            },
        },
    }],
    handlers={"image_gen": my_handler},
    default_queue="media_queue",
)

# 发布（priority 决定进哪个 stream）
await mq.publish(
    "media_queue",
    {"task_id": "t1", "task_type": "image_gen", "priority": "critical", "payload": {...}},
)

# 死信管理
dead = await mq.get_dead_letters("media_queue", priority="critical")
await mq.retry_dead_letter("media_queue", dead[0]["dead_letter_id"], priority="critical")
```

> **优先级是真优先级**：Redis 的 `XREADGROUP` 按传入的 streams 参数顺序分组返回，
> 所以消费顺序由 Redis 原生保证——不是应用层排序（单 stream 排序做不到真优先级）。

### 分布式调度（infra.schedule）

APScheduler + Redis leader 选举——**多实例下只有一个跑定时任务**。

```python
from infra.schedule import LeaderElection, SchedulerManager, get_scheduler_manager

scheduler = get_scheduler_manager()
leader = LeaderElection(
    {"host": "127.0.0.1", "port": 6379},
    # 失去 leader 时停掉调度器——否则会出现「脑裂」：
    # 锁已被别人抢走，本实例却还在跑定时任务
    on_lost_leader=scheduler.on_lost_leader,
)

if await leader.try_acquire():
    scheduler.init_scheduler(is_leader=True)
    scheduler.start()
```

> **务必挂上 `on_lost_leader`**：`LeaderElection` 与 `SchedulerManager` 是两个独立对象，
> 不联动就会在 leader 切换时出现定时任务双执行。

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

# 批量并发（保持输入顺序）
results = await client.abatch([req1, req2, req3], concurrency=5)

# 流式但要完整响应（一站式）
full = await client.astream_collect(req)
```

支持 OpenAI / DeepSeek / OpenRouter 等任何 OpenAI 兼容 API。

#### 能力自适应

首次调用前自动探测端点能力（thinking 参数是否被拒收、json_schema 是否支持等），
结论写入 `ProviderCapabilityRegistry`——后续调用自适应调整参数。

```python
from infra.llm import get_registry

caps = get_registry().get("https://api.deepseek.com/v1", "deepseek-chat")
print(caps.thinking_param_rejected)  # True/False/None(未探明)
```

#### 成本统计

文本按 **token** 计，媒体按 **次数/张数/时长** 计——两种维度分开聚合，不混算。

```python
from infra.llm import get_usage_summary, get_media_usage_summary

get_usage_summary()     # 按用途汇总 token
get_media_usage_summary()  # 按模态汇总张数/秒数/金额（金额按币种分桶）
```

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
- **核心组件自包含**——DB/Redis/Config/Logger/MQ/Schedule 可独立使用
- **pytest 友好**——mock 点少（DI 模式）
- **mypy 全项目 0 error**——类型标注不欠债

## 测试

```bash
# 默认（跳过 slow 与需要真 Redis 的集成测试）
.venv/bin/pytest tests/ -q

# 全部（含 slow）
.venv/bin/pytest tests/ -m "" -q

# 真 Redis 集成测试（需本地起 redis）
docker run -d --rm --name infra-redis-verify -p 16379:6379 redis/redis-stack:7.4.0-v1
REDIS_INTEGRATION_HOST=127.0.0.1 REDIS_INTEGRATION_PORT=16379 \
  .venv/bin/pytest tests/ -m integration -q
```

集成测试覆盖 `infra.mq`（消息队列全链路）与 `infra.schedule`（leader 选举）——
这两块的 mock 测不出真问题：字段名读错、返回值顺序解包错、锁被抢后不停调度器。

类型检查：

```bash
.venv/bin/mypy src/infra/
```

## 版本

0.1.0（20260829 init）
