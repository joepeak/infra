#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 Redis Stream 的消息队列
准生产级别：支持多队列、并发消费、PEL 管理、死信队列、时效性检查
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Awaitable, List, Tuple, Union
from dataclasses import dataclass

import redis.asyncio as redis
from redis.exceptions import ResponseError
from infra.messages import TaskMessage

from infra.logger import get_logger

logger = get_logger(__name__)


class MessagePriority:
    """消息优先级枚举"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

@dataclass
class QueueConfig:
    """单个队列配置"""
    stream_name: str
    group_name: str = ""
    maxlen: int = 10000
    block_ms: int = 1000
    batch_size: int = 10
    max_retries: int = 3
    concurrency: int = 5
    claim_interval: int = 30
    claim_min_idle_ms: int = 60000
    ttl_seconds: int = 0  # 0 表示永不过期


class RedisStreamMQ:
    """
    基于 Redis Stream 的消息队列
    支持多队列、并发消费、PEL 管理、死信队列、时效性检查
    """
    
    def __init__(self, redis_config: Dict[str, Any]):
        self.redis_config = redis_config
        self._redis: Optional[redis.Redis] = None
        self._queues: Dict[str, QueueConfig] = {}
        self._handlers: Dict[str, Dict[str, Callable]] = {}
        self._routing: Dict[str, str] = {}
        self._consumer_tasks: Dict[str, List[asyncio.Task]] = {}
        self._claim_tasks: Dict[str, asyncio.Task] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._default_queue: Optional[str] = None
        # 动态属性：init_mq 注入——给静态类型一个声明
        self.config: Dict[str, Any] = {}
        # narrow 时——重连失败时存具体 Exception
        self._connection_error: Optional[Exception] = None
        self._stats: Dict[str, Any] = {
            'total_processed': 0,
            'total_failed': 0,
            'total_expired': 0,
            'last_error': None
        }
        self._reconnect_count = 0
        self._max_reconnect = 10
        self._connection_error = None

    def _get_redis_url(self) -> str:
        """获取 Redis 连接 URL"""
        host = self.redis_config.get('host', 'localhost')
        port = self.redis_config.get('port', 6379)
        db = self.redis_config.get('db', 0)
        password = self.redis_config.get('password')

        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

    def _r(self) -> "redis.Redis":
        """narrow self._redis: Optional[Redis] → Redis；未连时抛 DatabaseError。

        业务方法第一行调 `redis = self._r()`，之后 redis 不再是 Optional——
        mypy 不再 union-attr 报错；运行时若未 connect 也立刻抛错而不是 NoneType。
        """
        from infra.exceptions import DatabaseError
        if self._redis is None:
            raise DatabaseError("Redis 客户端未初始化", operation="mq")
        return self._redis

    async def connect(self) -> bool:
        """
        连接 Redis，返回是否连接成功
        """
        if self._redis is not None:
            try:
                await self._redis.ping()
                return True
            except Exception as e:
                logger.warning(f"Redis 连接无效: {e}, 准备重连")
                self._redis = None
        
        # 重连
        for attempt in range(self._max_reconnect):
            try:
                self._redis = await redis.from_url(
                    self._get_redis_url(),
                    decode_responses=True,
                    max_connections=40,
                    socket_keepalive=True,
                    socket_timeout=60,
                    socket_connect_timeout=10
                )
                await self._redis.ping()
                self._reconnect_count = 0
                self._connection_error = None
                logger.info("Redis 连接成功")
                return True
            except Exception as e:
                self._reconnect_count += 1
                self._connection_error = e
                wait = min(2 ** attempt, 10)
                logger.warning(f"Redis 重连失败 (尝试 {attempt + 1}/{self._max_reconnect}): {e}, 等待 {wait}s")
                await asyncio.sleep(wait)
        
        # 重连失败
        logger.error(f"Redis 连接失败，已达最大重试次数: {self._connection_error}")
        return False
    
    def create_queue(self, stream_name: str, group_name: Optional[str] = None, **kwargs: Any) -> 'RedisStreamMQ':
        """创建/注册一个队列"""
        if group_name is None:
            group_name = f"{stream_name}_group"
        
        config = QueueConfig(
            stream_name=stream_name,
            group_name=group_name,
            maxlen=kwargs.get('maxlen', 10000),
            block_ms=kwargs.get('block_ms', 1000),
            batch_size=kwargs.get('batch_size', 10),
            max_retries=kwargs.get('max_retries', 3),
            concurrency=kwargs.get('concurrency', 5),
            claim_interval=kwargs.get('claim_interval', 30),
            claim_min_idle_ms=kwargs.get('claim_min_idle_ms', 60000),
            ttl_seconds=kwargs.get('ttl_seconds', 0)
        )
        
        self._queues[stream_name] = config
        self._handlers[stream_name] = {}
        
        logger.info(f"队列已注册: {stream_name} (concurrency={config.concurrency}, ttl={config.ttl_seconds}s)")
        return self
    
    def set_routing(self, routing: Dict[str, str]) -> None:
        """设置消息路由规则"""
        self._routing.update(routing)
        logger.info(f"路由规则已更新: {list(self._routing.keys())}")
    
    def register(self, stream_name: str, msg_type: str, handler: Callable[[Dict[str, str]], Awaitable[Any]]) -> None:
        """注册消息处理器"""
        if stream_name not in self._handlers:
            raise ValueError(f"队列不存在: {stream_name}，请先调用 create_queue")
        
        self._handlers[stream_name][msg_type] = handler
        logger.info(f"注册处理器: {stream_name}/{msg_type}")
    
    async def publish(self, stream_name: str, message: Union[TaskMessage, Dict[str, Any]]) -> Optional[str]:
        """
        发布消息到指定队列
        支持 TaskMessage 对象或普通字典
        """
        if stream_name not in self._queues:
            raise ValueError(f"队列不存在: {stream_name}")
        
        # 尝试连接
        if not await self.connect():
            logger.error(f"Redis 未连接，消息发送失败: {stream_name}")
            return None
        
        config = self._queues[stream_name]

        # 处理 TaskMessage 对象
        if isinstance(message, TaskMessage):
            data = message.to_mq()
        else:
            # 兼容旧的字典格式
            #
            # 序列化单独包 try：payload 不可 JSON 序列化是**调用方的 bug**
            # （传了非 JSON 类型），不是运行时故障——必须抛出来暴露问题，
            # 不能伪装成「发送失败返回 None」，否则消息静默丢失。
            try:
                payload_json = json.dumps(
                    message.get('payload', {}), ensure_ascii=False
                )
                metadata_json = (
                    json.dumps(message.get('metadata', {}), ensure_ascii=False)
                    if message.get('metadata') else ''
                )
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"消息 payload/metadata 无法 JSON 序列化: {e}"
                ) from e
            data = {
                'task_id': message.get('task_id', str(uuid.uuid4())),
                'task_type': message.get('task_type', 'unknown'),
                'scheduled_time': message.get('scheduled_time', datetime.now().isoformat()),
                'payload': payload_json,
                'metadata': metadata_json,
                'priority': message.get('priority', MessagePriority.NORMAL),
                'created_at': str(time.time())
            }
        
        # narrow self._redis: Optional[Redis] → Redis（避免每行都重写 None 检查）
        redis = self._redis
        if redis is None:
            logger.error(f"Redis 未连接，消息发送失败: {stream_name}")
            return None
        try:
            msg_id = await redis.xadd(
                config.stream_name,
                data,  # type: ignore[arg-type]
                maxlen=config.maxlen
            )
            logger.debug(f"消息已发布: {config.stream_name}/{msg_id!r}")
            return str(msg_id) if msg_id is not None else None
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
            return None
    
    async def publish_by_routing(self, message: Union[TaskMessage, Dict[str, Any]]) -> Optional[str]:
        """根据路由规则自动发布消息"""
        # 获取 task_type——narrow Any | None → str
        # 真业务 bug：缺 task_type 时下面会抛 ValueError——已处理
        if isinstance(message, TaskMessage):
            task_type = message.task_type
        else:
            task_type = message.get('task_type', "")

        if not task_type:
            raise ValueError("消息缺少 task_type 字段")

        stream_name = self._routing.get(task_type, self._default_queue)
        if not stream_name:
            raise ValueError(f"未找到 {task_type} 的路由规则，且未设置默认队列")
        
        return await self.publish(stream_name, message)
    
    async def publish_simple(self, message: Union[TaskMessage, Dict[str, Any]]) -> Optional[str]:
        """发布消息到默认队列"""
        if not self._default_queue:
            raise ValueError("未设置默认队列")
        return await self.publish(self._default_queue, message)
    
    def set_default_queue(self, stream_name: str) -> None:
        """设置默认队列"""
        if stream_name not in self._queues:
            raise ValueError(f"队列不存在: {stream_name}")
        self._default_queue = stream_name
        logger.info(f"默认队列已设置: {stream_name}")
    
    # def send_message_sync(self, stream_name: str, message: Dict[str, Any]) -> bool:
    #     """
    #     同步发送消息到指定队列（供 APScheduler 等同步环境使用）
    #     """
    #     try:
    #         loop = asyncio.new_event_loop()
    #         asyncio.set_event_loop(loop)
    #         try:
    #             result = loop.run_until_complete(self.publish(stream_name, message))
    #             return result is not None
    #         finally:
    #             loop.close()
    #     except Exception as e:
    #         logger.error(f"同步发送消息失败: {e}")
    #         return False
    
    # def send_message_sync_by_routing(self, message: Dict[str, Any]) -> bool:
    #     """
    #     同步发送消息（根据路由规则）
    #     """
    #     task_type = message.get('task_type')
    #     if not task_type:
    #         logger.error("消息缺少 task_type 字段")
    #         return False
        
    #     stream_name = self._routing.get(task_type, self._default_queue)
    #     if not stream_name:
    #         logger.error(f"未找到 {task_type} 的路由规则")
    #         return False
        
    #     return self.send_message_sync(stream_name, message)
    
    async def _ensure_consumer_group(self, config: QueueConfig) -> bool:
        """确保消费者组存在"""
        redis = self._r()
        try:
            await redis.xgroup_create(
                config.stream_name,
                config.group_name,
                id='0',
                mkstream=True
            )
            logger.debug(f"创建消费者组: {config.group_name}")
            return True
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"消费者组已存在: {config.group_name}")
                return True
            else:
                logger.warning(f"创建消费者组失败: {e}")
                return False
        except Exception as e:
            logger.warning(f"创建消费者组异常: {e}")
            return False
    
    def _is_message_expired(self, created_at: str, ttl_seconds: int) -> bool:
        """检查消息是否过期"""
        if ttl_seconds <= 0:
            return False
        
        try:
            created_time = float(created_at)
            return (time.time() - created_time) > ttl_seconds
        except (ValueError, TypeError):
            return False
    
    async def _claim_pending_loop(self, stream_name: str, config: QueueConfig) -> None:
        """独立协程：定期扫描并认领超时的待处理消息"""
        consumer_name = f"claimer_{uuid.uuid4().hex[:8]}"
        logger.info(f"🔁 认领协程启动: {stream_name}/{consumer_name}")

        # XAUTOCLAIM 的游标——续扫用，避免每次都从 0-0 开始
        cursor = "0-0"

        while self._running:
            try:
                if self._redis is None:
                    await asyncio.sleep(1)
                    continue

                redis = self._redis  # narrow

                # 用 XAUTOCLAIM 批量认领超时消息。
                #
                # 为什么不用 xpending_range + xclaim 逐条处理：
                #   1. XAUTOCLAIM 一次最多可认领 count 条，RTT 更少；
                #   2. XAUTOCLAIM **自动累加** delivery counter
                #      （xclaim 也会累加，但需要逐条调用）；
                #   3. XAUTOCLAIM 会返回「已从 stream 中删除」的消息 id
                #      （被 MAXLEN 裁剪的幽灵消息）——这类消息内容已丢失，
                #      无法重试，只能记日志避免它们永远占着 PEL。
                # XAUTOCLAIM 返回 [next_cursor, messages, deleted]
                # 注意顺序——cursor 在**第一个**，不是最后一个。
                next_cursor, claimed, deleted = await redis.xautoclaim(
                    config.stream_name,
                    config.group_name,
                    consumer_name,
                    config.claim_min_idle_ms,
                    cursor,
                    count=config.batch_size,
                )

                for msg_id in deleted or ():
                    # 消息内容已被 MAXLEN 裁剪——无法重试也无法进死信
                    # （死信需要原始数据）。只记日志并从 PEL 移除。
                    logger.warning(
                        f"消息内容已被 MAXLEN 裁剪，无法恢复，从 PEL 移除: "
                        f"{stream_name}/{msg_id!r}"
                    )
                    try:
                        await redis.xack(
                            config.stream_name, config.group_name, msg_id
                        )
                    except Exception as e:
                        logger.warning(f"清理裁剪消息失败: {e}")

                for msg_id, data in claimed or ():
                    # 认领后**立即处理**——否则消息归 claimer 所有，
                    # 而 worker 用 '>' 只读新消息，永远读不到它，
                    # 消息会在 claimer 之间空转、永远不消费。
                    await self._handle_claimed_message(
                        stream_name, config, str(msg_id), data
                    )

                # 更新游标——下一轮从上次的位置继续扫。
                # 扫完一圈（cursor 回到 0-0）说明没有更多了，保持 0-0 即可。
                cursor = next_cursor if next_cursor else "0-0"

                await asyncio.sleep(config.claim_interval)

            except asyncio.CancelledError:
                logger.info(f"认领协程被取消: {stream_name}")
                break
            except Exception as e:
                if "Event loop is closed" not in str(e):
                    logger.error(f"认领协程异常: {stream_name}, error={e}")
                await asyncio.sleep(config.claim_interval)

        logger.info(f"👋 认领协程已停止: {stream_name}")

    async def _handle_claimed_message(
        self,
        stream_name: str,
        config: QueueConfig,
        msg_id: str,
        data: Dict[str, str],
    ) -> None:
        """处理一条**已认领**的消息。

        认领只是把消息的所有权转给 claimer——还必须真正执行 handler，
        否则消息永远不会被消费。

        处理成功 → XACK；失败 → 不 ACK，留在 PEL 等待下一轮认领
        （重试次数超限则移入死信队列）。
        """
        try:
            # 认领前的投递次数——用于判断是否已超 max_retries。
            # XAUTOCLAIM 会自动累加 delivery counter，所以认领后
            # 次数 = 认领前 + 1。
            delivery_count = await self._get_delivery_count(
                stream_name, config, msg_id
            )
            if delivery_count > config.max_retries:
                await self._move_to_dead_letter_by_id(
                    stream_name, config, msg_id, delivery_count
                )
                return

            success, error_msg = await self._process_message(
                stream_name, config, msg_id, data
            )
            if success:
                logger.info(f"已认领消息处理完成: {stream_name}/{msg_id}")
            else:
                logger.warning(
                    f"已认领消息处理失败（将重试）: {stream_name}/{msg_id}, "
                    f"error={error_msg}"
                )
        except Exception as e:
            # 处理本身抛异常（如 Redis 断连）——不 ACK，留在 PEL
            logger.error(
                f"处理已认领消息时异常: {stream_name}/{msg_id}, error={e}"
            )

    async def _get_delivery_count(
        self,
        stream_name: str,
        config: QueueConfig,
        msg_id: str,
    ) -> int:
        """查询某条消息的投递次数（times_delivered）。

        查不到时返回 0（不阻塞主流程）。

        Note:
            ``xpending_range`` 返回的字段是 ``time_since_delivered``
            （空闲毫秒数），**不是** ``idle``——读错字段名会恒拿到 0，
            导致超时判断永远不成立。
        """
        redis = self._r()
        try:
            pending = await redis.xpending_range(
                stream_name, config.group_name, msg_id, msg_id, count=1
            )
            for item in pending or ():
                if str(item.get("message_id")) == str(msg_id):
                    return int(item.get("times_delivered", 0) or 0)
        except Exception as e:
            logger.warning(f"查询投递次数失败: {e}")
        return 0

    
    async def _move_to_dead_letter_by_id(self, stream_name: str, config: QueueConfig,
                                          msg_id: str, delivery_count: int,
                                          reason: str = "max_retries_exceeded") -> None:
        """根据消息 ID 移动到死信队列。

        Args:
            stream_name: 流名
            config: 队列配置
            msg_id: 消息 ID
            delivery_count: 已投递次数
            reason: 移入死信的原因（写入死信记录，便于排查）
        """
        redis = self._r()
        dead_letter_key = f"{stream_name}:dead_letter"
        msgs = await redis.xrange(config.stream_name, msg_id, msg_id)
        if not msgs:
            # 消息内容已被 MAXLEN 裁剪——死信需要原始数据，没法搬运。
            # 但仍要把它从 PEL 移除，否则它会永远占着 PEL 变成幽灵消息。
            logger.warning(
                f"消息内容已不可用（可能被 MAXLEN 裁剪），从 PEL 移除: "
                f"{stream_name}/{msg_id}, reason={reason}"
            )
            try:
                await redis.xack(config.stream_name, config.group_name, msg_id)
            except Exception as e:
                logger.warning(f"清理裁剪消息失败: {e}")
            return

        for entry in msgs:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            data = entry[1]
            await redis.xadd(
                dead_letter_key,
                {
                    'original_id': msg_id,
                    'original_data': json.dumps(data, ensure_ascii=False),
                    'delivery_count': str(delivery_count),
                    'reason': reason,
                    'failed_at': str(time.time())
                },
                maxlen=1000
            )
            await redis.xack(config.stream_name, config.group_name, msg_id)
            logger.warning(
                f"消息已移至死信队列: {stream_name}/{msg_id}, "
                f"delivery_count={delivery_count}, reason={reason}"
            )
    
    async def _process_message(self, stream_name: str, config: QueueConfig, 
                                msg_id: str, data: Dict[str, str]) -> Tuple[bool, Optional[str]]:
        """处理单条消息"""
        redis = self._r()  # narrow
        task_type = data.get('task_type', 'unknown')
        created_at = data.get('created_at', '0')

        # 时效性检查
        if self._is_message_expired(created_at, config.ttl_seconds):
            logger.warning(f"⏰ 消息已过期，丢弃: {stream_name}/{msg_id}, type={task_type}")
            await redis.xack(config.stream_name, config.group_name, msg_id)
            self._stats['total_expired'] += 1
            return True, None

        try:
            # 自动反序列化为 TaskMessage 对象
            message = TaskMessage.from_mq(data)

            logger.info(f"📨 处理消息: {stream_name}/{msg_id}, type={task_type}")

            handlers = self._handlers.get(stream_name, {})
            handler = handlers.get(task_type)

            if handler:
                await handler(message)
                await redis.xack(config.stream_name, config.group_name, msg_id)
                self._stats['total_processed'] += 1
                logger.debug(f"✅ 消息处理完成: {msg_id}")
                return True, None
            else:
                # 没有对应的处理器——**移入死信队列**而不是静默丢弃。
                # 静默 XACK 会让消息彻底消失且无任何痕迹：既不知道丢了什么，
                # 也没法事后重放。进死信后至少可查、可 retry_dead_letter。
                logger.warning(
                    f"⚠️ 未找到处理器，移入死信队列: {stream_name}/{task_type}"
                )
                await self._move_to_dead_letter_by_id(
                    stream_name, config, msg_id, 0, reason="no_handler"
                )
                self._stats['total_failed'] += 1
                self._stats['last_error'] = f"no handler for task_type={task_type}"
                return True, None
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 消息处理失败: {msg_id}, error={error_msg}")
            self._stats['total_failed'] += 1
            self._stats['last_error'] = error_msg
            return False, error_msg
    
    async def _consume_worker(self, stream_name: str, config: QueueConfig, worker_id: int) -> None:
        """单个消费工作协程"""
        consumer_name = f"worker_{worker_id}_{uuid.uuid4().hex[:8]}"
        
        # 等待一下，避免所有 worker 同时启动
        await asyncio.sleep(worker_id * 0.1)
        
        try:
            if not await self._ensure_consumer_group(config):
                logger.error(f"无法创建消费者组，跳过 worker: {stream_name}/{consumer_name}")
                return
            
            logger.info(f"🚀 消费者启动: {stream_name}/{consumer_name}")
            
            while self._running:
                try:
                    if self._redis is None:
                        logger.warning(f"Redis 连接不存在，等待重连...")
                        await asyncio.sleep(1)
                        continue

                    redis = self._redis  # narrow
                    result = await redis.xreadgroup(
                        groupname=config.group_name,
                        consumername=consumer_name,
                        streams={config.stream_name: '>'},
                        count=config.batch_size,
                        block=config.block_ms
                    )
                    
                    if not result:
                        continue

                    # result: List[Tuple[Any, List[Tuple[Any, Any]]]]——mypy 看不到解构
                    for entry in result:
                        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                            continue
                        messages = entry[1]
                        tasks = [
                            self._process_message(stream_name, config, msg_id, data)
                            for msg_id, data in messages
                        ]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for i, res in enumerate(results):
                            if isinstance(res, Exception):
                                if "Event loop is closed" not in str(res):
                                    logger.error(f"批量处理异常: {res}")
                            
                except asyncio.CancelledError:
                    logger.info(f"消费者任务被取消: {stream_name}/{worker_id}")
                    break
                except Exception as e:
                    if "Event loop is closed" in str(e):
                        logger.warning(f"事件循环已关闭，停止消费者: {stream_name}/{worker_id}")
                        break
                    else:
                        logger.error(f"消费循环异常: {stream_name}, error={e}")
                        await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"消费者启动失败: {stream_name}/{worker_id}, error={e}")
        finally:
            logger.info(f"👋 消费者已停止: {stream_name}/{worker_id}")
    
    async def _monitor_loop(self) -> None:
        """监控协程：定期输出统计信息"""
        while self._running:
            await asyncio.sleep(60)
            if self._running:
                stats = await self.get_stats()
                logger.info(f"📊 MQ 统计: processed={self._stats['total_processed']}, "
                           f"failed={self._stats['total_failed']}, "
                           f"expired={self._stats['total_expired']}, "
                           f"queues={list(stats.keys())}")
    
    async def start(self, stream_names: Optional[List[str]] = None) -> None:
        """启动消费者"""
        # 1. 先建立 Redis 连接
        if not await self.connect():
            logger.error("Redis 连接失败，无法启动消费者")
            return
        
        # 2. 验证连接
        try:
            redis = self._r()
            await redis.ping()
            logger.info("Redis 连接验证成功")
        except Exception as e:
            logger.error(f"Redis 连接验证失败: {e}")
            return
        
        self._running = True
        
        if stream_names is None:
            stream_names = list(self._queues.keys())
        
        # 3. 先创建所有消费者组（同步操作，避免并发问题）
        for stream_name in stream_names:
            if stream_name not in self._queues:
                logger.warning(f"队列不存在，跳过: {stream_name}")
                continue
            
            config = self._queues[stream_name]
            try:
                await self._ensure_consumer_group(config)
                logger.debug(f"消费者组已就绪: {config.group_name}")
            except Exception as e:
                logger.error(f"创建消费者组失败 {stream_name}: {e}")
        
        # 4. 等待一下，让消费者组完全创建
        await asyncio.sleep(0.5)
        
        # 5. 启动认领协程和消费者工作协程
        for stream_name in stream_names:
            if stream_name not in self._queues:
                continue
            
            config = self._queues[stream_name]
            
            claim_task = asyncio.create_task(self._claim_pending_loop(stream_name, config))
            self._claim_tasks[stream_name] = claim_task
            
            worker_tasks = []
            for i in range(config.concurrency):
                task = asyncio.create_task(self._consume_worker(stream_name, config, i))
                worker_tasks.append(task)
            
            self._consumer_tasks[stream_name] = worker_tasks
            logger.info(f"队列已启动: {stream_name} (concurrency={config.concurrency})")
        
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("MQ 监控协程已启动")
    
    async def stop(self) -> None:
        """优雅停止所有消费者"""
        logger.info("正在停止消息队列...")

        self._running = False

        # 等正在处理的消息完成。
        #
        # 旧实现是固定 sleep(3)——handler 执行超过 3 秒会被强行中断
        # （消息未 ACK 会重试，行为可接受但很粗暴）。改成：
        #   1. 先等一小段时间让 worker 看到 _running=False 自然退出；
        #   2. 再 cancel 剩余任务并等待它们真正结束。
        await asyncio.sleep(0.1)

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        for stream_name, task in self._claim_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._claim_tasks.clear()

        for stream_name, tasks in self._consumer_tasks.items():
            for task in tasks:
                task.cancel()
            # 等待所有 worker 真正结束（而不是逐个 await，
            # 避免第一个 await 阻塞时其它任务还没被 cancel）
            await asyncio.gather(*tasks, return_exceptions=True)
        self._consumer_tasks.clear()

        if self._redis:
            try:
                await self._redis.aclose()
            except Exception as e:
                logger.warning(f"关闭 Redis 连接失败: {e}")
            finally:
                # 置空——让下次 connect() 走完整重连流程，
                # 而不是复用已关闭的连接对象
                self._redis = None

        logger.info("消息队列已停止")
    
    async def get_stats(self, stream_name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        await self.connect()
        
        if stream_name:
            return await self._get_queue_stats(stream_name)
        
        stats = {}
        for name in self._queues.keys():
            stats[name] = await self._get_queue_stats(name)
        stats['_global'] = {
            'total_processed': self._stats['total_processed'],
            'total_failed': self._stats['total_failed'],
            'total_expired': self._stats['total_expired'],
            'last_error': self._stats['last_error']
        }
        return stats
    
    async def _get_queue_stats(self, stream_name: str) -> Dict[str, Any]:
        """获取单个队列统计"""
        if stream_name not in self._queues:
            return {'error': f'queue not found: {stream_name}'}

        config = self._queues[stream_name]
        redis = self._r()  # narrow

        stream_len = await redis.xlen(config.stream_name)
        dead_len = await redis.xlen(f"{config.stream_name}:dead_letter")

        try:
            pending = await redis.xpending(config.stream_name, config.group_name)
            pending_count = pending.get('pending', 0) if pending else 0
        except Exception as e:
            logger.warning(f"查询 pending 失败(按0处理): {e}")
            pending_count = 0
        
        return {
            'stream_name': config.stream_name,
            'group_name': config.group_name,
            'stream_length': stream_len,
            'pending_count': pending_count,
            'dead_letter_count': dead_len,
            'registered_handlers': list(self._handlers.get(stream_name, {}).keys()),
            'worker_count': len(self._consumer_tasks.get(stream_name, [])),
            'claimer_running': stream_name in self._claim_tasks,
            'ttl_seconds': config.ttl_seconds
        }
    
    async def retry_dead_letter(self, stream_name: str, msg_id: str) -> bool:
        """从死信队列重新处理消息"""
        redis = self._r()  # narrow
        dead_letter_key = f"{stream_name}:dead_letter"
        data = await redis.xrange(dead_letter_key, msg_id, msg_id)

        if not data:
            return False

        config = self._queues[stream_name]
        for entry in data:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            msg_data = entry[1]
            # narrow bytes | str → str（json 字符串）+ fallback {}（防 None 真业务 bug）
            raw = msg_data.get('original_data', '{}') if msg_data else '{}'
            # 死信数据可能损坏（人工改过 / 旧版本写入的格式不同）——
            # 解析失败不能抛异常打断调用方，记日志并跳过该条。
            try:
                original_data = json.loads(raw)
            except (TypeError, ValueError) as e:
                logger.error(
                    f"死信消息数据解析失败，跳过: {dead_letter_key}/{msg_id!r}, error={e}"
                )
                continue
            if not isinstance(original_data, dict):
                logger.error(
                    f"死信消息数据不是 dict，跳过: {dead_letter_key}/{msg_id!r}"
                )
                continue
            await redis.xadd(config.stream_name, original_data)
            await redis.xdel(dead_letter_key, msg_id)
            logger.info(f"消息已从死信队列恢复: {msg_id!r}")
            return True

        return False


# 队列名称常量（项目自定义——infra 不再硬编码）
# 各项目在自己的模块中定义 QueueName，比如:
#   class QueueName:
#       WORKFLOW = "dramacraft_workflow_queue"
#       DEFAULT = WORKFLOW
# 然后在 init_mq(queues=[{...}]) 传入。
class QueueName:
    """队列名称常量 —— 供 infra.mq.__init__ 重新导出。

    各项目应在自己的模块中定义项目特定的 QueueName；
    此处提供 infra 层的缺省值，避免 __init__.py 导入失败。
    """
    WORKFLOW = "workflow_queue"
    DEFAULT = WORKFLOW

# 全局实例
_mq: Optional[RedisStreamMQ] = None


async def init_mq(
    config: Dict[str, Any],
    routing: Dict[str, str],
    queues: Optional[List[Dict[str, Any]]] = None,
    handlers: Optional[Dict[str, Callable]] = None,
    default_queue: Optional[str] = None,
) -> RedisStreamMQ:
    """初始化全局消息队列

    Args:
        config: 应用配置（从中读取 redis.*）
        routing: 任务类型到队列名称的映射 {task_type: stream_name}
        queues: 队列定义列表，每个包含 name/concurrency/batch_size 等
        handlers: 任务类型到处理函数的映射 {task_type: handler_coroutine}
        default_queue: 默认队列名称（用于路由未匹配的消息）
    """
    global _mq
    
    redis_config = config.get('redis', {})
    _mq = RedisStreamMQ(redis_config)
    await _mq.connect()
    
    queue_list = queues or []
    for q in queue_list:
        # narrow q: Dict[str, Any]——防御性检查
        if not isinstance(q, dict):
            logger.warning(f"queue 配置不是 dict，跳过: {q!r}")
            continue
        q_dict: Dict[str, Any] = q
        _mq.create_queue(
            stream_name=q_dict['name'],
            group_name=q_dict.get('group'),
            maxlen=q_dict.get('maxlen', 10000),
            block_ms=q_dict.get('block_ms', 1000),
            batch_size=q_dict.get('batch_size', 10),
            max_retries=q_dict.get('max_retries', 3),
            concurrency=q_dict.get('concurrency', 3),
            claim_interval=q_dict.get('claim_interval', 30),
            claim_min_idle_ms=q_dict.get('claim_min_idle_ms', 60000),
            ttl_seconds=q_dict.get('ttl_seconds', 0)
        )
    
    routing_rules_dict = routing
    _mq.set_routing(routing_rules_dict)
    
    if default_queue:
        _mq.set_default_queue(default_queue)
    
    # 注册处理器
    if handlers:
        for task_type, handler in handlers.items():
            stream_name = _mq._routing.get(task_type, _mq._default_queue)
            # narrow 路由结果——若无路由且无默认队列，register 会抛 ValueError
            if stream_name is None:
                logger.warning(f"未找到 {task_type} 路由且无默认队列，跳过注册")
                continue
            _mq.register(stream_name, task_type, handler)
    else:
        logger.warning("没有提供处理器映射，消息队列将不会处理任何消息")
    
    # 添加 config 属性到 _mq 实例
    _mq.config = {
        'default_queue': _mq._default_queue,
        'queues': {name: {'concurrency': q.concurrency, 'batch_size': q.batch_size} 
                   for name, q in _mq._queues.items()},
        'routing': _mq._routing.copy()
    }
    
    logger.info(f"消息队列初始化完成，默认队列: {_mq._default_queue}")
    logger.info(f"可用队列: {list(_mq._queues.keys())}")
    return _mq


def get_mq() -> RedisStreamMQ:
    """获取全局消息队列实例"""
    global _mq
    if _mq is None:
        raise RuntimeError("消息队列未初始化，请先调用 init_mq")
    return _mq