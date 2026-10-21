# src/macro_monitor/core/redis/client.py
"""
Redis 统一客户端
提供连接管理、基础操作、分布式锁
"""

import json
import asyncio
import time
import uuid
from typing import AsyncIterator, Optional, Any, Dict, List, Set, Callable, Tuple, cast
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from infra.logger import get_logger
from infra.exceptions import RedisError

logger = get_logger(__name__)


class RedisClient:
    """Redis 统一客户端（异步）"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._client: Optional[aioredis.Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._lock_tokens: Dict[str, str] = {}
        self._init_connection()
    
    def _init_connection(self) -> None:
        """初始化连接"""
        try:
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 6379)
            db = self.config.get('db', 0)
            password = self.config.get('password') or None
            
            # 获取连接池配置（支持两种格式）
            pool_config = self.config.get('connection_pool', {})
            max_connections = pool_config.get('max_connections', 
                               self.config.get('max_connections', 20))
            socket_timeout = pool_config.get('socket_timeout', 
                           self.config.get('socket_timeout', 5))
            socket_connect_timeout = pool_config.get('socket_connect_timeout',
                                   self.config.get('socket_connect_timeout', 5))
            
            self._pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=True,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)
            logger.info(f"Redis 连接成功: {host}:{port}")
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            raise RedisError(f"Redis 连接失败: {e}")
    
    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RedisError("Redis 客户端未初始化")
        return self._client

    @property
    def client(self) -> aioredis.Redis:
        """底层 redis.asyncio 客户端（公开访问器）。

        用于 pipeline / scan_iter / 批量命令等本类未封装的能力。
        优先用本类提供的方法（keys / scan_iter / pipeline 等）；
        确实需要原生能力时再用这个属性——不要再去碰 ``_get_client()``。
        """
        return self._get_client()

    # ========== Key 扫描 ==========

    async def keys(self, pattern: str = "*") -> List[str]:
        """按模式列出所有匹配的 key（返回 str）。

        ⚠️ 生产环境慎用：``KEYS`` 会阻塞 Redis 单线程。
        大 keyspace 请改用 :meth:`scan_iter`（增量扫描，不阻塞）。
        """
        raw = await self._get_client().keys(pattern)
        return [self._decode_key(k) for k in raw]

    async def scan_iter(
        self, pattern: str = "*", count: int = 100
    ) -> AsyncIterator[str]:
        """增量扫描匹配的 key（生产推荐——不阻塞 Redis）。

        用法::

            async for key in redis.scan_iter("llm:article_body:*"):
                ...

        Args:
            pattern: 匹配模式（glob 语法）
            count: 每批扫描的 hint 条数（越大吞吐越高，单次阻塞越久）
        """
        async for k in self._get_client().scan_iter(match=pattern, count=count):
            yield self._decode_key(k)

    @staticmethod
    def _decode_key(key: Any) -> str:  # noqa: ANN401
        """key 统一成 str。

        连接池开了 ``decode_responses=True``，正常返回就是 str；
        这里保留 bytes 兜底（自定义连接池 / mock 场景）。
        """
        if isinstance(key, bytes):
            return key.decode("utf-8", errors="replace")
        return str(key)

    @classmethod
    def _decode_str(cls, value: Any) -> str:  # noqa: ANN401
        """任意返回值统一成 str（bytes 走 decode 而不是 ``str()``）。

        ``str(b"abc")`` 会得到 ``"b'abc'"``——必须用 decode。
        """
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    # ========== 批量操作 ==========

    def pipeline(self, transaction: bool = True) -> Any:  # noqa: ANN401
        """创建 pipeline（批量命令，减少 RTT）。

        用法::

            async with redis.pipeline() as pipe:
                pipe.set("k1", "v1")
                pipe.get("k1")
                await pipe.execute()

        底层返回 ``redis.asyncio.client.Pipeline``。
        """
        return self._get_client().pipeline(transaction=transaction)

    async def mget(self, *keys: str) -> List[Any]:  # noqa: ANN401
        """批量取多个 key 的值（一次 RTT）。

        不存在的 key 对应位置为 None。
        """
        if not keys:
            return []
        return list(await self._get_client().mget(*keys))

    async def mset(self, mapping: Dict[str, Any]) -> bool:  # noqa: ANN401
        """批量写多个 key（一次 RTT）。

        Args:
            mapping: {key: value}；value 为 dict/list 时自动 JSON 序列化
        """
        if not mapping:
            return True
        payload: Dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(v, (dict, list)):
                payload[k] = json.dumps(v, ensure_ascii=False)
            elif not isinstance(v, str):
                payload[k] = str(v)
            else:
                payload[k] = v
        return bool(await self._get_client().mset(payload))

    async def msetnx(self, mapping: Dict[str, Any]) -> bool:
        """批量写多个 key——**仅当所有 key 都不存在时才写**（原子操作）。

        适合「配置原子下发」：要么整组配置写入成功，要么一个都不写，
        不会出现「写了一半」的中间状态。

        Args:
            mapping: {key: value}；value 为 dict/list 时自动 JSON 序列化

        Returns:
            True 表示全部写入；False 表示存在任一 key 已存在（此时不写任何）
        """
        if not mapping:
            return True
        payload: Dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(v, (dict, list)):
                payload[k] = json.dumps(v, ensure_ascii=False)
            elif not isinstance(v, str):
                payload[k] = str(v)
            else:
                payload[k] = v
        return bool(await self._get_client().msetnx(payload))

    async def delete_pattern(self, pattern: str, batch: int = 500) -> int:
        """按模式批量删除 key（用 SCAN 增量扫描，不阻塞 Redis）。

        Args:
            pattern: 匹配模式（glob 语法）
            batch: 每批删除的条数

        Returns:
            实际删除的 key 数量
        """
        deleted = 0
        batch_keys: List[str] = []
        async for key in self.scan_iter(pattern):
            batch_keys.append(key)
            if len(batch_keys) >= batch:
                deleted += await self.delete(*batch_keys)
                batch_keys = []
        if batch_keys:
            deleted += await self.delete(*batch_keys)
        return deleted

    # ========== 基础操作 ==========
    
    async def get(self, key: str) -> Any:  # noqa: ANN401  # bytes | str | None
        """获取键值"""
        return await self._get_client().get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置键值，支持 TTL"""
        client = self._get_client()
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif not isinstance(value, str):
            value = str(value)
        
        if ttl:
            return bool(await client.set(key, value, ex=ttl))
        return bool(await client.set(key, value))
    
    async def set_nx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        原子 SET NX（不存在才设置），可选 TTL。
        
        底层单命令 ``SET key value NX EX ttl``，共享 ``set`` 的序列化逻辑，
        但额外保证加锁原子性——避免 ``setnx`` + ``expire`` 两步非原子导致的死锁。
        """
        client = self._get_client()
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        elif not isinstance(value, str):
            value = str(value)
        if ttl:
            return bool(await client.set(key, value, nx=True, ex=ttl))
        return bool(await client.set(key, value, nx=True))
    
    async def delete(self, *keys: str) -> int:
        """删除键"""
        if not keys:
            return 0
        return await self._get_client().delete(*keys)
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return await self._get_client().exists(key) > 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        return await self._get_client().expire(key, ttl)
    
    async def ttl(self, key: str) -> int:
        """获取剩余生存时间"""
        return await self._get_client().ttl(key)

    async def pttl(self, key: str) -> int:
        """获取剩余生存时间（毫秒）。

        比 :meth:`ttl` 精度更高——适合需要精确控制过期窗口的场景。
        """
        return await self._get_client().pttl(key)

    async def persist(self, key: str) -> bool:
        """取消 TTL——把 key 变成永久。

        与 :meth:`expire` 成对使用。key 本来就没有 TTL 时返回 False。
        """
        return bool(await self._get_client().persist(key))

    async def pexpire(self, key: str, ms: int) -> bool:
        """设置毫秒级过期时间。"""
        return bool(await self._get_client().pexpire(key, ms))

    async def expireat(self, key: str, when: Any) -> bool:  # noqa: ANN401
        """设置绝对时间过期（``when`` 为 datetime 或时间戳秒）。"""
        return bool(await self._get_client().expireat(key, when))

    async def type(self, key: str) -> str:
        """返回 key 的类型（string / list / set / zset / hash / none）。

        动态 key 场景下判断该用哪套操作方式。
        """
        return self._decode_str(await self._get_client().type(key))

    async def strlen(self, key: str) -> int:
        """返回字符串 key 的长度（字节数）。"""
        return int(await self._get_client().strlen(key))

    async def append(self, key: str, value: Any) -> int:  # noqa: ANN401
        """在字符串末尾追加内容，返回追加后的长度。

        key 不存在时等价于 SET（长度为追加内容的长度）。
        """
        return int(await self._get_client().append(key, self._encode_list_value(value)))

    async def getrange(self, key: str, start: int, end: int) -> str:
        """截取字符串片段（0-based，含两端；负数索引从尾部数）。

        等价于 Redis 的 ``SUBSTR``。key 不存在返回空串。
        """
        value = await self._get_client().getrange(key, start, end)
        return self._decode_str(value) if value is not None else ""

    async def setrange(self, key: str, offset: int, value: Any) -> int:  # noqa: ANN401
        """从指定偏移处覆写字符串，返回写入后的总长度。

        超出当前长度的部分用零字节填充。
        """
        return int(
            await self._get_client().setrange(
                key, offset, self._encode_list_value(value)
            )
        )

    async def getdel(self, key: str) -> Optional[str]:
        """原子「取值并删除」——key 不存在返回 None。

        适合「取出即消费」的场景（比 get + delete 两步更安全，不会重复消费）。
        """
        value = await self._get_client().getdel(key)
        return self._decode_str(value) if value is not None else None

    async def getex(self, key: str, ttl: Optional[int] = None) -> Optional[str]:
        """原子「取值并（可选）重设 TTL」。

        ``ttl=None`` 时只取值不动过期时间；``ttl>0`` 时把 TTL 重设为该秒数
        （滑动过期窗口）。key 不存在返回 None。
        """
        if ttl:
            value = await self._get_client().getex(key, ex=ttl)
        else:
            value = await self._get_client().getex(key)
        return self._decode_str(value) if value is not None else None

    async def unlink(self, *keys: str) -> int:
        """异步删除 key（UNLINK）——不阻塞主线程。

        与 :meth:`delete`（阻塞式 DEL）的区别：删除大 key 时 UNLINK 由后台线程
        释放内存，主线程立即返回。key 不存在不计入删除数。
        """
        if not keys:
            return 0
        return int(await self._get_client().unlink(*keys))

    async def dbsize(self) -> int:
        """返回当前数据库的 key 数量（巡检 / 监控用）。"""
        return int(await self._get_client().dbsize())

    async def rename(self, source: str, dest: str) -> bool:
        """重命名 key。

        source 不存在时抛 ``redis.exceptions.ResponseError``。
        """
        return bool(await self._get_client().rename(source, dest))

    async def copy(self, source: str, dest: str) -> bool:
        """复制 key（Redis 6.2+）。

        source 不存在时返回 False（不报错）。dest 已存在会被覆盖。
        """
        return bool(await self._get_client().copy(source, dest))

    async def touch(self, *keys: str) -> int:
        """批量刷新 key 的 LRU 空闲时间（不读取 value）。

        用于把「偶尔访问一下就不该被淘汰」的 key 标记为活跃。
        """
        if not keys:
            return 0
        return int(await self._get_client().touch(*keys))

    # ========== 计数器 ==========
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """递增计数器"""
        return await self._get_client().incrby(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """递减计数器"""
        return await self._get_client().decrby(key, amount)
    
    # ========== Hash 操作 ==========
    
    async def hset(self, key: str, field: Optional[str] = None, value: Any = None, mapping: Optional[Dict[str, Any]] = None) -> int:  # noqa: ANN401
        """设置 Hash 字段。

        三种用法（redis-py 原生语义）：

        1. ``hset(key, "f", "v")`` —— 单字段
        2. ``hset(key, mapping={"f1": "v1", "f2": "v2"})`` —— 批量（一次 RTT）
        3. ``hset(key, "f", "v", mapping={...})`` —— 混合

        Args:
            key: Hash key
            field: 字段名（与 ``mapping`` 二选一或混用）
            value: 字段值
            mapping: 字段名 -> 值 的映射（值会被 JSON 序列化）

        Returns:
            新写入的字段数（更新已有字段不计数）
        """
        if mapping:
            payload: Dict[str, Any] = {
                str(k): self._encode_list_value(v) for k, v in mapping.items()
            }
            if field is not None:
                payload[str(field)] = self._encode_list_value(value)
            result: Any = await self._get_client().hset(
                key, mapping=cast(Any, payload)
            )
            return int(result) if result is not None else 0
        if field is None:
            raise ValueError("hset 需要 field 或 mapping 参数")
        return int(
            await self._get_client().hset(key, field, self._encode_list_value(value))
        )

    async def hget(self, key: str, field: str) -> Any:  # noqa: ANN401  # bytes | str | None
        """获取 Hash 字段"""
        return await self._get_client().hget(key, field)

    async def hgetall(self, key: str) -> Dict[str, str]:
        """获取整个 Hash（redis-py 返 dict[bytes|str, bytes|str]——decode_responses=True 实际是 str）"""
        raw: Any = await self._get_client().hgetall(key)
        if not raw:
            return {}
        # 用 _decode_str 而非 str()——bytes 会变成 "b'k'"（自定义连接池场景）
        return {self._decode_str(k): self._decode_str(v) for k, v in raw.items()}

    async def hdel(self, key: str, *fields: str) -> int:
        """删除 Hash 字段"""
        if not fields:
            return 0
        return int(await self._get_client().hdel(key, *fields))

    async def hlen(self, key: str) -> int:
        """Hash 字段数量。"""
        return int(await self._get_client().hlen(key))

    async def hexists(self, key: str, field: str) -> bool:
        """判断 Hash 字段是否存在（比 ``hget is not None`` 省一次取值）。"""
        return bool(await self._get_client().hexists(key, field))

    async def hmget(self, key: str, *fields: str) -> List[Optional[str]]:
        """批量获取 Hash 字段（一次 RTT）。

        字段不存在时对应位置为 None（保持位置对应关系）。
        """
        if not fields:
            return []
        raw: Any = await self._get_client().hmget(key, list(fields))
        if not raw:
            return []
        return [self._decode_str(v) if v is not None else None for v in raw]

    async def hkeys(self, key: str) -> List[str]:
        """Hash 所有字段名。"""
        raw: Any = await self._get_client().hkeys(key)
        return [self._decode_str(f) for f in raw] if raw else []

    async def hvals(self, key: str) -> List[str]:
        """Hash 所有字段值。"""
        raw: Any = await self._get_client().hvals(key)
        return [self._decode_str(v) for v in raw] if raw else []

    async def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        """Hash 字段级计数器——返回累加后的值。

        典型用途：点赞数、库存扣减、字段级 PV 计数。
        """
        return int(await self._get_client().hincrby(key, field, amount))

    async def hincrbyfloat(self, key: str, field: str, amount: float) -> float:
        """Hash 字段级浮点累加——返回累加后的值。

        Redis 内部用十进制字符串做运算，避免浮点精度丢失。
        """
        return float(await self._get_client().hincrbyfloat(key, field, amount))

    async def hsetnx(self, key: str, field: str, value: Any) -> bool:  # noqa: ANN401
        """仅当字段不存在时才写入（字段级幂等 / 轻量锁）。

        返回 True 表示写入成功，False 表示字段已存在。
        """
        return bool(
            await self._get_client().hsetnx(
                key, field, self._encode_list_value(value)
            )
        )

    # ========== Set 操作（去重用） ==========
    
    async def sadd(self, key: str, *members: str) -> int:
        """添加成员到 Set"""
        return await self._get_client().sadd(key, *members)
    
    async def sismember(self, key: str, member: str) -> bool:
        """检查是否为 Set 成员（redis-py 5+ 返 Literal[0, 1]——mypy 期望 bool）"""
        result = await self._get_client().sismember(key, member)
        return bool(result)
    
    async def smembers(self, key: str) -> Set[str]:
        """获取 Set 所有成员"""
        raw: Any = await self._get_client().smembers(key)
        # 用 _decode_str 而非 str()——bytes 会变成 "b'm'"（自定义连接池场景）
        return {self._decode_str(m) for m in raw} if raw else set()
    
    async def srem(self, key: str, *members: str) -> int:
        """从 Set 移除成员"""
        return await self._get_client().srem(key, *members)

    async def sinter(self, *keys: str) -> Set[str]:
        """多个 Set 的交集"""
        if not keys:
            return set()
        raw: Any = await self._get_client().sinter(list(keys))
        return {self._decode_str(m) for m in raw} if raw else set()

    async def sunion(self, *keys: str) -> Set[str]:
        """多个 Set 的并集"""
        if not keys:
            return set()
        raw: Any = await self._get_client().sunion(list(keys))
        return {self._decode_str(m) for m in raw} if raw else set()

    async def sdiff(self, *keys: str) -> Set[str]:
        """多个 Set 的差集（第一个减去其余）"""
        if not keys:
            return set()
        raw: Any = await self._get_client().sdiff(list(keys))
        return {self._decode_str(m) for m in raw} if raw else set()

    async def sdiffstore(self, dest: str, *keys: str) -> int:
        """差集写入 dest，返回写入元素数量"""
        if not keys:
            return 0
        return int(await self._get_client().sdiffstore(dest, list(keys)))

    async def sinterstore(self, dest: str, *keys: str) -> int:
        """交集写入 dest，返回写入元素数量"""
        if not keys:
            return 0
        return int(await self._get_client().sinterstore(dest, list(keys)))

    async def sunionstore(self, dest: str, *keys: str) -> int:
        """并集写入 dest，返回写入元素数量"""
        if not keys:
            return 0
        return int(await self._get_client().sunionstore(dest, list(keys)))

    async def scard(self, key: str) -> int:
        """Set 元素数量"""
        return int(await self._get_client().scard(key))

    async def srandmember(self, key: str, count: Optional[int] = None) -> Any:  # noqa: ANN401
        """从 Set 随机取成员（不删除）。

        Args:
            key: Set key
            count: 取几个；不传则取 1 个（返回 str 或 None），传了则返回 list

        Note:
            count 为负数时允许返回重复元素。
        """
        if count is None:
            value = await self._get_client().srandmember(key)
            return self._decode_str(value) if value is not None else None
        if count <= 0:
            return []
        raw: Any = await self._get_client().srandmember(key, count)
        return [self._decode_str(m) for m in raw] if raw else []

    async def spop(self, key: str, count: Optional[int] = None) -> Any:  # noqa: ANN401
        """从 Set 随机弹出成员（**会删除**）。

        Args:
            key: Set key
            count: 弹几个；不传则弹 1 个（返回 str 或 None），传了则返回 list
        """
        if count is None:
            value = await self._get_client().spop(key)
            return self._decode_str(value) if value is not None else None
        if count <= 0:
            return []
        raw: Any = await self._get_client().spop(key, count)
        return [self._decode_str(m) for m in raw] if raw else []

    async def smove(self, source: str, dest: str, member: str) -> bool:
        """把成员从 source Set 原子移动到 dest Set。

        返回 True 表示移动成功，False 表示 source 里没有该成员。
        """
        return bool(await self._get_client().smove(source, dest, member))

    async def sintercard(
        self, keys: List[str], limit: int = 0
    ) -> int:
        """多个 Set 交集的**基数**（不去重计算元素个数）——Redis 7.0+。

        比 :meth:`sinter` 更省内存：只算数量不实际构造交集集合。

        Args:
            keys: Set key 列表
            limit: 超过该数量就提前停止（0 表示不限制）
        """
        if not keys:
            return 0
        result: Any = await self._get_client().sintercard(
            len(keys), cast(Any, keys), limit
        )
        return int(result) if result is not None else 0

    # ========== List 操作（队列 / 栈 / 最近 N 条） ==========

    async def lpush(self, key: str, *values: Any) -> int:  # noqa: ANN401
        """从头部插入一个或多个元素，返回列表长度。

        注意：``LPUSH`` 是「左推」——后插入的元素在**前面**。
        要当队列用（FIFO）请用 :meth:`rpush` + :meth:`lpop`。
        """
        if not values:
            return await self.llen(key)
        return int(await self._get_client().lpush(key, *self._encode_list_values(values)))

    async def rpush(self, key: str, *values: Any) -> int:  # noqa: ANN401
        """从尾部追加一个或多个元素，返回列表长度（队列 FIFO 用这个）。"""
        if not values:
            return await self.llen(key)
        return int(await self._get_client().rpush(key, *self._encode_list_values(values)))

    async def lpushx(self, key: str, *values: Any) -> int:  # noqa: ANN401
        """仅当 key 已存在时才从头部插入（不存在则不创建）。"""
        if not values:
            return await self.llen(key)
        return int(await self._get_client().lpushx(key, *self._encode_list_values(values)))

    async def rpushx(self, key: str, *values: Any) -> int:  # noqa: ANN401
        """仅当 key 已存在时才从尾部追加。"""
        if not values:
            return await self.llen(key)
        return int(await self._get_client().rpushx(key, *self._encode_list_values(values)))

    async def lpop(self, key: str, count: Optional[int] = None) -> Any:  # noqa: ANN401
        """从头部弹出元素。

        Args:
            count: 弹出个数；不传则弹 1 个（返回 str），传了则返回 list
        """
        if count is None:
            value = await self._get_client().lpop(key)
            # 与 count 路径保持一致——统一成 str（自定义连接池可能返回 bytes）
            return self._decode_str(value) if value is not None else None
        if count <= 0:
            return []
        raw: Any = await self._get_client().lpop(key, count)
        return [self._decode_str(v) for v in raw] if raw else []

    async def rpop(self, key: str, count: Optional[int] = None) -> Any:  # noqa: ANN401
        """从尾部弹出元素（栈 LIFO 用这个）。

        Args:
            count: 弹出个数；不传则弹 1 个（返回 str），传了则返回 list
        """
        if count is None:
            value = await self._get_client().rpop(key)
            return self._decode_str(value) if value is not None else None
        if count <= 0:
            return []
        raw: Any = await self._get_client().rpop(key, count)
        return [self._decode_str(v) for v in raw] if raw else []

    async def blpop(self, key: str, timeout: float = 0) -> Optional[Tuple[str, str]]:
        """阻塞式从头部弹出（队列消费者用）。

        Args:
            key: 列表 key
            timeout: 阻塞秒数；0 表示永久阻塞（谨慎使用）

        Returns:
            (key, value) 元组；超时返回 None
        """
        result: Any = await self._get_client().blpop([key], timeout=timeout)
        if not result:
            return None
        # redis-py 直接返回 (key, value) 元组——不是 [(key, value)]
        k, v = result
        return (self._decode_str(k), self._decode_str(v))

    async def brpop(self, key: str, timeout: float = 0) -> Optional[Tuple[str, str]]:
        """阻塞式从尾部弹出（栈消费者用）。

        Args:
            key: 列表 key
            timeout: 阻塞秒数；0 表示永久阻塞（谨慎使用）

        Returns:
            (key, value) 元组；超时返回 None
        """
        result: Any = await self._get_client().brpop([key], timeout=timeout)
        if not result:
            return None
        # redis-py 直接返回 (key, value) 元组——不是 [(key, value)]
        k, v = result
        return (self._decode_str(k), self._decode_str(v))

    async def lrange(self, key: str, start: int = 0, stop: int = -1) -> List[str]:
        """取列表区间元素（0-based，含两端；stop=-1 表示到末尾）。

        取「最近 N 条」用 ``lrange(key, -N, -1)``。
        """
        raw: Any = await self._get_client().lrange(key, start, stop)
        return [self._decode_str(v) for v in raw] if raw else []

    async def llen(self, key: str) -> int:
        """列表长度"""
        return int(await self._get_client().llen(key))

    async def lindex(self, key: str, index: int) -> Optional[str]:
        """取列表第 index 个元素（0-based；负数从尾部数）。"""
        value = await self._get_client().lindex(key, index)
        return self._decode_str(value) if value is not None else None

    async def lset(self, key: str, index: int, value: Any) -> bool:  # noqa: ANN401
        """设置列表第 index 个元素（index 越界抛 IndexError）。"""
        return bool(await self._get_client().lset(key, index, self._encode_list_value(value)))

    async def lrem(self, key: str, count: int, value: Any) -> int:  # noqa: ANN401
        """从列表删除匹配 value 的元素。

        Args:
            count: >0 从头往尾删 count 个；<0 从尾往头删 |count| 个；0 删全部
            value: 目标值
        """
        return int(await self._get_client().lrem(key, count, self._encode_list_value(value)))

    async def ltrim(self, key: str, start: int, stop: int) -> bool:
        """只保留列表 [start, stop] 区间的元素（裁剪两端）。"""
        return bool(await self._get_client().ltrim(key, start, stop))

    async def lpos(
        self,
        key: str,
        value: Any,  # noqa: ANN401
        rank: int = 1,
        count: Optional[int] = None,
        maxlen: Optional[int] = None,
    ) -> Any:  # noqa: ANN401
        """查找列表中元素的位置。

        Args:
            key: 列表 key
            value: 要查找的值
            rank: 第几个匹配（正数从头数，负数从尾数）
            count: 返回几个匹配位置（0 表示只找第 rank 个）
            maxlen: 只扫描前 maxlen 个元素

        Returns:
            不传 count 时返回位置 int（或 None）；传 count 时返回位置列表
        """
        encoded = self._encode_list_value(value)
        if count is None:
            result: Any = await self._get_client().lpos(key, encoded, rank=rank)
            return int(result) if result is not None else None
        raw: Any = await self._get_client().lpos(
            key, encoded, rank=rank, count=count, maxlen=maxlen
        )
        return [int(p) for p in raw] if raw else []

    async def linsert(
        self, key: str, where: str, refvalue: Any, value: Any  # noqa: ANN401
    ) -> int:
        """在列表中 refvalue 的前面/后面插入新值。

        Args:
            key: 列表 key
            where: ``"before"`` 或 ``"after"``
            refvalue: 参照元素
            value: 要插入的新值

        Returns:
            插入后的列表长度；**参照元素不存在时返回 -1**
        """
        if where not in ("before", "after"):
            raise ValueError(f"where 必须是 'before' 或 'after'，收到: {where}")
        return int(
            await self._get_client().linsert(
                key, where, self._encode_list_value(refvalue),
                self._encode_list_value(value),
            )
        )

    async def rpoplpush(self, source: str, dest: str) -> Optional[str]:
        """从 source 尾部弹出一个元素，压入 dest 头部（原子操作）。

        典型用途：把「处理中」列表的元素挪到「等待中」列表。
        """
        value = await self._get_client().rpoplpush(source, dest)
        return self._decode_str(value) if value is not None else None

    async def lmove(
        self,
        source: str,
        dest: str,
        where_from: str = "left",
        where_to: str = "left",
    ) -> Optional[str]:
        """原子地把元素从 source 移到 dest（Redis 6.2+）。

        Args:
            source: 源列表
            dest: 目标列表
            where_from: 从源列表哪端取（"left" / "right"）
            where_to: 往目标列表哪端放（"left" / "right"）
        """
        value = await self._get_client().lmove(
            source, dest, where_from, where_to
        )
        return self._decode_str(value) if value is not None else None

    @staticmethod
    def _encode_list_value(value: Any) -> str:  # noqa: ANN401
        """列表元素统一编码成 str（与 string / hash 的序列化约定一致）。"""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, str):
            return value
        return str(value)

    @classmethod
    def _encode_list_values(cls, values: Any) -> List[str]:  # noqa: ANN401
        """批量编码列表元素。"""
        return [cls._encode_list_value(v) for v in values]

    # ========== ZSET 操作（有序集合，用于限流/排行榜） ==========
    
    async def zadd(
        self,
        key: str,
        mapping: Dict[str, float],
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
        gt: bool = False,
        lt: bool = False,
    ) -> int:
        """
        添加元素到有序集合。

        Args:
            key: Redis key
            mapping: {member: score} 字典
            nx: 只添加不存在的成员（幂等写入）
            xx: 只更新已存在的成员
            ch: 返回「实际发生变更」的成员数（而非新增数）
            gt: 只在新分数**大于**当前分数时更新（Redis 6.2+）
            lt: 只在新分数**小于**当前分数时更新（Redis 6.2+）

        Returns:
            添加/变更的元素数量

        Note:
            ``nx`` 与 ``xx`` 互斥；``gt`` 与 ``lt`` 互斥；``nx`` 不能与 ``gt``/``lt`` 同用。
        """
        if not mapping:
            return 0
        result = await self._get_client().zadd(
            key, mapping, nx=nx, xx=xx, ch=ch, gt=gt, lt=lt
        )
        return int(result) if result is not None else 0

    async def zmscore(self, key: str, *members: str) -> List[Optional[float]]:
        """批量获取成员的分数（一次 RTT）。

        成员不存在时对应位置为 None。
        """
        if not members:
            return []
        raw: Any = await self._get_client().zmscore(key, list(members))
        if not raw:
            return []
        return [float(v) if v is not None else None for v in raw]

    async def zrandmember(self, key: str, count: Optional[int] = None) -> Any:  # noqa: ANN401
        """从有序集合随机取成员（不删除）。

        Args:
            key: ZSet key
            count: 取几个；不传则取 1 个（返回 str 或 None），传了则返回 list
        """
        if count is None:
            value = await self._get_client().zrandmember(key)
            return self._decode_str(value) if value is not None else None
        if count <= 0:
            return []
        raw: Any = await self._get_client().zrandmember(key, count)
        return [self._decode_str(m) for m in raw] if raw else []

    async def zunion(
        self,
        keys: List[str],
        aggregate: Optional[str] = None,
        withscores: bool = False,
    ) -> list:
        """多个有序集合的并集（Redis 6.2+）。

        Args:
            keys: ZSet key 列表
            aggregate: 分数聚合方式——``"SUM"``（默认）/ ``"MIN"`` / ``"MAX"``
            withscores: 是否带分数
        """
        if not keys:
            return []
        return await self._get_client().zunion(
            keys, aggregate=aggregate, withscores=withscores
        )

    async def zinter(
        self,
        keys: List[str],
        aggregate: Optional[str] = None,
        withscores: bool = False,
    ) -> list:
        """多个有序集合的交集（Redis 6.2+）。"""
        if not keys:
            return []
        return await self._get_client().zinter(
            keys, aggregate=aggregate, withscores=withscores
        )

    async def zdiff(
        self,
        keys: List[str],
        withscores: bool = False,
    ) -> list:
        """多个有序集合的差集（第一个减去其余，Redis 6.2+）。"""
        if not keys:
            return []
        return await self._get_client().zdiff(keys, withscores=withscores)

    async def zunionstore(
        self, dest: str, keys: List[str], aggregate: Optional[str] = None
    ) -> int:
        """并集写入 dest，返回写入元素数量。"""
        if not keys:
            return 0
        return int(
            await self._get_client().zunionstore(dest, keys, aggregate=aggregate)
        )

    async def zinterstore(
        self, dest: str, keys: List[str], aggregate: Optional[str] = None
    ) -> int:
        """交集写入 dest，返回写入元素数量。"""
        if not keys:
            return 0
        result: Any = await cast(Any, self._get_client().zinterstore)(
            dest, cast(Any, keys), aggregate=aggregate
        )
        return int(result) if result is not None else 0

    async def zdiffstore(self, dest: str, keys: List[str]) -> int:
        """差集写入 dest，返回写入元素数量。"""
        if not keys:
            return 0
        return int(await self._get_client().zdiffstore(dest, keys))

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """
        按分数范围移除元素
        
        Args:
            key: Redis key
            min_score: 最小分数（包含）
            max_score: 最大分数（包含）
        
        Returns:
            移除的元素数量
        """
        return await self._get_client().zremrangebyscore(key, min_score, max_score)
    
    async def zcard(self, key: str) -> int:
        """
        获取有序集合的元素数量
        
        Args:
            key: Redis key
        
        Returns:
            元素数量
        """
        return await self._get_client().zcard(key)
    
    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list:
        """
        按索引范围获取有序集合的元素
        
        Args:
            key: Redis key
            start: 起始索引
            stop: 结束索引
            withscores: 是否返回分数
        
        Returns:
            元素列表，如果 withscores=True，返回 [(member, score), ...]
        """
        return await self._get_client().zrange(key, start, stop, withscores=withscores)
    
    async def zrangebyscore(
        self, 
        key: str, 
        min_score: float, 
        max_score: float, 
        withscores: bool = False,
        offset: Optional[int] = None,
        count: Optional[int] = None
    ) -> list:
        """
        按分数范围获取有序集合的元素
        
        Args:
            key: Redis key
            min_score: 最小分数（包含）
            max_score: 最大分数（包含）
            withscores: 是否返回分数
            offset: 偏移量（分页）
            count: 返回数量（分页）
        
        Returns:
            元素列表
        """
        client = self._get_client()
        if offset is not None and count is not None:
            return await client.zrangebyscore(
                key, min_score, max_score, 
                withscores=withscores, 
                start=offset, 
                num=count
            )
        return await client.zrangebyscore(key, min_score, max_score, withscores=withscores)
    
    async def zrem(self, key: str, *members: str) -> int:
        """
        移除有序集合中的指定元素
        
        Args:
            key: Redis key
            members: 要移除的元素
        
        Returns:
            移除的元素数量
        """
        if not members:
            return 0
        return await self._get_client().zrem(key, *members)
    
    async def zscore(self, key: str, member: str) -> Optional[float]:
        """
        获取元素的分数
        
        Args:
            key: Redis key
            member: 元素
        
        Returns:
            分数，不存在返回 None
        """
        return await self._get_client().zscore(key, member)
    
    async def zcount(self, key: str, min_score: float, max_score: float) -> int:
        """
        获取分数范围内的元素数量

        Args:
            key: Redis key
            min_score: 最小分数（包含）
            max_score: 最大分数（包含）

        Returns:
            元素数量
        """
        return await self._get_client().zcount(key, min_score, max_score)

    async def zrank(self, key: str, member: str) -> Optional[int]:
        """成员在有序集合中的排名（按分数升序，0-based）。

        分数大的排名靠后。成员不存在返回 None。
        """
        rank: Any = await self._get_client().zrank(key, member)
        return int(rank) if rank is not None else None

    async def zrevrank(self, key: str, member: str) -> Optional[int]:
        """成员在有序集合中的倒序排名（按分数降序，0-based）。"""
        rank: Any = await self._get_client().zrevrank(key, member)
        return int(rank) if rank is not None else None

    async def zremrangebyrank(self, key: str, start: int, stop: int) -> int:
        """按排名区间删除元素（0-based，含两端；stop=-1 表示到末尾）。"""
        return int(await self._get_client().zremrangebyrank(key, start, stop))

    async def zrevrange(
        self, key: str, start: int, stop: int, withscores: bool = False
    ) -> list:
        """按分数降序取区间元素（取「排行榜前 N」用这个）。"""
        return await self._get_client().zrevrange(
            key, start, stop, withscores=withscores
        )

    async def zrevrangebyscore(
        self,
        key: str,
        max_score: float,
        min_score: float,
        withscores: bool = False,
        offset: Optional[int] = None,
        count: Optional[int] = None,
    ) -> list:
        """按分数区间降序取元素。

        Args:
            key: Redis key
            max_score: 最大分数（包含）
            min_score: 最小分数（包含）
            withscores: 是否带分数
            offset: 分页偏移
            count: 返回数量
        """
        client = self._get_client()
        if offset is not None and count is not None:
            return await client.zrevrangebyscore(
                key, max_score, min_score,
                withscores=withscores, start=offset, num=count,
            )
        return await client.zrevrangebyscore(
            key, max_score, min_score, withscores=withscores
        )


    async def zpopmin(self, key: str, count: int = 1) -> list:
        """
        弹出分数最小的元素
        
        Args:
            key: Redis key
            count: 弹出数量
        
        Returns:
            [(member, score), ...]
        """
        return await self._get_client().zpopmin(key, count)
    
    async def zpopmax(self, key: str, count: int = 1) -> list:
        """
        弹出分数最大的元素
        
        Args:
            key: Redis key
            count: 弹出数量
        
        Returns:
            [(member, score), ...]
        """
        return await self._get_client().zpopmax(key, count)
    
    # ========== 分布式锁 ==========
    
    _UNLOCK_LUA = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        return redis.call('del', KEYS[1])
    else
        return 0
    end
    """

    async def lock(self, key: str, ttl: int = 30) -> bool:
        """
        获取分布式锁（原子安全版）
        
        Args:
            key: 锁的键名
            ttl: 锁的过期时间（秒）
        
        Returns:
            是否获取成功
        """
        lock_key = f"lock:{key}"
        token = uuid.uuid4().hex
        acquired = await self.set_nx(lock_key, token, ttl=ttl)
        if acquired:
            self._lock_tokens[key] = token
        return acquired
    
    async def unlock(self, key: str, token: Optional[str] = None) -> bool:
        """
        释放分布式锁（持有者校验版）
        
        Args:
            key: 锁的键名
            token: 持有者 token；不传时使用 lock() 记录的 token
        
        Returns:
            是否释放成功
        """
        lock_key = f"lock:{key}"
        check = token if token is not None else self._lock_tokens.pop(key, None)
        if check is None:
            return False
        client = self._get_client()
        result = await client.eval(self._UNLOCK_LUA, 1, lock_key, check)
        return bool(result)
    
    @asynccontextmanager
    async def lock_context(
        self, key: str, ttl: int = 30, wait: bool = False, timeout: int = 10
    ) -> "AsyncIterator[bool]":
        """
        分布式锁上下文管理器
        
        Args:
            key: 锁的键名
            ttl: 锁的过期时间（秒）
            wait: 是否等待获取锁
            timeout: 等待超时时间（秒）
        
        Yields:
            bool: 是否获取到锁
        """
        acquired = await self.lock(key, ttl)
        if not acquired and wait:
            start = time.time()
            while not acquired and (time.time() - start) < timeout:
                await asyncio.sleep(0.5)
                acquired = await self.lock(key, ttl)
        
        try:
            yield acquired
        finally:
            if acquired:
                await self.unlock(key)
    
    # ========== 高级缓存 ==========
    
    async def get_or_set(self, key: str, func: Callable, ttl: Optional[int] = None) -> Any:
        """
        获取缓存，如果不存在则通过函数设置
        
        Args:
            key: 缓存键名
            func: 获取数据的函数（可以是同步或异步）
            ttl: 缓存过期时间（秒）
        
        Returns:
            缓存的数据
        """
        cached = await self.get(key)
        if cached is not None:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                return cached
        
        result = func()
        if asyncio.iscoroutinefunction(func):
            result = await result
        await self.set(key, result, ttl)
        return result
    
    # ========== 健康检查 ==========

    async def ping(self) -> bool:
        """
        健康检查

        Returns:
            Redis 是否可用
        """
        try:
            return await self._get_client().ping()
        except Exception:
            return False

    # ========== HyperLogLog（近似去重计数） ==========

    async def pfadd(self, key: str, *values: Any) -> int:  # noqa: ANN401
        """向 HyperLogLog 添加元素——返回「是否发生变化」（0/1）。

        HyperLogLog 用固定大小（~12KB）估算基数，适合 UV / 独立访客统计
        这类「允许小概率误差」的场景。

        Returns:
            1 表示 HyperLogLog 被修改，0 表示没变
        """
        if not values:
            return 0
        encoded = [self._encode_list_value(v) for v in values]
        return int(await self._get_client().pfadd(key, *encoded))

    async def pfcount(self, *keys: str) -> int:
        """返回 HyperLogLog 的估算基数（多个 key 会先做并集再统计）。"""
        if not keys:
            return 0
        return int(await self._get_client().pfcount(*keys))

    async def pfmerge(self, dest: str, *sources: str) -> bool:
        """把多个 HyperLogLog 合并到 dest（不删除源）。"""
        if not sources:
            return False
        return bool(await self._get_client().pfmerge(dest, *sources))

    # ========== 运维 / 诊断 ==========

    async def memory_usage(self, key: str, samples: int = 5) -> Optional[int]:
        """返回 key 占用的内存字节数——**大 key 排查**的必备工具。

        Args:
            key: 要检查的 key
            samples: 采样元素数（影响嵌套类型估算精度）

        Returns:
            字节数；key 不存在返回 None
        """
        value = await self._get_client().memory_usage(key, samples)
        return int(value) if value is not None else None

    async def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """返回服务端运行信息（INFO）。

        Args:
            section: 段落名（``"server"`` / ``"memory"`` / ``"clients"`` / ``"stats"`` 等）；
                     不传则返回全部

        Returns:
            键值字典（值可能是 str / int / list）
        """
        raw: Any = await self._get_client().info(section)
        return dict(raw) if raw else {}

    async def close(self) -> None:
        """关闭连接"""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis 连接已关闭")


# 全局客户端实例
_redis_client: Optional[RedisClient] = None


async def init_redis(config: Dict[str, Any]) -> RedisClient:
    """初始化全局 Redis 客户端"""
    global _redis_client
    _redis_client = RedisClient(config.get('redis', {}))
    await _redis_client.ping()
    logger.info("Redis 全局客户端初始化完成")
    return _redis_client


def get_redis() -> Optional[RedisClient]:
    """获取全局 Redis 客户端"""
    global _redis_client
    if _redis_client is None:
        logger.warning("Redis 客户端未初始化")
    return _redis_client

async def close_redis() -> None:
    """关闭 Redis 连接"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis 全局客户端已关闭")