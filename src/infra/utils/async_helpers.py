# macro_monitor/utils/async_helpers.py
"""
异步辅助函数 - 使用 asyncio.to_thread 避免事件循环冲突
"""
import asyncio
import time
import random
from typing import List, Callable, Any, Dict, Optional

from infra.logger import get_logger

logger = get_logger(__name__)


def sync_retry(func: Callable, max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 30.0) -> Any:
    """同步重试装饰器（在线程中执行）"""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"{func.__name__} 重试 {max_retries} 次后失败: {e}")
                raise
            
            delay = base_delay * (2 ** attempt)
            delay = min(delay, max_delay)
            delay = delay * (0.5 + random.random())
            
            logger.warning(
                f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}. "
                f"等待 {delay:.2f} 秒后重试..."
            )
            time.sleep(delay)


async def merge_sync_results(
    *functions: Callable,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    timeout: float = 60
) -> Optional[Dict[str, Any]]:
    """
    并发执行多个同步函数并合并结果
    
    使用 asyncio.to_thread 避免事件循环冲突
    
    Example:
        result = await merge_sync_results(
            scrape_coinmarketcap,
            scrape_cmc_prices,
            scrape_cmc_fgi
        )
    """
    if not functions:
        return None
    
    async def run_one(func: Callable) -> Any:
        """运行单个函数"""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(sync_retry, func, max_retries, base_delay, max_delay),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"{func.__name__} 执行超时 ({timeout}秒)")
            return None
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败: {e}")
            return None
    
    # 并发执行
    tasks = [run_one(func) for func in functions]
    results = await asyncio.gather(*tasks)
    
    # 合并结果
    merged = {}
    for func, result in zip(functions, results):
        if result and isinstance(result, dict):
            merged.update(result)
            logger.debug(f"{func.__name__} 成功，返回 {len(result)} 个字段")
        elif result is not None:
            logger.warning(f"{func.__name__} 返回非字典类型: {type(result)}")
    
    if not merged:
        logger.error("所有任务都失败")
        return None
    
    logger.info(f"成功合并 {len(merged)} 个数据字段")
    return merged