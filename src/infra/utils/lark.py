
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lark 通知处理器 - 纯发送功能
只负责发送消息到 Lark Webhook，不包含任何业务逻辑
支持重试、超时控制、多种消息格式
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union

import aiohttp

from infra.logger import get_logger
from infra.config import get_config

logger = get_logger(__name__)

# Lark Webhook 配置（启动时加载一次）
_webhook = None
_retry_config = None


def _get_webhook() -> Optional[str]:
    """获取 Lark Webhook URL（懒加载）"""
    global _webhook
    if _webhook is None:
        config = get_config()
        _webhook = config.get('lark', {}).get('webhook', '')
        if not _webhook:
            logger.warning("Lark Webhook 未配置")
    return _webhook


def _get_retry_config() -> Dict[str, Any]:
    """获取重试配置（懒加载）"""
    global _retry_config
    if _retry_config is None:
        config = get_config()
        lark_config = config.get('lark', {})
        _retry_config = lark_config.get('retry', {})
    
    return {
        'max_retries': _retry_config.get('max_retries', 3),
        'base_delay': _retry_config.get('base_delay', 1.0),
        'max_delay': _retry_config.get('max_delay', 10.0),
        'timeout': _retry_config.get('timeout', 10)
    }


async def _send_request(data: Dict[str, Any], webhook_url: str, timeout: int) -> Dict[str, Any]:
    """
    发送单次请求到 Lark
    
    Args:
        data: 消息数据
        webhook_url: Webhook URL
        timeout: 超时时间（秒）
    
    Returns:
        发送结果
    """
    headers = {"Content-Type": "application/json"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False),
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        result = await response.json()
                        if result.get("code") == 0:
                            return {"status": "success", "message": "Lark发送成功"}
                        else:
                            # 业务错误，不重试
                            error_msg = result.get('msg', 'unknown error')
                            logger.error(f"Lark API错误: code={result.get('code')}, msg={error_msg}")
                            return {"status": "error", "error": error_msg, "should_retry": False}
                    except json.JSONDecodeError:
                        logger.error(f"Lark响应JSON解析失败: {response_text[:200]}")
                        return {"status": "error", "error": "invalid json response", "should_retry": False}
                else:
                    # HTTP 错误，某些状态码可以重试
                    should_retry = response.status in [502, 503, 504]
                    logger.error(f"Lark HTTP错误: {response.status}, 响应: {response_text[:200]}")
                    return {
                        "status": "error", 
                        "error": f"HTTP {response.status}",
                        "should_retry": should_retry
                    }
                    
    except asyncio.TimeoutError:
        logger.error(f"Lark请求超时 ({timeout}秒)")
        return {"status": "error", "error": "timeout", "should_retry": True}
        
    except aiohttp.ClientError as e:
        logger.error(f"Lark客户端错误: {e}")
        return {"status": "error", "error": str(e), "should_retry": True}
        
    except Exception as e:
        logger.error(f"Lark未知错误: {e}")
        return {"status": "error", "error": str(e), "should_retry": False}


def _build_message_data(
    content: Union[str, Dict[str, Any]],
    msg_type: str = "text",
    title: Optional[str] = None,
    template: str = "blue",
    fields: Optional[List[Union[str, tuple]]] = None,
    is_raw: bool = False
) -> Dict[str, Any]:
    """
    构建 Lark 消息数据
    
    Args:
        content: 消息内容
        msg_type: 消息类型
        title: 卡片标题
        template: 卡片模板颜色
        fields: 额外字段
        is_raw: 是否直接使用原始内容
    
    Returns:
        Lark 消息数据
    """
    if is_raw:
        return content  # type: ignore[return-value]  # is_raw: content 本身是 dict（上游约定）
    
    if msg_type == "interactive":
        # 构建卡片元素
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": content
                }
            }
        ]
        
        # 添加额外字段
        if fields:
            lark_fields = []
            for field in fields:
                if isinstance(field, tuple):
                    field_content, is_short = field
                else:
                    field_content, is_short = field, False
                
                lark_fields.append({
                    "is_short": is_short,
                    "text": {
                        "tag": "lark_md",
                        "content": field_content
                    }
                })
            elements.append({
                "tag": "div",
                "fields": lark_fields
            })
        
        data = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "elements": elements
            }
        }
        
        if title:
            data["card"]["header"] = {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": template
            }
        
        return data
    
    elif msg_type == "markdown":
        data = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content
                    }
                ]
            }
        }
        if title:
            data["card"]["header"] = {  # type: ignore[index]
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": template
            }
        return data
    
    else:  # text
        return {
            "msg_type": "text",
            "content": {"text": content}
        }


async def send_to_lark(
    content: Union[str, Dict[str, Any]],
    msg_type: str = "text",
    title: Optional[str] = None,
    template: str = "blue",
    fields: Optional[List[Union[str, tuple]]] = None,
    is_raw: bool = False,
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    发送消息到 Lark（支持重试、超时控制）
    
    Args:
        content: 消息内容
            - 如果 is_raw=True: 必须是完整的 Lark 消息结构
            - 如果 is_raw=False: 字符串内容
        msg_type: 消息类型，"text"、"interactive"、"markdown"（仅当 is_raw=False 时有效）
        title: 卡片标题（仅 interactive/markdown 模式有效）
        template: 卡片模板颜色，"blue", "red", "green", "yellow" 等
        fields: 额外的字段列表，每个元素可以是：
            - 字符串：直接作为字段内容，is_short 默认 False
            - 元组 (内容, is_short)：指定 is_short
        is_raw: 如果为 True，content 已经是完整的数据结构，直接发送
        max_retries: 最大重试次数（覆盖配置文件）
        base_delay: 基础延迟时间（覆盖配置文件）
        max_delay: 最大延迟时间（覆盖配置文件）
        timeout: 请求超时时间（覆盖配置文件）
    
    Returns:
        发送结果字典
            {
                "status": "success" | "error",
                "message": "描述信息",
                "error": "错误信息"  # 仅当 status=error 时存在
            }
    
    Example:
        # 发送文本消息
        result = await send_to_lark("Hello, Lark!")
        
        # 发送卡片消息
        result = await send_to_lark(
            content="这是一条重要通知",
            msg_type="interactive",
            title="系统告警",
            template="red",
            fields=["字段1", "字段2"]
        )
        
        # 发送原始消息
        raw_data = {"msg_type": "text", "content": {"text": "raw message"}}
        result = await send_to_lark(raw_data, is_raw=True)
    """
    webhook_url = _get_webhook()
    if not webhook_url:
        logger.error("Lark Webhook 未配置，无法发送消息")
        return {"status": "error", "error": "webhook not configured"}
    
    # 获取重试配置（参数优先，否则使用配置文件）
    retry_config = _get_retry_config()
    max_retries = max_retries if max_retries is not None else retry_config['max_retries']
    base_delay = base_delay if base_delay is not None else retry_config['base_delay']
    max_delay = max_delay if max_delay is not None else retry_config['max_delay']
    timeout = timeout if timeout is not None else retry_config.get('timeout', 10)
    
    # 构建消息数据
    data = _build_message_data(content, msg_type, title, template, fields, is_raw)
    
    # 带重试的发送
    last_error = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            result = await _send_request(data, webhook_url, timeout)
            
            if result.get('status') == 'success':
                if attempt > 0:
                    logger.info(f"Lark消息发送成功 (重试 {attempt} 次后成功)")
                else:
                    logger.debug("Lark消息发送成功")
                return result
            else:
                # 检查是否应该重试
                if result.get('should_retry', False) and attempt < max_retries:
                    # 可重试的错误，继续重试
                    logger.warning(
                        f"Lark请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {result.get('error')}. "
                        f"等待 {delay:.2f}秒后重试..."
                    )
                    await asyncio.sleep(delay)
                    # 指数退避，但不超过最大延迟
                    delay = min(delay * 2, max_delay)
                    continue
                else:
                    # 不可重试的错误或已达最大重试次数
                    return result
                    
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    f"Lark请求异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}. "
                    f"等待 {delay:.2f}秒后重试..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"Lark请求失败，已重试 {max_retries} 次: {e}")
    
    return {"status": "error", "error": str(last_error) if last_error else "unknown error"}


def create_field(label: str, value: str, is_short: bool = True) -> Dict[str, Any]:
    """
    创建字段（用于卡片消息）
    
    Args:
        label: 字段标签
        value: 字段值（支持 Markdown）
        is_short: 是否短字段（一行两个）
    
    Returns:
        字段字典，可直接用于 fields 参数
    
    Example:
        fields = [
            create_field("价格", "$50,000"),
            create_field("变化", "+5.2%", is_short=True),
        ]
        await send_to_lark("市场数据", msg_type="interactive", fields=fields)
    """
    return {
        "is_short": is_short,
        "text": {
            "tag": "lark_md",
            "content": f"**{label}**\n{value}"
        }
    }


# 便捷函数
async def send_text(content: str) -> Dict[str, Any]:
    """发送纯文本消息"""
    return await send_to_lark(content, msg_type="text")


async def send_markdown(content: str, title: Optional[str] = None) -> Dict[str, Any]:
    """发送 Markdown 消息"""
    return await send_to_lark(content, msg_type="markdown", title=title)


async def send_card(
    content: str,
    title: str,
    template: str = "blue",
    fields: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """发送卡片消息"""
    return await send_to_lark(
        content=content,
        msg_type="interactive",
        title=title,
        template=template,
        fields=fields
    )


# 测试代码
if __name__ == "__main__":
    async def test():
        # 测试发送文本
        result = await send_text("Hello, Lark!")
        print(f"文本消息: {result}")
        
        # 测试发送卡片
        fields = [
            create_field("BTC", "$50,000"),
            create_field("ETH", "$3,000"),
        ]
        result = await send_card(
            content="加密货币市场更新",
            title="市场数据",
            template="blue",
            fields=fields
        )
        print(f"卡片消息: {result}")
    
    asyncio.run(test())