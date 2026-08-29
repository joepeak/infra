#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微信消息发送处理器
"""

import subprocess
import asyncio
import aiohttp
import random
import json
import re

from infra.logger import get_logger
from .retry import retry_async_decorator

logger = get_logger(__name__)


def _send_wechat_sync(message: str, timeout: int = 10, max_length: int = 2000) -> bool:
    """
    同步发送微信消息（内部使用）
    """
    if not message or not message.strip():
        logger.warning("微信消息为空，跳过发送")
        return False
    
    if len(message) > max_length:
        message = message[:max_length] + f"\n\n... [消息过长已截断]"
    
    cmd = ["hermes", "send", "--to", "weixin", message]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        
        if result.returncode == 0:
            logger.debug(f"微信消息发送成功: {len(message)} 字符")
            return True
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            logger.error(f"微信消息发送失败: {error_msg}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"微信消息发送超时 ({timeout}秒)")
        return False
    except FileNotFoundError:
        logger.error("找不到 hermes 命令，请检查是否已安装")
        return False
    except Exception as e:
        logger.error(f"微信消息发送异常: {e}")
        return False

# 从错误信息中提取冷却时间
def _extract_cooldown(error_msg: str) -> float:
    """从错误信息中提取冷却时间，若提取失败返回 None"""
    match = re.search(r'cooldown active for ([\d.]+)s', error_msg)
    if match:
        base_cooldown = float(match.group(1))
        # 加 3~10 秒随机缓冲，避免重试请求在同一个时间点集中
        extra = random.uniform(3, 10)
        total = base_cooldown + extra
        logger.info(f"⏳ 限流冷却 {base_cooldown}s，额外缓冲 {extra:.1f}s，总计等待 {total:.1f}s")
        return total
    return None

@retry_async_decorator(max_retries=5)
async def send_to_wechat(message: str, timeout: int = 180) -> bool:
    """
    发送消息到微信（纯粹 HTTP 层）

    Args:
        message: 消息内容
        timeout: 超时时间（秒）

    Returns:
        是否发送成功
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://172.17.0.1:9090/send',
                json={'message': message},
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                response_text = await resp.text()
                logger.info(f"HTTP {resp.status}: {response_text[:300]}")

                if resp.status == 200:
                    try:
                        data = json.loads(response_text)
                        if data.get('status') == 'ok':
                            return True
                        logger.warning(f"业务返回错误: {data.get('msg', 'unknown')}")
                        return False
                    except json.JSONDecodeError:
                        return True

                elif resp.status == 500:
                    try:
                        data = json.loads(response_text)
                        error_msg = data.get('msg', '')
                        logger.warning(f"服务端错误: {error_msg}")

                        cooldown = _extract_cooldown(error_msg)
                        if cooldown:
                            logger.info(f"⏳ 微信限流，等待 {cooldown:.1f} 秒后再重试...")
                            await asyncio.sleep(cooldown)
                        else:
                            await asyncio.sleep(3)

                        raise Exception(f"Server error: {error_msg}")
                    except json.JSONDecodeError:
                        logger.error(f"服务端返回非JSON: {response_text[:200]}")
                        await asyncio.sleep(3)
                        raise Exception(f"Server error: {response_text[:200]}")

                else:
                    logger.error(f"HTTP {resp.status}: {response_text[:200]}")
                    await asyncio.sleep(3)
                    raise Exception(f"HTTP {resp.status}: {response_text[:200]}")

    except asyncio.TimeoutError:
        logger.error(f"请求超时 ({timeout}秒)")
        raise Exception("Request timeout")
    except aiohttp.ClientError as e:
        logger.error(f"HTTP客户端错误: {e}")
        raise Exception(f"Client error: {e}")
    except Exception as e:
        logger.warning(f"发送失败: {e}")
        raise

# 项目运行在docker中，所以无法直接调用 hermes命令，所以需要在机机主上运行下面HTHTTP简易服务器
#
# #!/usr/bin/env python3
# import subprocess
# import json
# import sys
# from http.server import HTTPServer, BaseHTTPRequestHandler

# class WechatHandler(BaseHTTPRequestHandler):
#     def do_POST(self):
#         if self.path != '/send':
#             self.send_response(404)
#             self.end_headers()
#             return

#         content_length = int(self.headers.get('Content-Length', 0))
#         body = self.rfile.read(content_length)

#         try:
#             data = json.loads(body.decode('utf-8'))
#             message = data.get('message', '')
#             print(f"[INFO] Received message: {message[:50]}...")  # 添加日志
#         except Exception as e:
#             print(f"[ERROR] Invalid JSON: {e}")
#             self._send_json({'status': 'error', 'msg': 'invalid json'}, 400)
#             return

#         if not message:
#             self._send_json({'status': 'error', 'msg': 'empty message'}, 400)
#             return

#         # 执行 hermes send
#         print(f"[INFO] Sending to WeChat: {message[:50]}...")
#         result = subprocess.run(
#             ["hermes", "send", "--to", "weixin", message],
#             capture_output=True,
#             text=True,
#             timeout=30  # 加超时，防止卡住
#         )

#         if result.returncode == 0:
#             print("[INFO] Send success")
#             self._send_json({'status': 'ok'})
#         else:
#             error_msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
#             print(f"[ERROR] Send failed (rc={result.returncode}): {error_msg}")
#             self._send_json({'status': 'error', 'msg': error_msg}, 500)

#     def _send_json(self, data, status=200):
#         body = json.dumps(data).encode('utf-8')
#         self.send_response(status)
#         self.send_header('Content-Type', 'application/json')
#         self.send_header('Content-Length', str(len(body)))
#         self.end_headers()
#         self.wfile.write(body)

#     # 保留HTTP访问日志
#     def log_message(self, format, *args):
#         print(f"[ACCESS] {format % args}")  # 改为输出到stdout

# if __name__ == '__main__':
#     port = 8090
#     server = HTTPServer(('0.0.0.0', port), WechatHandler)
#     print(f'微信转发服务运行在 http://0.0.0.0:{port}/send')
#     server.serve_forever()