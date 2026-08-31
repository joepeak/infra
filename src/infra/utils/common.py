import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional

def parse_json_string(s: Any) -> Any:
    """
    尝试将字符串解析为 JSON 格式，支持不同的数据类型。
    """
    try:
        # 尝试解析字符串为 JSON
        return json.loads(s)
    except json.JSONDecodeError:
        # 如果解析失败，说明它不是有效的 JSON 字符串
        return s  # 返回原始字符串
    
def merge_results(*functions: Callable) -> Optional[Dict[str, Any]]:
    merged_result = {}
    
    for func in functions:
        result = func()  # 调用函数并获取返回值
        if result is not None:
            merged_result.update(result)
    return merged_result if merged_result else None


def parse_datetime_with_format(dt_str: str, fmt: str = "%Y.%m.%d %H:%M:%S") -> datetime:
    return datetime.strptime(dt_str, fmt)