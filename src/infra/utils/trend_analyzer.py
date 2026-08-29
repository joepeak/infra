#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
趋势分析工具
"""

from typing import List, Dict, Literal
import numpy as np


def analyze_trend(
    data: List[Dict[str, any]], 
    window: int = 10,
    threshold: float = 0.01
) -> Dict[str, any]:
    """
    简单趋势分析
    
    Args:
        data: 时间序列数据 [{'date': '2026-07-01', 'value': 100}, ...]
        window: 分析窗口大小（最近N个点）
        threshold: 趋势判断阈值（斜率绝对值大于此值才算趋势）
    
    Returns:
        {
            'trend': 'up' | 'down' | 'sideways',
            'slope': 斜率值,
            'strength': 'strong' | 'moderate' | 'weak',
            'window': 使用的窗口大小
        }
    """
    if len(data) < 2:
        return {'trend': 'sideways', 'slope': 0, 'strength': 'weak', 'window': 0}
    
    # 取最近 window 个点
    recent_data = data[-window:] if len(data) >= window else data
    
    # 提取 x (时间索引) 和 y (值)
    x = np.arange(len(recent_data))
    y = np.array([d['value'] for d in recent_data])
    
    # 线性回归计算斜率
    slope, intercept = np.polyfit(x, y, 1)
    
    # 判断趋势
    if slope > threshold:
        trend = 'up'
    elif slope < -threshold:
        trend = 'down'
    else:
        trend = 'sideways'
    
    # 判断强度（基于斜率绝对值）
    abs_slope = abs(slope)
    if abs_slope > threshold * 3:
        strength = 'strong'
    elif abs_slope > threshold:
        strength = 'moderate'
    else:
        strength = 'weak'
    
    return {
        'trend': trend,
        'slope': float(slope),
        'strength': strength,
        'window': len(recent_data)
    }
