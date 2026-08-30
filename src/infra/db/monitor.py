#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库监控模块
提供数据库性能监控、连接池监控和健康检查
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from infra.db.connection_manager import get_db_manager_or_raise
from infra.logger import get_logger
from infra.exceptions import DatabaseError


@dataclass
class DatabaseMetrics:
    """数据库指标"""
    timestamp: float
    connection_pool_size: int
    connections_checked_in: int
    connections_checked_out: int
    connection_pool_overflow: int
    connection_pool_invalid: int
    query_count: int
    error_count: int
    avg_response_time: float
    status: str


class DatabaseMonitor:
    """数据库监控器"""
    
    def __init__(self, check_interval: int = 60):
        self.logger = get_logger(__name__)
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: list[DatabaseMetrics] = []
        self.max_history_size = 1000
        self.query_count = 0
        self.error_count = 0
        self.response_times: list[float] = []
        self.alert_callbacks: list[Callable] = []
        self.last_health_check = 0
        
    def start_monitoring(self) -> None:
        """开始监控"""
        if self.is_running:
            self.logger.warning("数据库监控已在运行中")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("数据库监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("数据库监控已停止")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.is_running:
            try:
                self._collect_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"数据库监控循环出错: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> None:
        """收集指标"""
        try:
            manager = get_db_manager_or_raise("db")
            if manager is None:
                self.logger.error("数据库管理器未初始化")
                return
                
            stats = manager.get_connection_stats()
            if 'error' in stats:
                self.logger.error(f"获取连接池统计失败: {stats['error']}")
                return
            
            # 确保所有数值字段都是数字类型
            connection_pool_invalid = stats.get('invalid', 0)
            if callable(connection_pool_invalid):
                connection_pool_invalid = connection_pool_invalid()
            elif not isinstance(connection_pool_invalid, (int, float)):
                try:
                    connection_pool_invalid = int(connection_pool_invalid) if connection_pool_invalid else 0
                except (ValueError, TypeError):
                    connection_pool_invalid = 0
            
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            metrics = DatabaseMetrics(
                timestamp=time.time(),
                connection_pool_size=stats.get('pool_size', 0),
                connections_checked_in=stats.get('checked_in', 0),
                connections_checked_out=stats.get('checked_out', 0),
                connection_pool_overflow=stats.get('overflow', 0),
                connection_pool_invalid=connection_pool_invalid,
                query_count=self.query_count,
                error_count=self.error_count,
                avg_response_time=avg_response_time,
                status='healthy'
            )
            
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            self._check_alerts(metrics)
            
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]
            
        except Exception as e:
            self.logger.error(f"收集数据库指标失败: {e}")
            self.error_count += 1
    
    def _check_alerts(self, metrics: DatabaseMetrics) -> None:
        """检查告警条件"""
        alerts = []
        
        if metrics.connection_pool_size > 0:
            usage_rate = metrics.connections_checked_out / metrics.connection_pool_size
            if usage_rate > 0.8:
                alerts.append(f"连接池使用率过高: {usage_rate:.1%}")
        
        if metrics.connection_pool_overflow > 0:
            alerts.append(f"连接池溢出: {metrics.connection_pool_overflow} 个连接")
        
        if metrics.connection_pool_invalid > 5:
            alerts.append(f"无效连接过多: {metrics.connection_pool_invalid} 个")
        
        if metrics.query_count > 0:
            error_rate = metrics.error_count / metrics.query_count
            if error_rate > 0.05:
                alerts.append(f"错误率过高: {error_rate:.1%}")
        
        if metrics.avg_response_time > 1.0:
            alerts.append(f"平均响应时间过长: {metrics.avg_response_time:.2f}s")
        
        if alerts:
            alert_message = f"数据库告警: {'; '.join(alerts)}"
            self.logger.warning(alert_message)
            for callback in self.alert_callbacks:
                try:
                    callback(alert_message, metrics)
                except Exception as e:
                    self.logger.error(f"告警回调执行失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, DatabaseMetrics], None]) -> None:
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def record_query(self, response_time: float, success: bool = True) -> None:
        """记录查询执行（同步方法）"""
        self.query_count += 1
        self.response_times.append(response_time)
        if not success:
            self.error_count += 1
    
    def get_current_metrics(self) -> Optional[DatabaseMetrics]:
        """获取当前指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.metrics_history:
            return {'error': '没有可用的指标数据'}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': f'最近 {minutes} 分钟内没有指标数据'}
        
        pool_sizes = [m.connection_pool_size for m in recent_metrics]
        response_times = [m.avg_response_time for m in recent_metrics]
        
        return {
            'time_range_minutes': minutes,
            'sample_count': len(recent_metrics),
            'connection_pool': {
                'avg_size': sum(pool_sizes) / len(pool_sizes),
                'max_size': max(pool_sizes),
                'min_size': min(pool_sizes),
                'current_size': pool_sizes[-1] if pool_sizes else 0
            },
            'performance': {
                'avg_response_time': sum(response_times) / len(response_times),
                'max_response_time': max(response_times),
                'min_response_time': min(response_times),
                'current_response_time': response_times[-1] if response_times else 0
            },
            'queries': {
                'total_count': self.query_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / self.query_count if self.query_count > 0 else 0
            },
            'status': 'healthy' if all(m.status == 'healthy' for m in recent_metrics) else 'unhealthy'
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            manager = get_db_manager_or_raise("db")
            if manager is None:
                return {
                    'overall_status': 'unhealthy',
                    'error': '数据库管理器未初始化',
                    'alerts': ['管理器未初始化']
                }
            
            # 注意：health_check 现在是异步的，但在监控线程中不能直接调用
            # 使用同步的 get_connection_stats 代替
            stats = manager.get_connection_stats()
            if 'error' in stats:
                return {
                    'overall_status': 'unhealthy',
                    'error': stats['error'],
                    'alerts': ['获取连接池统计失败']
                }
            
            current_metrics = self.get_current_metrics()

            status: Dict[str, Any] = {
                'overall_status': 'healthy',
                'connection_stats': stats,
                'monitoring_metrics': {
                    'is_monitoring': self.is_running,
                    'current_metrics': current_metrics.__dict__ if current_metrics else None,
                    'metrics_count': len(self.metrics_history)
                },
                'alerts': []
            }

            if stats.get('pool_size', 0) == 0:
                status['overall_status'] = 'unhealthy'
                status['alerts'].append("连接池大小为0")
            
            if current_metrics and current_metrics.status != 'healthy':
                status['overall_status'] = 'unhealthy'
                status['alerts'].append("数据库指标异常")
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取健康状态失败: {e}")
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'alerts': ['健康状态检查失败']
            }


# 全局数据库监控器实例
db_monitor = DatabaseMonitor()


# 便捷函数
def start_db_monitoring():
    """启动数据库监控"""
    db_monitor.start_monitoring()


def stop_db_monitoring():
    """停止数据库监控"""
    db_monitor.stop_monitoring()


def get_db_metrics():
    """获取数据库指标"""
    return db_monitor.get_current_metrics()


def get_db_health_status():
    """获取数据库健康状态"""
    return db_monitor.get_health_status()


def record_db_query(response_time: float, success: bool = True):
    """记录数据库查询"""
    db_monitor.record_query(response_time, success)


def add_db_alert_callback(callback):
    """添加数据库告警回调"""
    db_monitor.add_alert_callback(callback)