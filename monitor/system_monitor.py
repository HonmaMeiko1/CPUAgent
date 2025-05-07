import psutil
import wmi
import logging
from typing import Dict, List
import time

class SystemMonitor:
    def __init__(self):
        """
        初始化系统监控器
        """
        self.wmi = wmi.WMI()
        self.logger = logging.getLogger(__name__)
        
    def get_cpu_metrics(self) -> Dict:
        """
        获取CPU相关指标
        
        Returns:
            Dict: CPU指标，包括使用率、频率等
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq(percpu=True)
            cpu_count = psutil.cpu_count()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_freq': [freq.current for freq in cpu_freq],
                'cpu_count': cpu_count
            }
        except Exception as e:
            self.logger.error(f"获取CPU指标失败: {str(e)}")
            return {}
            
    def get_memory_metrics(self) -> Dict:
        """
        获取内存相关指标
        
        Returns:
            Dict: 内存指标，包括使用率、可用内存等
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_free': swap.free,
                'swap_percent': swap.percent
            }
        except Exception as e:
            self.logger.error(f"获取内存指标失败: {str(e)}")
            return {}
            
    def get_disk_metrics(self) -> Dict:
        """
        获取磁盘相关指标
        
        Returns:
            Dict: 磁盘指标，包括使用率、IO等
        """
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
        except Exception as e:
            self.logger.error(f"获取磁盘指标失败: {str(e)}")
            return {}
            
    def get_network_metrics(self) -> Dict:
        """
        获取网络相关指标
        
        Returns:
            Dict: 网络指标，包括带宽使用率等
        """
        try:
            net_io = psutil.net_io_counters()
            
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            }
        except Exception as e:
            self.logger.error(f"获取网络指标失败: {str(e)}")
            return {}
            
    def get_power_metrics(self) -> Dict:
        """
        获取电源相关指标（需要管理员权限）
        
        Returns:
            Dict: 电源指标，包括当前功耗等
        """
        try:
            power_data = {}
            for battery in self.wmi.Win32_Battery():
                power_data['battery_status'] = battery.BatteryStatus
                power_data['estimated_charge_remaining'] = battery.EstimatedChargeRemaining
                power_data['estimated_run_time'] = battery.EstimatedRunTime
                
            return power_data
        except Exception as e:
            self.logger.error(f"获取电源指标失败: {str(e)}")
            return {}
            
    def get_all_metrics(self) -> Dict:
        """
        获取所有系统指标
        
        Returns:
            Dict: 所有系统指标的综合数据
        """
        return {
            'cpu': self.get_cpu_metrics(),
            'memory': self.get_memory_metrics(),
            'disk': self.get_disk_metrics(),
            'network': self.get_network_metrics(),
            'power': self.get_power_metrics(),
            'timestamp': time.time()
        } 