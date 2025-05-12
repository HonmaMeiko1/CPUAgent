import psutil
import logging
from typing import Dict, List
import time
from pathlib import Path
import glob

class SystemMonitor:
    def __init__(self):
        """
        初始化系统监控器
        """
        self.logger = logging.getLogger(__name__)
        self._init_system_paths()
        
    def _init_system_paths(self):
        """
        初始化系统路径
        """
        # 初始化CPU和电源相关路径
        self.cpu_paths = glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq')
        self.power_paths = {
            'intel_rapl': '/sys/class/powercap/intel-rapl:0/energy_uj',
            'amd_rapl': '/sys/class/powercap/amd-rapl:0/energy_uj',
            'battery': '/sys/class/power_supply/BAT0/power_now'
        }
        
        # 初始化温度传感器路径
        self.temp_paths = []
        for i in range(10):
            path = Path(f"/sys/class/thermal/thermal_zone{i}/temp")
            if path.exists():
                self.temp_paths.append(path)
        
    def get_cpu_metrics(self) -> Dict:
        """
        获取CPU相关指标
        
        Returns:
            Dict: CPU指标，包括使用率、频率、温度等
        """
        try:
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': [],
                'cpu_temp': 0.0
            }
            
            # 获取CPU频率
            for cpu_path in self.cpu_paths:
                try:
                    with open(f"{cpu_path}/scaling_cur_freq", 'r') as f:
                        freq = float(f.read().strip()) / 1000  # 转换为MHz
                        metrics['cpu_freq'].append(freq)
                except Exception:
                    continue
                    
            # 如果无法从cpufreq获取，则使用psutil
            if not metrics['cpu_freq']:
                cpu_freq = psutil.cpu_freq(percpu=True)
                metrics['cpu_freq'] = [freq.current for freq in cpu_freq]
                
            # 获取CPU温度
            for temp_path in self.temp_paths:
                try:
                    with open(temp_path, 'r') as f:
                        temp = float(f.read().strip()) / 1000  # 转换为摄氏度
                        if temp > 0:
                            metrics['cpu_temp'] = temp
                            break
                except Exception:
                    continue
                    
            return metrics
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
            
            metrics = {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count
            }
            
            # 获取磁盘IO延迟（Linux特有）
            try:
                with open('/proc/diskstats', 'r') as f:
                    for line in f:
                        if 'sda' in line:  # 主硬盘
                            fields = line.strip().split()
                            metrics['io_time'] = int(fields[12])  # IO操作花费的毫秒数
                            metrics['weighted_io_time'] = int(fields[13])  # 加权IO时间
                            break
            except Exception:
                pass
                
            return metrics
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
            metrics = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout
            }
            
            # 获取网络接口速度（Linux特有）
            try:
                for interface in psutil.net_if_stats().keys():
                    if interface != 'lo':  # 排除回环接口
                        speed_path = Path(f"/sys/class/net/{interface}/speed")
                        if speed_path.exists():
                            with open(speed_path, 'r') as f:
                                metrics[f'{interface}_speed'] = int(f.read().strip())
            except Exception:
                pass
                
            return metrics
        except Exception as e:
            self.logger.error(f"获取网络指标失败: {str(e)}")
            return {}
            
    def get_power_metrics(self) -> Dict:
        """
        获取电源相关指标
        
        Returns:
            Dict: 电源指标，包括当前功耗等
        """
        try:
            power_data = {'power_consumption': 0.0}
            
            # 尝试从RAPL获取功耗数据
            for path in [self.power_paths['intel_rapl'], self.power_paths['amd_rapl']]:
                try:
                    with open(path, 'r') as f:
                        power_data['power_consumption'] = float(f.read().strip()) / 1_000_000  # 转换为瓦特
                        break
                except Exception:
                    continue
                    
            # 如果RAPL不可用，尝试从电池获取
            if power_data['power_consumption'] == 0.0:
                try:
                    with open(self.power_paths['battery'], 'r') as f:
                        power_data['power_consumption'] = float(f.read().strip()) / 1_000_000
                except Exception:
                    pass
                    
            # 获取电池状态（如果有）
            bat_path = Path("/sys/class/power_supply/BAT0")
            if bat_path.exists():
                try:
                    with open(bat_path / "capacity", 'r') as f:
                        power_data['battery_percent'] = int(f.read().strip())
                    with open(bat_path / "status", 'r') as f:
                        power_data['battery_status'] = f.read().strip()
                except Exception:
                    pass
                    
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