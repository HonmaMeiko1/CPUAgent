import os
import logging
import subprocess
from typing import Dict, List
from pathlib import Path

class CPUControl:
    """Linux CPU控制器"""
    
    def __init__(self):
        """初始化CPU控制器"""
        self.logger = logging.getLogger(__name__)
        self._init_cpu_management()
        
    def _init_cpu_management(self):
        """初始化CPU管理"""
        try:
            # 检查是否有root权限
            if os.geteuid() != 0:
                self.logger.warning("需要root权限来控制CPU频率")
                return
                
            # 检查cpufreq系统是否可用
            self.cpu_dirs = list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*'))
            if not self.cpu_dirs:
                raise RuntimeError("未找到CPU频率控制接口")
                
            # 获取可用的频率调节器
            self.available_governors = self._get_available_governors()
            
        except Exception as e:
            self.logger.error(f"初始化CPU管理失败: {str(e)}")
            
    def _get_available_governors(self) -> List[str]:
        """获取可用的CPU频率调节器"""
        try:
            governor_path = self.cpu_dirs[0] / 'cpufreq/scaling_available_governors'
            with open(governor_path, 'r') as f:
                return f.read().strip().split()
        except Exception:
            return []
            
    def get_available_power_plans(self) -> List[str]:
        """获取可用的电源计划（频率调节器）
        
        Returns:
            List[str]: 可用的频率调节器列表
        """
        return self.available_governors
        
    def set_power_plan(self, governor: str) -> bool:
        """设置CPU频率调节器
        
        Args:
            governor: 频率调节器名称（如 'performance', 'powersave', 'ondemand'）
            
        Returns:
            bool: 是否设置成功
        """
        try:
            if governor not in self.available_governors:
                self.logger.error(f"不支持的频率调节器: {governor}")
                return False
                
            for cpu_dir in self.cpu_dirs:
                governor_path = cpu_dir / 'cpufreq/scaling_governor'
                with open(governor_path, 'w') as f:
                    f.write(governor)
            return True
            
        except Exception as e:
            self.logger.error(f"设置频率调节器失败: {str(e)}")
            return False
            
    def set_cpu_frequency(self, frequency_mhz: int) -> bool:
        """设置CPU频率
        
        Args:
            frequency_mhz: 目标频率（MHz）
            
        Returns:
            bool: 是否设置成功
        """
        try:
            frequency_khz = frequency_mhz * 1000
            
            for cpu_dir in self.cpu_dirs:
                # 检查最大频率
                max_freq_path = cpu_dir / 'cpufreq/cpuinfo_max_freq'
                with open(max_freq_path, 'r') as f:
                    max_freq = int(f.read().strip())
                    
                if frequency_khz > max_freq:
                    self.logger.warning(f"目标频率 {frequency_mhz}MHz 超过最大频率 {max_freq/1000}MHz")
                    return False
                    
                # 设置最大和最小频率
                scaling_max_path = cpu_dir / 'cpufreq/scaling_max_freq'
                scaling_min_path = cpu_dir / 'cpufreq/scaling_min_freq'
                
                with open(scaling_max_path, 'w') as f:
                    f.write(str(frequency_khz))
                with open(scaling_min_path, 'w') as f:
                    f.write(str(frequency_khz))
                    
            return True
            
        except Exception as e:
            self.logger.error(f"设置CPU频率失败: {str(e)}")
            return False
            
    def get_current_cpu_state(self) -> Dict:
        """获取当前CPU状态
        
        Returns:
            Dict: CPU状态信息
        """
        try:
            # 使用lscpu获取CPU信息
            lscpu_output = subprocess.check_output(['lscpu']).decode()
            cpu_info = {}
            
            for line in lscpu_output.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    cpu_info[key.strip()] = value.strip()
                    
            # 获取CPU负载
            with open('/proc/loadavg', 'r') as f:
                load = f.read().split()
                
            # 获取当前频率
            cpu0_freq_path = self.cpu_dirs[0] / 'cpufreq/scaling_cur_freq'
            with open(cpu0_freq_path, 'r') as f:
                current_freq = int(f.read().strip()) // 1000  # 转换为MHz
                
            return {
                'name': cpu_info.get('Model name', 'Unknown'),
                'current_frequency': current_freq,
                'max_frequency': int(cpu_info.get('CPU max MHz', '0').split('.')[0]),
                'min_frequency': int(cpu_info.get('CPU min MHz', '0').split('.')[0]),
                'load_percentage': float(load[0]) * 100,
                'number_of_cores': int(cpu_info.get('CPU(s)', '1')),
                'number_of_logical_processors': int(cpu_info.get('Thread(s) per core', '1')) * int(cpu_info.get('Core(s) per socket', '1'))
            }
            
        except Exception as e:
            self.logger.error(f"获取CPU状态失败: {str(e)}")
            return {} 