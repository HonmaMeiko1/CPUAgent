import wmi
import logging
from typing import Dict
import ctypes
from ctypes import wintypes

class MemoryControl:
    def __init__(self):
        """
        初始化内存控制器
        """
        self.wmi = wmi.WMI()
        self.logger = logging.getLogger(__name__)
        self._init_memory_management()
        
    def _init_memory_management(self):
        """
        初始化内存管理
        """
        try:
            # 获取管理员权限
            if not ctypes.windll.shell32.IsUserAnAdmin():
                self.logger.warning("需要管理员权限来控制内存频率")
                return
                
            # 获取内存信息
            self.memory_info = {}
            for memory in self.wmi.Win32_PhysicalMemory():
                self.memory_info[memory.DeviceLocator] = {
                    'capacity': int(memory.Capacity),
                    'speed': int(memory.Speed),
                    'manufacturer': memory.Manufacturer,
                    'serial_number': memory.SerialNumber
                }
                
        except Exception as e:
            self.logger.error(f"初始化内存管理失败: {str(e)}")
            
    def get_memory_info(self) -> Dict:
        """
        获取内存信息
        
        Returns:
            Dict: 内存信息
        """
        return self.memory_info
        
    def set_memory_power_mode(self, mode: str) -> bool:
        """
        设置内存电源模式
        
        Args:
            mode: 电源模式 ('high_performance', 'balanced', 'power_save')
            
        Returns:
            bool: 是否设置成功
        """
        try:
            # 使用powercfg命令设置内存电源模式
            import subprocess
            
            if mode == 'high_performance':
                subprocess.run([
                    'powercfg', '/change', 'memory-power-settings-ac', '0',
                    'powercfg', '/change', 'memory-power-settings-dc', '0'
                ], check=True)
            elif mode == 'balanced':
                subprocess.run([
                    'powercfg', '/change', 'memory-power-settings-ac', '1',
                    'powercfg', '/change', 'memory-power-settings-dc', '1'
                ], check=True)
            elif mode == 'power_save':
                subprocess.run([
                    'powercfg', '/change', 'memory-power-settings-ac', '2',
                    'powercfg', '/change', 'memory-power-settings-dc', '2'
                ], check=True)
            else:
                self.logger.error(f"不支持的内存电源模式: {mode}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"设置内存电源模式失败: {str(e)}")
            return False
            
    def get_current_memory_state(self) -> Dict:
        """
        获取当前内存状态
        
        Returns:
            Dict: 内存状态信息
        """
        try:
            memory_state = {}
            for memory in self.wmi.Win32_PhysicalMemory():
                memory_state[memory.DeviceLocator] = {
                    'capacity': int(memory.Capacity),
                    'speed': int(memory.Speed),
                    'manufacturer': memory.Manufacturer,
                    'serial_number': memory.SerialNumber,
                    'configured_voltage': memory.ConfiguredVoltage,
                    'configured_clock_speed': memory.ConfiguredClockSpeed
                }
            return memory_state
            
        except Exception as e:
            self.logger.error(f"获取内存状态失败: {str(e)}")
            return {}
            
    def optimize_memory_usage(self) -> bool:
        """
        优化内存使用
        
        Returns:
            bool: 是否优化成功
        """
        try:
            # 使用powercfg命令优化内存使用
            import subprocess
            subprocess.run([
                'powercfg', '/change', 'memory-power-settings-ac', '1',
                'powercfg', '/change', 'memory-power-settings-dc', '1',
                'powercfg', '/change', 'memory-throttle-ac', '0',
                'powercfg', '/change', 'memory-throttle-dc', '0'
            ], check=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"优化内存使用失败: {str(e)}")
            return False 