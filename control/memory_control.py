import os
import logging
import subprocess
from typing import Dict
from pathlib import Path

class MemoryControl:
    """Linux内存控制器"""
    
    def __init__(self):
        """初始化内存控制器"""
        self.logger = logging.getLogger(__name__)
        self._init_memory_management()
        
    def _init_memory_management(self):
        """初始化内存管理"""
        try:
            # 检查是否有root权限
            if os.geteuid() != 0:
                self.logger.warning("需要root权限来控制内存")
                return
                
            # 检查/proc/sys/vm目录是否可访问
            self.vm_path = Path('/proc/sys/vm')
            if not self.vm_path.exists():
                raise RuntimeError("未找到内存管理接口")
                
        except Exception as e:
            self.logger.error(f"初始化内存管理失败: {str(e)}")
            
    def get_memory_info(self) -> Dict:
        """获取内存信息
        
        Returns:
            Dict: 内存信息
        """
        try:
            memory_info = {}
            
            # 使用dmidecode获取物理内存信息
            if os.geteuid() == 0:  # 需要root权限
                dmi_output = subprocess.check_output(['dmidecode', '--type', 'memory']).decode()
                current_device = None
                
                for line in dmi_output.split('\n'):
                    line = line.strip()
                    if line.startswith('Memory Device'):
                        current_device = {}
                    elif current_device is not None:
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            current_device[key.strip()] = value.strip()
                            if key == 'Bank Locator':
                                memory_info[value.strip()] = current_device
                                
            # 获取/proc/meminfo信息
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        memory_info[key.strip()] = value.strip()
                        
            return memory_info
            
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {str(e)}")
            return {}
            
    def set_memory_power_mode(self, mode: str) -> bool:
        """设置内存电源模式
        
        Args:
            mode: 电源模式 ('high_performance', 'balanced', 'power_save')
            
        Returns:
            bool: 是否设置成功
        """
        try:
            # 在Linux中，我们通过调整内存管理参数来实现类似的效果
            if mode == 'high_performance':
                # 减少内存回收的激进程度
                self._write_vm_parameter('swappiness', '10')
                self._write_vm_parameter('vfs_cache_pressure', '50')
                self._write_vm_parameter('dirty_ratio', '60')
            elif mode == 'balanced':
                # 使用默认值
                self._write_vm_parameter('swappiness', '60')
                self._write_vm_parameter('vfs_cache_pressure', '100')
                self._write_vm_parameter('dirty_ratio', '40')
            elif mode == 'power_save':
                # 增加内存回收的激进程度
                self._write_vm_parameter('swappiness', '100')
                self._write_vm_parameter('vfs_cache_pressure', '150')
                self._write_vm_parameter('dirty_ratio', '20')
            else:
                self.logger.error(f"不支持的内存电源模式: {mode}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"设置内存电源模式失败: {str(e)}")
            return False
            
    def _write_vm_parameter(self, parameter: str, value: str):
        """写入VM参数"""
        param_path = self.vm_path / parameter
        if param_path.exists():
            with open(param_path, 'w') as f:
                f.write(value)
                
    def get_current_memory_state(self) -> Dict:
        """获取当前内存状态
        
        Returns:
            Dict: 内存状态信息
        """
        try:
            memory_state = {}
            
            # 读取/proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        memory_state[key.strip()] = value.strip()
                        
            # 读取当前VM参数
            vm_params = ['swappiness', 'vfs_cache_pressure', 'dirty_ratio']
            for param in vm_params:
                param_path = self.vm_path / param
                if param_path.exists():
                    with open(param_path, 'r') as f:
                        memory_state[f'vm_{param}'] = f.read().strip()
                        
            return memory_state
            
        except Exception as e:
            self.logger.error(f"获取内存状态失败: {str(e)}")
            return {}
            
    def optimize_memory_usage(self) -> bool:
        """优化内存使用
        
        Returns:
            bool: 是否优化成功
        """
        try:
            # 设置合理的内存管理参数
            self._write_vm_parameter('swappiness', '60')
            self._write_vm_parameter('vfs_cache_pressure', '100')
            self._write_vm_parameter('dirty_ratio', '40')
            self._write_vm_parameter('dirty_background_ratio', '10')
            self._write_vm_parameter('min_free_kbytes', '65536')  # 64MB
            
            # 清理缓存（需要root权限）
            if os.geteuid() == 0:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1')
                    
            return True
            
        except Exception as e:
            self.logger.error(f"优化内存使用失败: {str(e)}")
            return False 