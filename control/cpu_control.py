import wmi
import logging
from typing import Dict, List
import ctypes
from ctypes import wintypes

class CPUControl:
    def __init__(self):
        """
        初始化CPU控制器
        """
        self.wmi = wmi.WMI()
        self.logger = logging.getLogger(__name__)
        self._init_power_management()
        
    def _init_power_management(self):
        """
        初始化电源管理
        """
        try:
            # 获取管理员权限
            if not ctypes.windll.shell32.IsUserAnAdmin():
                self.logger.warning("需要管理员权限来控制CPU频率")
                return
                
            # 获取电源计划
            self.power_plans = {}
            for plan in self.wmi.Win32_PowerPlan():
                self.power_plans[plan.ElementName] = plan.InstanceID
                
        except Exception as e:
            self.logger.error(f"初始化电源管理失败: {str(e)}")
            
    def get_available_power_plans(self) -> List[str]:
        """
        获取可用的电源计划
        
        Returns:
            List[str]: 电源计划名称列表
        """
        return list(self.power_plans.keys())
        
    def set_power_plan(self, plan_name: str) -> bool:
        """
        设置电源计划
        
        Args:
            plan_name: 电源计划名称
            
        Returns:
            bool: 是否设置成功
        """
        try:
            if plan_name not in self.power_plans:
                self.logger.error(f"电源计划 {plan_name} 不存在")
                return False
                
            # 使用powercfg命令设置电源计划
            import subprocess
            plan_id = self.power_plans[plan_name].split('\\')[-1]
            subprocess.run(['powercfg', '/setactive', plan_id], check=True)
            return True
            
        except Exception as e:
            self.logger.error(f"设置电源计划失败: {str(e)}")
            return False
            
    def set_cpu_frequency(self, frequency_mhz: int) -> bool:
        """
        设置CPU频率
        
        Args:
            frequency_mhz: 目标频率（MHz）
            
        Returns:
            bool: 是否设置成功
        """
        try:
            # 获取当前CPU信息
            cpu = self.wmi.Win32_Processor()[0]
            current_freq = cpu.CurrentClockSpeed
            
            if frequency_mhz > cpu.MaxClockSpeed:
                self.logger.warning(f"目标频率 {frequency_mhz}MHz 超过最大频率 {cpu.MaxClockSpeed}MHz")
                return False
                
            # 使用powercfg命令设置CPU频率
            import subprocess
            subprocess.run([
                'powercfg', '/change', 'processor-throttle-ac', '0',
                'powercfg', '/change', 'processor-throttle-dc', '0',
                'powercfg', '/change', 'processor-maximum-frequency-ac', str(frequency_mhz * 1000),
                'powercfg', '/change', 'processor-maximum-frequency-dc', str(frequency_mhz * 1000)
            ], check=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"设置CPU频率失败: {str(e)}")
            return False
            
    def get_current_cpu_state(self) -> Dict:
        """
        获取当前CPU状态
        
        Returns:
            Dict: CPU状态信息
        """
        try:
            cpu = self.wmi.Win32_Processor()[0]
            return {
                'name': cpu.Name,
                'current_frequency': cpu.CurrentClockSpeed,
                'max_frequency': cpu.MaxClockSpeed,
                'min_frequency': cpu.MinClockSpeed,
                'load_percentage': cpu.LoadPercentage,
                'number_of_cores': cpu.NumberOfCores,
                'number_of_logical_processors': cpu.NumberOfLogicalProcessors
            }
        except Exception as e:
            self.logger.error(f"获取CPU状态失败: {str(e)}")
            return {} 