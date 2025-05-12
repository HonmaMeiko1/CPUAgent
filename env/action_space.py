import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
from pathlib import Path
import glob

@dataclass
class HardwareConstraints:
    """硬件约束"""
    min_freq: float  # 最小CPU频率 (MHz)
    max_freq: float  # 最大CPU频率 (MHz)
    max_temp: float  # 最大温度 (°C)
    freq_step: float  # 频率调整步长 (MHz)
    temp_threshold: float  # 温度阈值 (°C)
    cooldown_time: float  # 冷却时间 (s)
    
    @classmethod
    def from_system(cls) -> 'HardwareConstraints':
        """从系统获取硬件约束"""
        # 默认值
        constraints = {
            'min_freq': 1200.0,  # 1.2GHz
            'max_freq': 3600.0,  # 3.6GHz
            'max_temp': 85.0,    # 85°C
            'freq_step': 100.0,  # 100MHz
            'temp_threshold': 75.0,  # 75°C
            'cooldown_time': 5.0  # 5秒
        }
        
        # Linux系统适配：尝试从cpufreq获取实际值
        try:
            # 获取所有CPU核心的频率信息
            cpu_paths = glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq')
            if cpu_paths:
                cpu_path = Path(cpu_paths[0])  # 使用第一个CPU核心的信息
                
                # 读取最小频率
                with open(cpu_path / 'scaling_min_freq', 'r') as f:
                    constraints['min_freq'] = float(f.read().strip()) / 1000  # 转换为MHz
                    
                # 读取最大频率
                with open(cpu_path / 'scaling_max_freq', 'r') as f:
                    constraints['max_freq'] = float(f.read().strip()) / 1000  # 转换为MHz
                    
                # 读取可用频率列表
                with open(cpu_path / 'scaling_available_frequencies', 'r') as f:
                    freqs = sorted([float(f) / 1000 for f in f.read().strip().split()])
                    if len(freqs) > 1:
                        constraints['freq_step'] = freqs[1] - freqs[0]
                        
                # 尝试从thermal_zone获取温度阈值
                for i in range(10):
                    trip_point_path = Path(f"/sys/class/thermal/thermal_zone{i}/trip_point_0_temp")
                    if trip_point_path.exists():
                        with open(trip_point_path, 'r') as f:
                            constraints['temp_threshold'] = float(f.read().strip()) / 1000
                            constraints['max_temp'] = constraints['temp_threshold'] + 10
                            break
        except Exception:
            pass
                
        return cls(**constraints)

class ActionSpace:
    """动作空间管理"""
    def __init__(self, config: Dict):
        self.config = config
        self.constraints = self._init_constraints()
        self.action_dim = 1  # 连续动作空间，只调整CPU频率
        
    def _init_constraints(self) -> HardwareConstraints:
        """初始化硬件约束"""
        # 首先尝试从系统获取约束
        constraints = HardwareConstraints.from_system()
        
        # 使用配置覆盖默认值
        config_constraints = self.config.get('action_space', {})
        return HardwareConstraints(
            min_freq=config_constraints.get('min_freq', constraints.min_freq),
            max_freq=config_constraints.get('max_freq', constraints.max_freq),
            max_temp=config_constraints.get('max_temp', constraints.max_temp),
            freq_step=config_constraints.get('freq_step', constraints.freq_step),
            temp_threshold=config_constraints.get('temp_threshold', constraints.temp_threshold),
            cooldown_time=config_constraints.get('cooldown_time', constraints.cooldown_time)
        )
        
    def clip_action(self, action: np.ndarray, current_freq: float) -> float:
        """裁剪动作到有效范围
        
        Args:
            action: 动作值（[-1, 1]范围内的频率调整）
            current_freq: 当前频率
            
        Returns:
            float: 裁剪后的频率
        """
        # 将动作从[-1, 1]映射到频率调整范围
        freq_adjustment = action[0] * self.constraints.freq_step
        new_freq = current_freq + freq_adjustment
        
        # 确保在有效范围内
        new_freq = np.clip(
            new_freq,
            self.constraints.min_freq,
            self.constraints.max_freq
        )
        
        # 确保频率是步长的整数倍
        steps = round(new_freq / self.constraints.freq_step)
        new_freq = steps * self.constraints.freq_step
        
        return new_freq
        
    def check_constraints(self, current_freq: float, temperature: float) -> Tuple[bool, str]:
        """检查是否满足硬件约束
        
        Args:
            current_freq: 当前频率
            temperature: 当前温度
            
        Returns:
            Tuple[bool, str]: (是否满足约束, 原因)
        """
        if temperature > self.constraints.max_temp:
            return False, "温度超过最大限制"
            
        if temperature > self.constraints.temp_threshold:
            return False, "温度超过阈值，需要冷却"
            
        if current_freq < self.constraints.min_freq:
            return False, "频率低于最小限制"
            
        if current_freq > self.constraints.max_freq:
            return False, "频率超过最大限制"
            
        return True, "满足所有约束"
        
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取动作空间的边界
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (最小动作值, 最大动作值)
        """
        return (
            np.array([-1.0]),  # 最小动作值
            np.array([1.0])    # 最大动作值
        )
        
    def sample_action(self) -> np.ndarray:
        """采样随机动作
        
        Returns:
            np.ndarray: 随机动作值
        """
        return np.random.uniform(-1, 1, size=(self.action_dim,))
        
    def action_to_freq(self, action: np.ndarray, current_freq: float) -> float:
        """将动作转换为实际频率
        
        Args:
            action: 动作值
            current_freq: 当前频率
            
        Returns:
            float: 目标频率
        """
        return self.clip_action(action, current_freq)
        
    def freq_to_action(self, target_freq: float, current_freq: float) -> np.ndarray:
        """将目标频率转换为动作值
        
        Args:
            target_freq: 目标频率
            current_freq: 当前频率
            
        Returns:
            np.ndarray: 动作值
        """
        freq_diff = target_freq - current_freq
        action = freq_diff / self.constraints.freq_step
        return np.array([np.clip(action, -1, 1)]) 