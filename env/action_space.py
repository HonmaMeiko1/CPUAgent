import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

@dataclass
class HardwareConstraints:
    """硬件约束"""
    min_freq: float  # 最小CPU频率
    max_freq: float  # 最大CPU频率
    max_temp: float  # 最大温度
    freq_step: float  # 频率调整步长
    temp_threshold: float  # 温度阈值
    cooldown_time: float  # 冷却时间

class ActionSpace:
    """动作空间管理"""
    def __init__(self, config: Dict):
        self.config = config
        self.constraints = self._init_constraints()
        self.action_dim = 1  # 连续动作空间，只调整CPU频率
        
    def _init_constraints(self) -> HardwareConstraints:
        """初始化硬件约束"""
        return HardwareConstraints(
            min_freq=self.config.get('action_space', {}).get('min_freq', 1.0),
            max_freq=self.config.get('action_space', {}).get('max_freq', 3.5),
            max_temp=self.config.get('action_space', {}).get('max_temp', 85.0),
            freq_step=self.config.get('action_space', {}).get('freq_step', 0.1),
            temp_threshold=self.config.get('action_space', {}).get('temp_threshold', 75.0),
            cooldown_time=self.config.get('action_space', {}).get('cooldown_time', 5.0)
        )
        
    def clip_action(self, action: np.ndarray, current_freq: float) -> float:
        """裁剪动作到有效范围"""
        # 将动作从[-1, 1]映射到频率调整范围
        freq_adjustment = action[0] * self.constraints.freq_step
        new_freq = current_freq + freq_adjustment
        
        # 确保在有效范围内
        new_freq = np.clip(
            new_freq,
            self.constraints.min_freq,
            self.constraints.max_freq
        )
        
        return new_freq
        
    def check_constraints(self, current_freq: float, temperature: float) -> Tuple[bool, str]:
        """检查是否满足硬件约束"""
        if temperature > self.constraints.max_temp:
            return False, "Temperature exceeds maximum limit"
            
        if temperature > self.constraints.temp_threshold:
            return False, "Temperature above threshold, need cooldown"
            
        return True, "Constraints satisfied"
        
    def get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取动作空间的边界"""
        return (
            np.array([-1.0]),  # 最小动作值
            np.array([1.0])    # 最大动作值
        )
        
    def sample_action(self) -> np.ndarray:
        """采样随机动作"""
        return np.random.uniform(-1, 1, size=(self.action_dim,))
        
    def action_to_freq(self, action: np.ndarray, current_freq: float) -> float:
        """将动作转换为实际频率"""
        return self.clip_action(action, current_freq)
        
    def freq_to_action(self, target_freq: float, current_freq: float) -> np.ndarray:
        """将目标频率转换为动作值"""
        freq_diff = target_freq - current_freq
        return np.array([freq_diff / self.constraints.freq_step]) 