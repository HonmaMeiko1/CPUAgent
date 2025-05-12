import torch
import numpy as np
from typing import Dict, Any, Tuple

class BaseAgent:
    """智能体基类"""
    def __init__(self, device: torch.device):
        self.device = device
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """选择动作（需要被子类实现）"""
        raise NotImplementedError
        
    def train(self, *args, **kwargs) -> Dict[str, float]:
        """训练智能体（需要被子类实现）"""
        raise NotImplementedError
        
    def save(self, path: str):
        """保存智能体（需要被子类实现）"""
        raise NotImplementedError
        
    def load(self, path: str):
        """加载智能体（需要被子类实现）"""
        raise NotImplementedError 