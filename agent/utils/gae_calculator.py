import numpy as np
from typing import List, Tuple

class GAECalculator:
    """广义优势估计(GAE)计算器"""
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
    def compute_gae(self, rewards: List[float], values: List[float], next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算广义优势估计
        
        Args:
            rewards: 奖励序列
            values: 状态价值序列
            next_value: 下一个状态的价值
            
        Returns:
            advantages: 优势值
            returns: 回报值
        """
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return np.array(advantages), np.array(returns) 