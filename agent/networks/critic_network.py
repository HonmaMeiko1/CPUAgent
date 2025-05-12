import torch
import torch.nn as nn
from typing import List
from .base_network import BaseNetwork

class CriticNetwork(BaseNetwork):
    """Critic网络，用于评估状态价值"""
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        # 构建网络层
        self.value_net = self._build_mlp(state_dim, 1, hidden_dims)
        
        # 初始化
        self.apply(self._init_weights)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.value_net(state) 