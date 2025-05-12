import torch
import torch.nn as nn
from typing import Tuple, List
import numpy as np
from .base_network import BaseNetwork

class ActorNetwork(BaseNetwork):
    """Actor网络，用于生成动作策略"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        # 构建特征提取层
        self.feature_net = self._build_mlp(state_dim, hidden_dims[-1], hidden_dims[:-1])
        
        # 输出层
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # 初始化
        self.apply(self._init_weights)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            return mean, torch.zeros_like(mean)
            
        # 重参数化采样
        noise = torch.randn_like(mean)
        action = mean + noise * std
        log_prob = self._compute_log_prob(mean, std, action)
        
        return action, log_prob
        
    def _compute_log_prob(self, mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        return -0.5 * (((action - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi)) 