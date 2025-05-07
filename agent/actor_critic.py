import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np

class ActorNetwork(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        self.layers = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
            
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.layers(state)
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
        log_prob = self._log_prob(mean, std, action)
        
        return action, log_prob
        
    def _log_prob(self, mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算动作的对数概率"""
        return -0.5 * (((action - mean) / std) ** 2 + 2 * torch.log(std) + np.log(2 * np.pi))

class CriticNetwork(nn.Module):
    """Critic网络"""
    def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        
        # 构建网络层
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        self.layers = nn.Sequential(*layers)
        self.value_layer = nn.Linear(prev_dim, 1)
        
        # 初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
            
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.layers(state)
        value = self.value_layer(features)
        return value

class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        mean, log_std = self.actor(state)
        value = self.critic(state)
        return mean, log_std, value
        
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和价值"""
        mean, log_std = self.actor(state)
        value = self.critic(state)
        
        if deterministic:
            return mean, torch.zeros_like(mean), value
            
        # 重参数化采样
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + noise * std
        log_prob = self.actor._log_prob(mean, std, action)
        
        return action, log_prob, value 