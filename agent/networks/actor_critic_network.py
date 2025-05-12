import torch
import torch.nn as nn
from typing import Tuple, List
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork

class ActorCriticNetwork(nn.Module):
    """组合的Actor-Critic网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic = CriticNetwork(state_dim, hidden_dims)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        mean, log_std = self.actor(state)
        value = self.critic(state)
        return mean, log_std, value
        
    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和价值"""
        action, log_prob = self.actor.get_action(state, deterministic)
        value = self.critic(state)
        return action, log_prob, value 