import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np
from .base_trainer import BaseTrainer
from ..networks.actor_critic_network import ActorCriticNetwork
from ..utils.gae_calculator import GAECalculator

class PPOTrainer(BaseTrainer):
    """PPO算法训练器"""
    def __init__(self, 
                 network: ActorCriticNetwork,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        super().__init__(network, optimizer, device)
        
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_calculator = GAECalculator(gamma, gae_lambda)
        
    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                  old_log_probs: torch.Tensor, advantages: torch.Tensor,
                  returns: torch.Tensor) -> Dict[str, float]:
        """执行一步PPO训练
        
        Args:
            states: 状态批次
            actions: 动作批次
            old_log_probs: 旧策略下的动作对数概率
            advantages: 优势值
            returns: 回报值
            
        Returns:
            训练指标字典
        """
        # 计算新的动作概率和价值
        mean, log_std, values = self.network(states)
        std = log_std.exp()
        new_log_probs = self.network.actor._compute_log_prob(mean, std, actions)
        
        # 计算策略比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算PPO目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 计算熵正则化
        entropy = -torch.mean(log_std + 0.5 * torch.log(2 * torch.pi * torch.exp(2 * log_std)))
        
        # 计算总损失
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }
        
    def process_rollout(self, states: List[np.ndarray], actions: List[np.ndarray],
                       rewards: List[float], next_state: np.ndarray) -> Dict[str, float]:
        """处理一个回合的数据
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            next_state: 下一个状态
            
        Returns:
            训练指标字典
        """
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # 计算价值
        with torch.no_grad():
            _, _, values = self.network(states)
            _, _, next_value = self.network(torch.FloatTensor(next_state).to(self.device))
            
        # 计算优势
        advantages, returns = self.gae_calculator.compute_gae(
            rewards, values.cpu().numpy(), next_value.item())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 计算旧的动作概率
        with torch.no_grad():
            mean, log_std = self.network.actor(states)
            std = log_std.exp()
            old_log_probs = self.network.actor._compute_log_prob(mean, std, actions)
            
        # 执行训练步骤
        return self.train_step(states, actions, old_log_probs, advantages, returns) 