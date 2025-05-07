import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from agent.actor_critic import ActorCriticNetwork
from env.state_space import StateSpace
from env.action_space import ActionSpace
from env.reward import RewardCalculator
import logging

class PPOTrainer:
    """PPO训练器"""
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.state_space = StateSpace(config)
        self.action_space = ActionSpace(config)
        self.reward_calculator = RewardCalculator(config)
        
        self.network = ActorCriticNetwork(
            state_dim=self.state_space.state_dim,
            action_dim=self.action_space.action_dim,
            hidden_dims=config.get('network', {}).get('hidden_dims', [256, 128])
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get('training', {}).get('learning_rate', 3e-4)
        )
        
        # 训练参数
        self.gamma = config.get('training', {}).get('gamma', 0.99)
        self.gae_lambda = config.get('training', {}).get('gae_lambda', 0.95)
        self.clip_ratio = config.get('training', {}).get('clip_ratio', 0.2)
        self.value_coef = config.get('training', {}).get('value_coef', 0.5)
        self.entropy_coef = config.get('training', {}).get('entropy_coef', 0.01)
        
    def compute_gae(self, rewards: List[float], values: List[float], next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算广义优势估计"""
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
        
    def update_network(self, states: torch.Tensor, actions: torch.Tensor,
                      old_log_probs: torch.Tensor, advantages: torch.Tensor,
                      returns: torch.Tensor) -> Dict[str, float]:
        """更新网络参数"""
        # 计算新的动作概率和价值
        mean, log_std, values = self.network(states)
        std = log_std.exp()
        new_log_probs = self.network.actor._log_prob(mean, std, actions)
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算PPO目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        value_loss = nn.MSELoss()(values, returns)
        
        # 熵正则化
        entropy = -torch.mean(log_std + 0.5 * torch.log(2 * torch.pi * torch.exp(2 * log_std)))
        
        # 总损失
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
        
    def train_step(self, states: List[np.ndarray], actions: List[np.ndarray],
                  rewards: List[float], next_state: np.ndarray) -> Dict[str, float]:
        """执行一步训练"""
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        
        # 计算价值
        with torch.no_grad():
            _, _, values = self.network(states)
            _, _, next_value = self.network(torch.FloatTensor(next_state).to(self.device))
            
        # 计算优势
        advantages, returns = self.compute_gae(rewards, values.cpu().numpy(), next_value.item())
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 计算旧的动作概率
        with torch.no_grad():
            mean, log_std = self.network.actor(states)
            std = log_std.exp()
            old_log_probs = self.network.actor._log_prob(mean, std, actions)
            
        # 更新网络
        metrics = self.update_network(states, actions, old_log_probs, advantages, returns)
        
        return metrics
        
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 