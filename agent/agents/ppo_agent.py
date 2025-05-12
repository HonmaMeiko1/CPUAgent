import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List
from .base_agent import BaseAgent
from ..networks.actor_critic_network import ActorCriticNetwork
from ..trainers.ppo_trainer import PPOTrainer

class PPOAgent(BaseAgent):
    """PPO算法智能体"""
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 learning_rate: float = 3e-4,
                 device: str = 'auto',
                 **kwargs):
        # 设备选择逻辑
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        super().__init__(self.device)
        
        # 创建网络
        self.network = ActorCriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate
        )
        
        # 创建训练器
        self.trainer = PPOTrainer(
            network=self.network,
            optimizer=self.optimizer,
            device=self.device,
            **kwargs
        )
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """选择动作
        
        Args:
            state: 环境状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的动作
            info: 额外信息
        """
        # 转换状态为张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 获取动作和价值
        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(state_tensor, deterministic)
            
        # 转换为numpy数组
        action_np = action.cpu().numpy()[0]
        
        info = {
            'log_prob': log_prob.cpu().numpy()[0],
            'value': value.cpu().numpy()[0]
        }
        
        return action_np, info
        
    def train(self, states: List[np.ndarray], actions: List[np.ndarray],
              rewards: List[float], next_state: np.ndarray) -> Dict[str, float]:
        """训练智能体
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            next_state: 下一个状态
            
        Returns:
            训练指标字典
        """
        return self.trainer.process_rollout(states, actions, rewards, next_state)
        
    def save(self, path: str):
        """保存智能体"""
        self.trainer.save_checkpoint(path)
        
    def load(self, path: str):
        """加载智能体"""
        self.trainer.load_checkpoint(path) 