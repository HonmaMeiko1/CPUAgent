import torch
import torch.nn as nn
from typing import Dict
import os
from pathlib import Path

class BaseTrainer:
    """训练器基类"""
    def __init__(self, network: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
        self.network = network
        self.optimizer = optimizer
        self.device = device
        
    def save_checkpoint(self, path: str):
        """保存检查点"""
        # 使用 Path 对象处理路径，确保跨平台兼容
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, str(save_path))
        
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint_path = Path(path)
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
    def train_step(self, *args, **kwargs) -> Dict:
        """训练步骤（需要被子类实现）"""
        raise NotImplementedError 