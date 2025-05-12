import torch
import torch.nn as nn
import numpy as np

class BaseNetwork(nn.Module):
    """神经网络基类"""
    def __init__(self):
        super().__init__()
        
    def _build_mlp(self, input_dim: int, output_dim: int, hidden_dims: list) -> nn.Sequential:
        """构建多层感知机"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
        
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_() 