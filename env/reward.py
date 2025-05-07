import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from env.state_space import SystemState, TaskFeature

@dataclass
class RewardWeights:
    """奖励权重"""
    energy_weight: float = 0.4
    performance_weight: float = 0.3
    deadline_weight: float = 0.2
    resource_weight: float = 0.1

class RewardCalculator:
    """奖励计算器"""
    def __init__(self, config: Dict):
        self.config = config
        self.weights = self._init_weights()
        self.baseline_power = self.config.get('reward', {}).get('baseline_power', 100.0)
        self.baseline_time = self.config.get('reward', {}).get('baseline_time', 300.0)
        
    def _init_weights(self) -> RewardWeights:
        """初始化奖励权重"""
        return RewardWeights(
            energy_weight=self.config.get('reward', {}).get('energy_weight', 0.4),
            performance_weight=self.config.get('reward', {}).get('performance_weight', 0.3),
            deadline_weight=self.config.get('reward', {}).get('deadline_weight', 0.2),
            resource_weight=self.config.get('reward', {}).get('resource_weight', 0.1)
        )
        
    def calculate_energy_reward(self, current_power: float) -> float:
        """计算能耗奖励"""
        # 相对于基准能耗的节省率
        energy_saving = (self.baseline_power - current_power) / self.baseline_power
        return np.clip(energy_saving, -1.0, 1.0)
        
    def calculate_performance_reward(self, execution_time: float) -> float:
        """计算性能奖励"""
        # 相对于基准时间的增加率
        time_increase = (execution_time - self.baseline_time) / self.baseline_time
        return np.clip(-time_increase, -1.0, 1.0)
        
    def calculate_deadline_reward(self, task: TaskFeature, current_time: float) -> float:
        """计算截止时间奖励"""
        if current_time > task.deadline:
            return -1.0
        time_remaining = task.deadline - current_time
        return np.clip(time_remaining / task.estimated_duration, -1.0, 1.0)
        
    def calculate_resource_reward(self, system_state: SystemState) -> float:
        """计算资源利用率奖励"""
        # 计算CPU和内存的平均使用率
        resource_utilization = (system_state.cpu_usage + system_state.memory_usage) / 2
        return np.clip(resource_utilization / 100.0, 0.0, 1.0)
        
    def calculate_reward(self, 
                        system_state: SystemState,
                        execution_time: float,
                        current_time: float) -> float:
        """计算总奖励"""
        # 计算各个奖励分量
        energy_reward = self.calculate_energy_reward(system_state.power_consumption)
        performance_reward = self.calculate_performance_reward(execution_time)
        resource_reward = self.calculate_resource_reward(system_state)
        
        # 计算任务相关的奖励
        deadline_reward = 0.0
        if system_state.active_tasks:
            # 使用最高优先级任务的截止时间
            highest_priority_task = max(
                system_state.active_tasks,
                key=lambda x: x.priority
            )
            deadline_reward = self.calculate_deadline_reward(
                highest_priority_task,
                current_time
            )
            
        # 加权求和
        total_reward = (
            self.weights.energy_weight * energy_reward +
            self.weights.performance_weight * performance_reward +
            self.weights.deadline_weight * deadline_reward +
            self.weights.resource_weight * resource_reward
        )
        
        return total_reward 