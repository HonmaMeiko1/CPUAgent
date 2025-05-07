import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class TaskType(Enum):
    COMPUTE_INTENSIVE = "compute"
    IO_INTENSIVE = "io"
    MIXED = "mixed"

@dataclass
class TaskFeature:
    """任务特征"""
    task_id: str
    task_type: TaskType
    priority: int
    deadline: float
    resource_requirements: Dict[str, float]
    dependencies: List[str]
    estimated_duration: float

@dataclass
class SystemState:
    """系统状态"""
    cpu_usage: float  # CPU使用率
    memory_usage: float  # 内存使用率
    cpu_freq: float  # 当前CPU频率
    power_consumption: float  # 当前功耗
    temperature: float  # CPU温度
    task_queue_size: int  # 任务队列大小
    active_tasks: List[TaskFeature]  # 活动任务列表

class StateSpace:
    """状态空间管理"""
    def __init__(self, config: Dict):
        self.config = config
        self.state_dim = self._calculate_state_dim()
        
    def _calculate_state_dim(self) -> int:
        """计算状态空间维度"""
        # 系统状态维度
        system_dim = 6  # cpu_usage, memory_usage, cpu_freq, power, temp, queue_size
        # 任务特征维度
        task_dim = 5  # type, priority, deadline, resource_req, duration
        # 考虑最近N个任务
        n_tasks = self.config.get('state_space', {}).get('n_recent_tasks', 3)
        
        return system_dim + task_dim * n_tasks
        
    def encode_state(self, system_state: SystemState) -> np.ndarray:
        """将系统状态编码为向量"""
        # 系统状态部分
        system_features = np.array([
            system_state.cpu_usage,
            system_state.memory_usage,
            system_state.cpu_freq,
            system_state.power_consumption,
            system_state.temperature,
            system_state.task_queue_size
        ])
        
        # 任务特征部分
        n_tasks = self.config.get('state_space', {}).get('n_recent_tasks', 3)
        task_features = []
        
        for task in system_state.active_tasks[:n_tasks]:
            task_features.extend([
                task.task_type.value,
                task.priority,
                task.deadline,
                sum(task.resource_requirements.values()),
                task.estimated_duration
            ])
            
        # 如果任务数量不足，用0填充
        if len(task_features) < n_tasks * 5:
            task_features.extend([0] * (n_tasks * 5 - len(task_features)))
            
        return np.concatenate([system_features, np.array(task_features)])
        
    def decode_state(self, state_vector: np.ndarray) -> SystemState:
        """将向量解码为系统状态（用于调试）"""
        system_features = state_vector[:6]
        task_features = state_vector[6:]
        
        return SystemState(
            cpu_usage=system_features[0],
            memory_usage=system_features[1],
            cpu_freq=system_features[2],
            power_consumption=system_features[3],
            temperature=system_features[4],
            task_queue_size=int(system_features[5]),
            active_tasks=[]  # 需要额外信息才能重建任务列表
        ) 