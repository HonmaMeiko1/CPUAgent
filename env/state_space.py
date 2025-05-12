import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import psutil
import os
from pathlib import Path

class TaskType(Enum):
    """任务类型"""
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
    cpu_usage: float  # CPU使用率 (%)
    memory_usage: float  # 内存使用率 (%)
    cpu_freq: float  # 当前CPU频率 (MHz)
    power_consumption: float  # 当前功耗 (W)
    temperature: float  # CPU温度 (°C)
    task_queue_size: int  # 任务队列大小
    active_tasks: List[TaskFeature]  # 活动任务列表
    
    @classmethod
    def from_system(cls) -> 'SystemState':
        """从系统状态创建实例"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent()
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # CPU频率
        cpu_freq = psutil.cpu_freq()
        current_freq = cpu_freq.current if cpu_freq else 0
        
        # CPU温度（Linux系统适配）
        temperature = 0.0
        try:
            # 首先尝试从thermal_zone获取温度
            for i in range(10):  # 通常thermal_zone的数量不会超过10
                thermal_path = Path(f"/sys/class/thermal/thermal_zone{i}/temp")
                if thermal_path.exists():
                    with open(thermal_path, 'r') as f:
                        temp = float(f.read().strip()) / 1000  # 转换为摄氏度
                        if temp > 0:
                            temperature = temp
                            break
            
            # 如果thermal_zone没有温度数据，尝试从k10temp模块获取
            if temperature == 0.0:
                k10temp_path = Path("/sys/class/hwmon/hwmon*/temp1_input")
                for path in Path("/sys/class/hwmon").glob("hwmon*/temp1_input"):
                    with open(path, 'r') as f:
                        temperature = float(f.read().strip()) / 1000
                        break
        except Exception:
            pass
                
        # 功耗（Linux系统适配）
        power_consumption = 0.0
        try:
            # 尝试从RAPL获取功耗数据
            rapl_paths = [
                Path("/sys/class/powercap/intel-rapl:0/energy_uj"),  # Intel CPU
                Path("/sys/class/powercap/amd-rapl:0/energy_uj"),    # AMD CPU
            ]
            
            for path in rapl_paths:
                if path.exists():
                    with open(path, 'r') as f:
                        power_consumption = float(f.read().strip()) / 1_000_000  # 转换为瓦特
                        break
                        
            # 如果RAPL不可用，尝试从ACPI获取
            if power_consumption == 0.0:
                acpi_path = Path("/sys/class/power_supply/BAT0/power_now")
                if acpi_path.exists():
                    with open(acpi_path, 'r') as f:
                        power_consumption = float(f.read().strip()) / 1_000_000
        except Exception:
            pass
                
        return cls(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            cpu_freq=current_freq,
            power_consumption=power_consumption,
            temperature=temperature,
            task_queue_size=0,
            active_tasks=[]
        )

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
                float(task.task_type.value == 'compute'),  # 将任务类型转换为数值
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
        
        return SystemState(
            cpu_usage=system_features[0],
            memory_usage=system_features[1],
            cpu_freq=system_features[2],
            power_consumption=system_features[3],
            temperature=system_features[4],
            task_queue_size=int(system_features[5]),
            active_tasks=[]  # 需要额外信息才能重建任务列表
        ) 