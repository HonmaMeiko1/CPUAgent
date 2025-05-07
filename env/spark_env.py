import gym
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from monitor.spark_monitor import SparkMonitor
from monitor.system_monitor import SystemMonitor
from monitor.hibench_monitor import HiBenchMonitor
from control.cpu_control import CPUControl
from control.memory_control import MemoryControl

class SparkEnv(gym.Env):
    def __init__(self, config: Dict):
        """
        初始化环境
        
        Args:
            config: 配置字典
        """
        super().__init__()
        
        # 初始化监控器
        self.spark_monitor = SparkMonitor(config['spark'])
        self.system_monitor = SystemMonitor(config['monitoring'])
        self.hibench_monitor = HiBenchMonitor(
            config['spark']['hibench']['hibench_home'],
            config['spark']['hibench']
        )
        
        # 初始化控制器
        self.cpu_control = CPUControl(config['control'])
        self.memory_control = MemoryControl(config['control'])
        
        # 环境配置
        self.config = config
        self.current_step = 0
        self.current_episode = 0
        self.current_workload = 0
        self.workloads = config['training']['hibench_training']['workloads']
        self.iterations = config['training']['hibench_training']['iterations']
        self.current_iteration = 0
        
        # 定义动作空间和观察空间
        self.action_space = gym.spaces.Discrete(9)  # 9种CPU频率组合
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),  # 状态维度
            dtype=np.float32
        )
        
        self.logger = logging.getLogger(__name__)
        
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            np.ndarray: 初始状态
        """
        self.current_step = 0
        self.current_iteration += 1
        
        # 如果完成所有迭代，切换到下一个工作负载
        if self.current_iteration >= self.iterations:
            self.current_iteration = 0
            self.current_workload = (self.current_workload + 1) % len(self.workloads)
            
        # 准备并运行当前工作负载
        workload = self.workloads[self.current_workload]
        if self.hibench_monitor.prepare_data(workload['name'], workload['scale']):
            self.hibench_monitor.run_workload(workload['name'], workload['scale'])
            
        # 获取初始状态
        return self._get_state()
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 动作索引
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: (状态, 奖励, 是否结束, 信息)
        """
        self.current_step += 1
        
        # 执行动作
        cpu_freq = self._action_to_freq(action)
        self.cpu_control.set_frequency(cpu_freq)
        
        # 等待动作生效
        time.sleep(self.config['inference']['action_interval'])
        
        # 获取新状态
        state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否结束
        done = self._is_done()
        
        # 额外信息
        info = {
            'step': self.current_step,
            'episode': self.current_episode,
            'workload': self.workloads[self.current_workload]['name'],
            'iteration': self.current_iteration,
            'cpu_freq': cpu_freq
        }
        
        return state, reward, done, info
        
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态
        
        Returns:
            np.ndarray: 状态向量
        """
        # 获取Spark指标
        spark_metrics = self.spark_monitor.get_metrics()
        
        # 获取系统指标
        system_metrics = self.system_monitor.get_metrics()
        
        # 获取工作负载指标
        workload_metrics = self.hibench_monitor.get_workload_metrics()
        
        # 构建状态向量
        state = np.array([
            spark_metrics['executor_cores'],
            spark_metrics['executor_memory'],
            spark_metrics['driver_memory'],
            system_metrics['cpu_usage'],
            system_metrics['memory_usage'],
            system_metrics['power_usage'],
            workload_metrics.get('elapsed_time', 0),
            workload_metrics.get('memory_seconds', 0),
            workload_metrics.get('vcore_seconds', 0),
            workload_metrics.get('containers', 0)
        ], dtype=np.float32)
        
        return state
        
    def _calculate_reward(self) -> float:
        """
        计算奖励
        
        Returns:
            float: 奖励值
        """
        # 获取当前指标
        system_metrics = self.system_monitor.get_metrics()
        workload_metrics = self.hibench_monitor.get_workload_metrics()
        
        # 计算能量效率
        if workload_metrics.get('elapsed_time', 0) > 0:
            energy_efficiency = system_metrics['power_usage'] / workload_metrics['elapsed_time']
        else:
            energy_efficiency = 0
            
        # 计算性能惩罚
        performance_penalty = 0
        if workload_metrics.get('elapsed_time', 0) > 0:
            # 如果执行时间超过预期，增加惩罚
            expected_time = self.config['training'].get('expected_time', 300)  # 默认5分钟
            if workload_metrics['elapsed_time'] > expected_time:
                performance_penalty = (workload_metrics['elapsed_time'] - expected_time) / expected_time
                
        # 计算资源利用率奖励
        resource_reward = (system_metrics['cpu_usage'] + system_metrics['memory_usage']) / 2
        
        # 总奖励
        reward = -energy_efficiency - performance_penalty + resource_reward
        
        return reward
        
    def _is_done(self) -> bool:
        """
        检查是否结束
        
        Returns:
            bool: 是否结束
        """
        # 检查步数限制
        if self.current_step >= self.config['training']['max_steps']:
            return True
            
        # 检查工作负载是否完成
        status = self.hibench_monitor.get_workload_status()
        if not status['running'] and status['final_status'] == 'SUCCEEDED':
            return True
            
        return False
        
    def _action_to_freq(self, action: int) -> float:
        """
        将动作转换为CPU频率
        
        Args:
            action: 动作索引
            
        Returns:
            float: CPU频率
        """
        # 9种频率组合
        freqs = [
            (2.0, 2.0),  # 全核心2.0GHz
            (2.0, 1.5),  # 核心1: 2.0GHz, 核心2: 1.5GHz
            (2.0, 1.0),  # 核心1: 2.0GHz, 核心2: 1.0GHz
            (1.5, 2.0),  # 核心1: 1.5GHz, 核心2: 2.0GHz
            (1.5, 1.5),  # 全核心1.5GHz
            (1.5, 1.0),  # 核心1: 1.5GHz, 核心2: 1.0GHz
            (1.0, 2.0),  # 核心1: 1.0GHz, 核心2: 2.0GHz
            (1.0, 1.5),  # 核心1: 1.0GHz, 核心2: 1.5GHz
            (1.0, 1.0)   # 全核心1.0GHz
        ]
        return freqs[action]
        
    def close(self):
        """关闭环境"""
        self.hibench_monitor.cleanup()
        self.spark_monitor.close()
        self.system_monitor.close()
        self.cpu_control.close()
        self.memory_control.close()
        
    def render(self, mode='human'):
        """
        渲染环境（可选）
        """
        pass 