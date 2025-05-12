import gym
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from pathlib import Path
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
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 验证配置路径
        self._validate_paths(config)
        
        # 初始化监控器
        self.spark_monitor = SparkMonitor(config['spark'])
        self.system_monitor = SystemMonitor(config['monitoring'])
        self.hibench_monitor = HiBenchMonitor(
            config['spark']['hibench']['hibench_home'],
            config['spark']['hibench']
        )
        
        # 初始化控制器
        self.cpu_control = CPUControl()
        self.memory_control = MemoryControl()
        
        # 环境配置
        self.config = config
        self.current_step = 0
        self.current_episode = 0
        self.current_workload = 0
        self.workloads = config['training']['hibench_training']['workloads']
        self.iterations = config['training']['hibench_training']['iterations']
        self.current_iteration = 0
        
        # 定义动作空间和观察空间
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),  # 最小CPU频率比例
            high=np.array([1.0]),  # 最大CPU频率比例
            shape=(1,),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),  # 状态维度
            dtype=np.float32
        )
        
    def _validate_paths(self, config: Dict):
        """验证配置中的路径（Linux系统适配）"""
        paths_to_check = [
            ('spark', 'home'),
            ('spark', 'hibench', 'hibench_home'),
            ('spark', 'hibench', 'report_path'),
            ('monitoring', 'log_dir')
        ]
        
        for path_keys in paths_to_check:
            path_value = config
            for key in path_keys:
                path_value = path_value.get(key, {})
            
            if isinstance(path_value, str):
                # 转换Windows路径为Linux路径
                path_str = str(path_value).replace('\\', '/')
                if ':' in path_str:  # 处理Windows驱动器号
                    path_str = '/' + path_str.replace(':', '')
                path = Path(path_str)
                
                if not path.exists():
                    self.logger.warning(f"路径不存在: {path}")
                    path.mkdir(parents=True, exist_ok=True)
                    
                # 设置适当的权限
                try:
                    path.chmod(0o755)  # rwxr-xr-x
                except Exception as e:
                    self.logger.warning(f"设置路径权限失败: {str(e)}")
        
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
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        
        Args:
            action: 动作值（CPU频率比例）
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: (状态, 奖励, 是否结束, 信息)
        """
        self.current_step += 1
        
        # 获取CPU频率范围（Linux系统适配）
        cpu_info = self.cpu_control.get_current_cpu_state()
        min_freq = cpu_info.get('min_frequency', 1200)  # 默认1.2GHz
        max_freq = cpu_info.get('max_frequency', 3600)  # 默认3.6GHz
        
        # 将动作值转换为实际频率
        target_freq = int(min_freq + action[0] * (max_freq - min_freq))
        
        # 使用Linux的cpufreq设置频率
        try:
            for cpu_path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_setspeed'):
                with open(cpu_path, 'w') as f:
                    f.write(str(target_freq * 1000))  # 转换为kHz
        except Exception as e:
            self.logger.warning(f"设置CPU频率失败: {str(e)}")
        
        # 等待动作生效
        time.sleep(self.config['inference'].get('action_interval', 1.0))
        
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
            'cpu_freq': target_freq
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
            spark_metrics.get('executor_cores', 0),
            spark_metrics.get('executor_memory', 0),
            spark_metrics.get('driver_memory', 0),
            system_metrics.get('cpu_usage', 0),
            system_metrics.get('memory_usage', 0),
            system_metrics.get('power_usage', 0),
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
            energy_efficiency = system_metrics.get('power_usage', 0) / workload_metrics['elapsed_time']
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
        resource_reward = (
            system_metrics.get('cpu_usage', 0) + 
            system_metrics.get('memory_usage', 0)
        ) / 2
        
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
        if self.current_step >= self.config['training'].get('max_steps', 1000):
            return True
            
        # 检查工作负载是否完成
        status = self.hibench_monitor.get_workload_status()
        if not status['running'] and status['final_status'] == 'SUCCEEDED':
            return True
            
        return False
        
    def close(self):
        """清理环境（Linux系统适配）"""
        try:
            # 恢复默认CPU频率
            for cpu_path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor'):
                with open(cpu_path, 'w') as f:
                    f.write('ondemand')  # 使用Linux默认的ondemand调频器
                    
            # 恢复默认内存设置
            self.memory_control.optimize_memory_usage()
        except Exception as e:
            self.logger.error(f"清理环境时出错: {str(e)}")
            
    def render(self, mode='human'):
        """
        渲染环境（可选）
        """
        pass 