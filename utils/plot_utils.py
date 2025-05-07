import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import os
from datetime import datetime

class PlotUtils:
    """绘图工具类"""
    
    @staticmethod
    def plot_training_results(rewards: List[float], losses: List[float], 
                            energy_consumption: List[float], cpu_freqs: List[float],
                            algorithm: str, save_dir: str = "results"):
        """
        绘制训练结果
        
        Args:
            rewards: 奖励历史
            losses: 损失历史
            energy_consumption: 能量消耗历史
            cpu_freqs: CPU频率历史
            algorithm: 算法名称
            save_dir: 保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algorithm} Training Results', fontsize=16)
        
        # 绘制奖励曲线
        ax1.plot(rewards, label='Reward')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        ax1.legend()
        
        # 绘制损失曲线
        ax2.plot(losses, label='Loss')
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        ax2.legend()
        
        # 绘制能量消耗曲线
        ax3.plot(energy_consumption, label='Energy Consumption')
        ax3.set_title('Energy Consumption')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Energy (J)')
        ax3.grid(True)
        ax3.legend()
        
        # 绘制CPU频率曲线
        ax4.plot(cpu_freqs, label='CPU Frequency')
        ax4.set_title('CPU Frequency')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Frequency (GHz)')
        ax4.grid(True)
        ax4.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'training_results_{algorithm}_{timestamp}.png')
        plt.savefig(filename)
        plt.close()
        
    @staticmethod
    def plot_job_execution(job_progress: List[float], execution_time: List[float],
                          energy_consumption: List[float], cpu_freqs: List[float],
                          algorithm: str, save_dir: str = "results"):
        """
        绘制作业执行结果
        
        Args:
            job_progress: 作业进度历史
            execution_time: 执行时间历史
            energy_consumption: 能量消耗历史
            cpu_freqs: CPU频率历史
            algorithm: 算法名称
            save_dir: 保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algorithm} Job Execution Results', fontsize=16)
        
        # 绘制作业进度曲线
        ax1.plot(job_progress, label='Job Progress')
        ax1.set_title('Job Progress')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Progress (%)')
        ax1.grid(True)
        ax1.legend()
        
        # 绘制执行时间曲线
        ax2.plot(execution_time, label='Execution Time')
        ax2.set_title('Execution Time')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Time (s)')
        ax2.grid(True)
        ax2.legend()
        
        # 绘制能量消耗曲线
        ax3.plot(energy_consumption, label='Energy Consumption')
        ax3.set_title('Energy Consumption')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Energy (J)')
        ax3.grid(True)
        ax3.legend()
        
        # 绘制CPU频率曲线
        ax4.plot(cpu_freqs, label='CPU Frequency')
        ax4.set_title('CPU Frequency')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Frequency (GHz)')
        ax4.grid(True)
        ax4.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'job_execution_{algorithm}_{timestamp}.png')
        plt.savefig(filename)
        plt.close()
        
    @staticmethod
    def plot_comparison(results: Dict[str, Dict[str, List[float]]], 
                       metrics: List[str], save_dir: str = "results"):
        """
        绘制不同算法的比较结果
        
        Args:
            results: 不同算法的结果字典
            metrics: 要比较的指标列表
            save_dir: 保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 为每个指标创建单独的图表
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            for algorithm, data in results.items():
                if metric in data:
                    plt.plot(data[metric], label=algorithm)
                    
            plt.title(f'{metric} Comparison')
            plt.xlabel('Step')
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f'comparison_{metric}_{timestamp}.png')
            plt.savefig(filename)
            plt.close() 