import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import logging

class PlotUtils:
    """绘图工具类"""
    
    def __init__(self):
        """初始化绘图工具"""
        self.logger = logging.getLogger(__name__)
        # 设置matplotlib后端为Agg，避免在无GUI环境下的问题
        plt.switch_backend('Agg')
        # 设置中文字体支持
        try:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            self.logger.warning(f"设置中文字体失败: {str(e)}")
    
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
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path.chmod(0o755)  # 设置目录权限
        
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
        filename = save_path / f'training_results_{algorithm}_{timestamp}.png'
        try:
            plt.savefig(filename)
            filename.chmod(0o644)  # 设置文件权限
        except Exception as e:
            logging.error(f"保存图表失败: {str(e)}")
        finally:
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
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path.chmod(0o755)  # 设置目录权限
        
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
        filename = save_path / f'job_execution_{algorithm}_{timestamp}.png'
        try:
            plt.savefig(filename)
            filename.chmod(0o644)  # 设置文件权限
        except Exception as e:
            logging.error(f"保存图表失败: {str(e)}")
        finally:
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
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path.chmod(0o755)  # 设置目录权限
        
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
            filename = save_path / f'comparison_{metric}_{timestamp}.png'
            try:
                plt.savefig(filename)
                filename.chmod(0o644)  # 设置文件权限
            except Exception as e:
                logging.error(f"保存图表失败: {str(e)}")
            finally:
                plt.close() 