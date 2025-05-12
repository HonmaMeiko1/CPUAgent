import argparse
import logging
import yaml
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import signal
from datetime import datetime
import torch
import json

from env.spark_env import SparkEnv
from utils.plot_utils import PlotUtils
from monitor.system_monitor import SystemMonitor
from agent.agents.ppo_agent import PPOAgent
from agent.trainers.ppo_trainer import PPOTrainer

class GracefulExit:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        
    def exit_gracefully(self, signum, frame):
        logging.info("接收到终止信号，正在优雅退出...")
        self.kill_now = True

def setup_logging(config: Dict):
    """设置日志"""
    log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o755)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"testing_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    log_file.chmod(0o644)
    return log_file

def save_test_config(config: Dict, save_dir: Path):
    """保存测试配置"""
    config_file = save_dir / 'test_config.json'
    with config_file.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    config_file.chmod(0o644)

def test(config: Dict, model_path: str):
    """测试主函数"""
    kill_signal = GracefulExit()
    
    try:
        # 设置日志
        log_file = setup_logging(config)
        
        # 创建结果目录
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)
        results_dir.chmod(0o755)
        
        # 保存测试配置
        save_test_config(config, results_dir)
        
        # 初始化系统监控
        system_monitor = SystemMonitor()
        
        # 初始化环境
        env = SparkEnv(config)
        
        # 加载模型
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        if not model_file.is_file():
            raise ValueError(f"指定的模型路径不是文件: {model_file}")
            
        # 初始化PPO智能体和训练器
        agent = PPOAgent(config)
        trainer = PPOTrainer(agent, config)
        trainer.load_model(str(model_file))
        
        # 测试参数
        n_episodes = config.get('testing', {}).get('n_episodes', 10)
        max_steps = config.get('testing', {}).get('max_steps', 1000)
        
        # 记录测试指标
        episode_rewards = []
        episode_energies = []
        episode_freqs = []
        episode_times = []
        
        # 测试循环
        for episode in range(n_episodes):
            if kill_signal.kill_now:
                break
                
            state = env.reset()
            episode_reward = 0
            episode_energy = 0
            episode_freq = []
            start_time = datetime.now()
            
            for step in range(max_steps):
                if kill_signal.kill_now:
                    break
                    
                # 获取系统状态
                system_state = system_monitor.get_state()
                
                # 选择动作
                action = agent.select_action(state, deterministic=True)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_energy += info.get('energy_consumption', 0)
                episode_freq.append(info.get('cpu_freq', 0))
                
                if done:
                    break
                    
            # 计算执行时间
            episode_time = (datetime.now() - start_time).total_seconds()
            
            # 记录指标
            episode_rewards.append(episode_reward)
            episode_energies.append(episode_energy)
            episode_freqs.append(np.mean(episode_freq))
            episode_times.append(episode_time)
            
            # 打印进度
            logging.info(f"Episode {episode+1}/{n_episodes}")
            logging.info(f"Reward: {episode_reward:.2f}")
            logging.info(f"Energy: {episode_energy:.2f}")
            logging.info(f"CPU Freq: {np.mean(episode_freq):.2f}")
            logging.info(f"Time: {episode_time:.2f}s")
            
        # 计算平均指标
        avg_reward = np.mean(episode_rewards)
        avg_energy = np.mean(episode_energies)
        avg_freq = np.mean(episode_freqs)
        avg_time = np.mean(episode_times)
        
        logging.info("\n测试结果:")
        logging.info(f"平均奖励: {avg_reward:.2f}")
        logging.info(f"平均能耗: {avg_energy:.2f}")
        logging.info(f"平均CPU频率: {avg_freq:.2f}")
        logging.info(f"平均执行时间: {avg_time:.2f}s")
        
        # 保存测试结果
        results = {
            'rewards': episode_rewards,
            'energies': episode_energies,
            'frequencies': episode_freqs,
            'times': episode_times,
            'averages': {
                'reward': float(avg_reward),
                'energy': float(avg_energy),
                'frequency': float(avg_freq),
                'time': float(avg_time)
            }
        }
        
        results_file = results_dir / 'test_results.json'
        with results_file.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        results_file.chmod(0o644)
        
        # 绘制测试结果
        PlotUtils.plot_test_results(
            rewards=episode_rewards,
            energies=episode_energies,
            frequencies=episode_freqs,
            times=episode_times,
            save_dir=results_dir
        )
        
    except Exception as e:
        logging.error(f"测试过程出错: {str(e)}")
        raise
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()
        if 'system_monitor' in locals():
            system_monitor.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DRL Energy Saving Agent Testing')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型文件路径')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        test(config, args.model_path)
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 