import argparse
import logging
import yaml
import numpy as np
from typing import Dict, List, Tuple
import os
from datetime import datetime
import torch

from env.spark_env import SparkEnv
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from utils.plot_utils import PlotUtils
from agent.ppo_trainer import PPOTrainer

def setup_logging(config: Dict):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.get('logging', {}).get('log_file', 'testing.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test(config: Dict, model_path: str):
    """测试主函数"""
    # 创建保存目录
    os.makedirs('results', exist_ok=True)
    
    # 初始化环境
    env = SparkEnv(config)
    
    # 初始化训练器
    trainer = PPOTrainer(config)
    trainer.load_model(model_path)
    
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
        state = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_freq = []
        episode_time = 0
        
        for step in range(max_steps):
            # 选择动作
            action, _, _ = trainer.network.get_action(
                torch.FloatTensor(state).to(trainer.device),
                deterministic=True
            )
            action = action.cpu().numpy()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_energy += info.get('energy_consumption', 0)
            episode_freq.append(info.get('cpu_freq', 0))
            episode_time += info.get('step_time', 0)
            
            if done:
                break
                
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
    
    # 绘制测试结果
    PlotUtils.plot_job_execution(
        job_progress=[i/max_steps for i in range(max_steps)],
        execution_time=episode_times,
        energy_consumption=episode_energies,
        cpu_frequencies=episode_freqs,
        save_dir="results"
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                      help='模型文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    
    # 开始测试
    test(config, args.model_path)

if __name__ == '__main__':
    main() 