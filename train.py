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
            logging.FileHandler(config.get('logging', {}).get('log_file', 'training.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_dqn(env: SparkEnv, agent: DQNAgent, config: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    训练DQN智能体
    
    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: (奖励历史, 损失历史, 能量消耗历史, CPU频率历史)
    """
    rewards_history = []
    losses_history = []
    energy_history = []
    cpu_freq_history = []
    
    episodes = config['training']['episodes']
    max_steps = config['training']['max_steps']
    save_interval = config['training']['save_interval']
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_cpu_freqs = []
        
        for step in range(max_steps):
            # 选择动作
            action = agent.get_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_energy += info.get('energy_consumption', 0)
            episode_cpu_freqs.append(info.get('cpu_freq', 0))
            
            # 训练智能体
            if len(agent.memory) > agent.batch_size:
                loss = agent.update()
                losses_history.append(loss)
            
            if done:
                break
                
        # 记录结果
        rewards_history.append(episode_reward)
        energy_history.append(episode_energy)
        cpu_freq_history.append(np.mean(episode_cpu_freqs))
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(f"models/dqn_episode_{episode+1}.pt")
            
        # 打印进度
        logging.info(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
                    f"Energy: {episode_energy:.2f}, CPU Freq: {np.mean(episode_cpu_freqs):.2f}")
                    
    # 保存最终模型
    agent.save("models/dqn_final.pt")
    
    return rewards_history, losses_history, energy_history, cpu_freq_history

def train_ppo(env: SparkEnv, agent: PPOAgent, config: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    训练PPO智能体
    
    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: (奖励历史, 损失历史, 能量消耗历史, CPU频率历史)
    """
    rewards_history = []
    losses_history = []
    energy_history = []
    cpu_freq_history = []
    
    episodes = config['training']['episodes']
    max_steps = config['training']['max_steps']
    save_interval = config['training']['save_interval']
    update_interval = config['training']['update_interval']
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_cpu_freqs = []
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_values = []
        episode_log_probs = []
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = agent.get_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储轨迹
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_values.append(value)
            episode_log_probs.append(log_prob)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_energy += info.get('energy_consumption', 0)
            episode_cpu_freqs.append(info.get('cpu_freq', 0))
            
            # 更新策略
            if (step + 1) % update_interval == 0:
                loss = agent.update(
                    episode_states,
                    episode_actions,
                    episode_rewards,
                    episode_values,
                    episode_log_probs
                )
                losses_history.append(loss)
                
                # 清空轨迹
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_values = []
                episode_log_probs = []
            
            if done:
                break
                
        # 记录结果
        rewards_history.append(episode_reward)
        energy_history.append(episode_energy)
        cpu_freq_history.append(np.mean(episode_cpu_freqs))
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(f"models/ppo_episode_{episode+1}.pt")
            
        # 打印进度
        logging.info(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, "
                    f"Energy: {episode_energy:.2f}, CPU Freq: {np.mean(episode_cpu_freqs):.2f}")
                    
    # 保存最终模型
    agent.save("models/ppo_final.pt")
    
    return rewards_history, losses_history, energy_history, cpu_freq_history

def train(config: Dict):
    """训练主函数"""
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 初始化环境
    env = SparkEnv(config)
    
    # 初始化训练器
    trainer = PPOTrainer(config)
    
    # 训练参数
    n_episodes = config.get('training', {}).get('n_episodes', 1000)
    max_steps = config.get('training', {}).get('max_steps', 1000)
    save_interval = config.get('training', {}).get('save_interval', 100)
    
    # 记录训练指标
    episode_rewards = []
    episode_losses = []
    episode_energies = []
    episode_freqs = []
    
    # 训练循环
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_freq = []
        
        # 收集轨迹
        states = []
        actions = []
        rewards = []
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = trainer.network.get_action(
                torch.FloatTensor(state).to(trainer.device)
            )
            action = action.cpu().numpy()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 记录数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_energy += info.get('energy_consumption', 0)
            episode_freq.append(info.get('cpu_freq', 0))
            
            if done:
                break
                
        # 训练网络
        metrics = trainer.train_step(states, actions, rewards, next_state)
        
        # 记录指标
        episode_rewards.append(episode_reward)
        episode_losses.append(metrics['total_loss'])
        episode_energies.append(episode_energy)
        episode_freqs.append(np.mean(episode_freq))
        
        # 打印进度
        logging.info(f"Episode {episode+1}/{n_episodes}")
        logging.info(f"Reward: {episode_reward:.2f}")
        logging.info(f"Loss: {metrics['total_loss']:.4f}")
        logging.info(f"Energy: {episode_energy:.2f}")
        logging.info(f"CPU Freq: {np.mean(episode_freq):.2f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            trainer.save_model(f"models/ppo_episode_{episode+1}.pt")
            
    # 保存最终模型
    trainer.save_model("models/ppo_final.pt")
    
    # 绘制训练结果
    PlotUtils.plot_training_results(
        rewards=episode_rewards,
        losses=episode_losses,
        energy_consumption=episode_energies,
        cpu_frequencies=episode_freqs,
        algorithm="PPO",
        save_dir="results"
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    
    # 开始训练
    train(config)

if __name__ == '__main__':
    main() 