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
    log_file = log_dir / f"training_{timestamp}.log"
    
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

def save_training_config(config: Dict, save_dir: Path):
    """保存训练配置"""
    config_file = save_dir / 'training_config.json'
    with config_file.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    config_file.chmod(0o644)

def train(config: Dict):
    """训练主函数"""
    kill_signal = GracefulExit()
    
    try:
        # 设置日志
        log_file = setup_logging(config)
        
        # 创建必要的目录
        for dir_path in ['models', 'results', 'checkpoints']:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o755)
        
        # 保存训练配置
        save_training_config(config, Path('results'))
        
        # 初始化系统监控
        system_monitor = SystemMonitor()
        
        # 初始化环境
        env = SparkEnv(config)
        
        # 初始化PPO智能体和训练器
        agent = PPOAgent(config)
        trainer = PPOTrainer(agent, config)
        
        # 训练参数
        n_episodes = config.get('training', {}).get('n_episodes', 1000)
        max_steps = config.get('training', {}).get('max_steps', 1000)
        save_interval = config.get('training', {}).get('save_interval', 100)
        
        # 记录训练指标
        episode_rewards = []
        episode_losses = []
        episode_energies = []
        
        # 开始训练循环
        for episode in range(n_episodes):
            if kill_signal.kill_now:
                break
                
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_energy = 0
            
            for step in range(max_steps):
                if kill_signal.kill_now:
                    break
                    
                # 获取系统状态
                system_state = system_monitor.get_state()
                
                # 选择动作
                action = agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储经验
                agent.store_transition(state, action, reward, next_state, done)
                
                # 更新状态
                state = next_state
                episode_reward += reward
                episode_energy += info.get('energy_consumption', 0)
                
                if done:
                    break
                    
            # 更新模型
            loss = trainer.update()
            episode_loss = loss if loss is not None else 0
            
            # 记录指标
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)
            episode_energies.append(episode_energy)
            
            # 保存检查点
            if (episode + 1) % save_interval == 0:
                checkpoint_path = Path('checkpoints') / f"checkpoint_episode_{episode+1}.pt"
                trainer.save_checkpoint(checkpoint_path)
                checkpoint_path.chmod(0o644)
                
                # 绘制训练进度图
                PlotUtils.plot_training_progress(
                    rewards=episode_rewards,
                    losses=episode_losses,
                    energies=episode_energies,
                    save_dir="results"
                )
            
            logging.info(f"Episode {episode+1}/{n_episodes}")
            logging.info(f"Reward: {episode_reward:.2f}")
            logging.info(f"Loss: {episode_loss:.4f}")
            logging.info(f"Energy: {episode_energy:.2f}")
            
        # 保存最终模型
        final_model_path = Path('models') / "final_model.pt"
        trainer.save_model(final_model_path)
        final_model_path.chmod(0o644)
        
        # 保存训练历史
        history = {
            'rewards': episode_rewards,
            'losses': episode_losses,
            'energies': episode_energies
        }
        history_path = Path('results') / 'training_history.json'
        with history_path.open('w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        history_path.chmod(0o644)
        
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}")
        raise
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()
        if 'system_monitor' in locals():
            system_monitor.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        train(config)
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 