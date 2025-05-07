import argparse
import yaml
import logging
import os
from typing import Dict
from train import train
from test import test

def setup_logging(config: Dict):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.get('logging', {}).get('log_file', 'drl_energy_saving.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DRL Energy Saving')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'test'],
                      required=True, help='运行模式')
    parser.add_argument('--model_path', type=str,
                      help='模型文件路径（测试模式需要）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(config)
    
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        if args.mode == 'train':
            train(config)
        else:
            if not args.model_path:
                raise ValueError("测试模式需要指定模型文件路径")
            test(config, args.model_path)
            
    except Exception as e:
        logging.error(f"运行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main() 