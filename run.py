import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict
import sys
from train import train
from test import test

def setup_logging(config: Dict):
    """设置日志"""
    log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_dir.chmod(0o755)  # Linux目录权限
    
    log_file = log_dir / config.get('logging', {}).get('log_file', 'drl_energy_saving.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Linux文件权限
    log_file.chmod(0o644)

def load_config(config_path: str) -> Dict:
    """加载配置"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
        with config_file.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 确保所有路径使用Linux格式
        if 'paths' in config:
            for key, path in config['paths'].items():
                config['paths'][key] = str(Path(path))
                
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise

def init_directories(config: Dict):
    """初始化必要的目录"""
    directories = {
        'models': Path('models'),
        'results': Path('results'),
        'logs': Path(config.get('logging', {}).get('log_dir', 'logs')),
        'checkpoints': Path('checkpoints'),
        'data': Path('data')
    }
    
    for dir_name, dir_path in directories.items():
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            dir_path.chmod(0o755)  # Linux目录权限
            logging.info(f"已创建目录: {dir_path}")
        except Exception as e:
            logging.error(f"创建目录 {dir_path} 失败: {str(e)}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DRL Energy Saving Agent')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'test'],
                      required=True, help='运行模式: train或test')
    parser.add_argument('--model_path', type=str,
                      help='模型文件路径（测试模式需要）')
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 设置日志
        setup_logging(config)
        
        # 初始化目录
        init_directories(config)
        
        # 运行对应模式
        if args.mode == 'train':
            train(config)
        else:
            if not args.model_path:
                raise ValueError("测试模式需要指定模型文件路径 (--model_path)")
            model_path = Path(args.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            if not model_path.is_file():
                raise ValueError(f"指定的模型路径不是文件: {model_path}")
            test(config, str(model_path))
            
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logging.error(f"运行出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 