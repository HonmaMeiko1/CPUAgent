import os
from pathlib import Path

class Config:
    """环境配置类"""
    
    @staticmethod
    def get_default_checkpoint_dir() -> Path:
        """获取默认检查点保存目录"""
        # 在用户主目录下创建 .cpuagent 目录
        return Path.home() / '.cpuagent' / 'checkpoints'
    
    @staticmethod
    def get_default_log_dir() -> Path:
        """获取默认日志保存目录"""
        return Path.home() / '.cpuagent' / 'logs'
    
    @staticmethod
    def ensure_dir(path: Path) -> None:
        """确保目录存在"""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_temp_dir() -> Path:
        """获取临时文件目录"""
        return Path('/tmp/cpuagent')
    
    @staticmethod
    def init_directories():
        """初始化所有必要的目录"""
        dirs = [
            Config.get_default_checkpoint_dir(),
            Config.get_default_log_dir(),
            Config.get_temp_dir()
        ]
        for dir_path in dirs:
            Config.ensure_dir(dir_path) 