import os
import logging
from pathlib import Path
import subprocess
from typing import Dict, Optional, List
import re
import json
import time

class HiBenchMonitor:
    def __init__(self, hibench_home: str, config: Dict):
        """
        初始化HiBench监控器
        
        Args:
            hibench_home: HiBench安装目录
            config: HiBench配置
        """
        self.hibench_home = Path(hibench_home)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validate_installation()
        
    def _validate_installation(self):
        """验证HiBench安装"""
        try:
            # 检查HiBench目录
            if not self.hibench_home.exists():
                raise FileNotFoundError(f"HiBench目录不存在: {self.hibench_home}")
                
            # 检查必要的目录和文件
            required_paths = [
                'conf',
                'bin',
                'workloads',
                'report'
            ]
            
            for path in required_paths:
                if not (self.hibench_home / path).exists():
                    raise FileNotFoundError(f"找不到必要的HiBench目录: {path}")
                    
            # 设置执行权限
            bin_dir = self.hibench_home / 'bin'
            for script in bin_dir.glob('*.sh'):
                script.chmod(0o755)
                
        except Exception as e:
            self.logger.error(f"HiBench安装验证失败: {str(e)}")
            raise
            
    def _execute_command(self, cmd: List[str], cwd: Optional[Path] = None) -> str:
        """执行命令
        
        Args:
            cmd: 命令列表
            cwd: 工作目录
            
        Returns:
            str: 命令输出
        """
        try:
            env = os.environ.copy()
            env['HIBENCH_HOME'] = str(self.hibench_home)
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd) if cwd else None,
                env=env
            )
            
            if result.returncode != 0:
                self.logger.error(f"命令执行失败: {result.stderr}")
                return ""
            return result.stdout
        except Exception as e:
            self.logger.error(f"执行命令失败: {str(e)}")
            return ""
            
    def prepare_data(self, workload: str, scale: str) -> bool:
        """准备工作负载数据
        
        Args:
            workload: 工作负载名称
            scale: 数据规模
            
        Returns:
            bool: 是否成功
        """
        try:
            # 设置环境变量
            os.environ['WORKLOAD'] = workload
            os.environ['SCALE'] = scale
            
            # 执行准备脚本
            cmd = [
                str(self.hibench_home / 'bin' / 'workloads' / workload / 'prepare' / 'prepare.sh')
            ]
            output = self._execute_command(cmd)
            
            return 'ERROR' not in output.upper()
        except Exception as e:
            self.logger.error(f"准备数据失败: {str(e)}")
            return False
            
    def run_workload(self, workload: str, scale: str) -> bool:
        """运行工作负载
        
        Args:
            workload: 工作负载名称
            scale: 数据规模
            
        Returns:
            bool: 是否成功
        """
        try:
            # 设置环境变量
            os.environ['WORKLOAD'] = workload
            os.environ['SCALE'] = scale
            
            # 执行运行脚本
            cmd = [
                str(self.hibench_home / 'bin' / 'workloads' / workload / 'spark' / 'run.sh')
            ]
            output = self._execute_command(cmd)
            
            return 'ERROR' not in output.upper()
        except Exception as e:
            self.logger.error(f"运行工作负载失败: {str(e)}")
            return False
            
    def get_workload_metrics(self) -> Dict:
        """获取工作负载指标
        
        Returns:
            Dict: 工作负载指标
        """
        try:
            metrics = {}
            report_file = self.hibench_home / 'report' / 'hibench.report'
            
            if report_file.exists():
                with open(report_file, 'r') as f:
                    last_line = f.readlines()[-1]
                    fields = last_line.strip().split(',')
                    
                    if len(fields) >= 8:
                        metrics = {
                            'workload': fields[0],
                            'scale': fields[1],
                            'start_time': float(fields[2]),
                            'finish_time': float(fields[3]),
                            'duration': float(fields[4]),
                            'throughput': float(fields[5]),
                            'memory_bytes': int(fields[6]),
                            'vcores': int(fields[7])
                        }
                        
            return metrics
        except Exception as e:
            self.logger.error(f"获取工作负载指标失败: {str(e)}")
            return {}
            
    def get_workload_status(self) -> Dict:
        """获取工作负载状态
        
        Returns:
            Dict: 工作负载状态
        """
        try:
            status = {
                'running': False,
                'final_status': 'UNKNOWN'
            }
            
            # 检查是否有正在运行的Spark应用
            cmd = ['ps', '-ef']
            output = self._execute_command(cmd)
            
            if 'spark-submit' in output and self.config['workload'] in output:
                status['running'] = True
                status['final_status'] = 'RUNNING'
            else:
                # 检查最近的报告
                report_file = self.hibench_home / 'report' / 'hibench.report'
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        last_line = f.readlines()[-1]
                        if 'SUCCESS' in last_line:
                            status['final_status'] = 'SUCCEEDED'
                        elif 'FAILED' in last_line:
                            status['final_status'] = 'FAILED'
                            
            return status
        except Exception as e:
            self.logger.error(f"获取工作负载状态失败: {str(e)}")
            return {
                'running': False,
                'final_status': 'UNKNOWN'
            } 