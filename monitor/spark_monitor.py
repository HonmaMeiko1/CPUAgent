import json
import requests
from typing import Dict, List, Optional
import logging
from pathlib import Path
import subprocess
import os
import re

class SparkMonitor:
    def __init__(self, config: Dict):
        """
        初始化Spark监控器
        
        Args:
            config: Spark配置字典，包含必要的路径和设置
        """
        self.config = config
        self.spark_home = Path(config['home'])
        self.spark_master_url = config.get('master_url', 'http://localhost:8080')
        self.logger = logging.getLogger(__name__)
        self._validate_spark_installation()
        
    def _validate_spark_installation(self):
        """验证Spark安装"""
        try:
            # 检查SPARK_HOME环境变量
            if not os.environ.get('SPARK_HOME'):
                os.environ['SPARK_HOME'] = str(self.spark_home)
                
            # 检查spark-submit是否可用
            spark_submit = self.spark_home / 'bin' / 'spark-submit'
            if not spark_submit.exists():
                raise FileNotFoundError(f"找不到spark-submit: {spark_submit}")
                
            # 检查必要的目录
            for dir_name in ['conf', 'jars', 'bin']:
                if not (self.spark_home / dir_name).exists():
                    raise FileNotFoundError(f"找不到必要的Spark目录: {dir_name}")
                    
            # 设置执行权限
            (self.spark_home / 'bin' / 'spark-submit').chmod(0o755)
            (self.spark_home / 'bin' / 'spark-class').chmod(0o755)
            
        except Exception as e:
            self.logger.error(f"Spark安装验证失败: {str(e)}")
            raise
            
    def _execute_spark_command(self, cmd: List[str]) -> str:
        """执行Spark命令
        
        Args:
            cmd: 命令列表
            
        Returns:
            str: 命令输出
        """
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=os.environ
            )
            if result.returncode != 0:
                self.logger.error(f"Spark命令执行失败: {result.stderr}")
                return ""
            return result.stdout
        except Exception as e:
            self.logger.error(f"执行Spark命令失败: {str(e)}")
            return ""
            
    def get_running_applications(self) -> List[Dict]:
        """
        获取正在运行的Spark应用程序信息
        
        Returns:
            List[Dict]: 应用程序信息列表
        """
        try:
            # 首先尝试REST API
            try:
                response = requests.get(f"{self.spark_master_url}/api/v1/applications")
                return response.json()
            except Exception:
                pass
                
            # 如果REST API失败，使用spark-submit --status
            cmd = [
                str(self.spark_home / 'bin' / 'spark-submit'),
                '--status'
            ]
            output = self._execute_spark_command(cmd)
            
            # 解析输出
            apps = []
            for line in output.split('\n'):
                if 'application_' in line:
                    match = re.search(r'(application_\d+_\d+)\s+(\S+)\s+(\S+)', line)
                    if match:
                        apps.append({
                            'id': match.group(1),
                            'name': match.group(2),
                            'state': match.group(3)
                        })
            return apps
            
        except Exception as e:
            self.logger.error(f"获取运行中的应用失败: {str(e)}")
            return []
            
    def get_application_details(self, app_id: str) -> Optional[Dict]:
        """
        获取特定应用程序的详细信息
        
        Args:
            app_id: 应用程序ID
            
        Returns:
            Optional[Dict]: 应用程序详细信息
        """
        try:
            # 首先尝试REST API
            try:
                response = requests.get(f"{self.spark_master_url}/api/v1/applications/{app_id}")
                return response.json()
            except Exception:
                pass
                
            # 如果REST API失败，使用spark-submit --status
            cmd = [
                str(self.spark_home / 'bin' / 'spark-submit'),
                '--status',
                app_id
            ]
            output = self._execute_spark_command(cmd)
            
            # 解析输出
            details = {}
            for line in output.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    details[key.strip()] = value.strip()
            return details
            
        except Exception as e:
            self.logger.error(f"获取应用 {app_id} 详情失败: {str(e)}")
            return None
            
    def get_executor_metrics(self, app_id: str) -> List[Dict]:
        """
        获取应用程序的执行器指标
        
        Args:
            app_id: 应用程序ID
            
        Returns:
            List[Dict]: 执行器指标列表
        """
        try:
            # 首先尝试REST API
            try:
                response = requests.get(f"{self.spark_master_url}/api/v1/applications/{app_id}/executors")
                return response.json()
            except Exception:
                pass
                
            # 如果REST API失败，从Spark日志获取信息
            log_dir = Path(self.config.get('log_dir', '/tmp/spark-events'))
            if log_dir.exists():
                for log_file in log_dir.glob(f"*{app_id}*"):
                    with open(log_file, 'r') as f:
                        executors = []
                        for line in f:
                            if 'Executor added' in line:
                                executor = self._parse_executor_log(line)
                                if executor:
                                    executors.append(executor)
                        return executors
            return []
            
        except Exception as e:
            self.logger.error(f"获取执行器指标失败: {str(e)}")
            return []
            
    def _parse_executor_log(self, log_line: str) -> Optional[Dict]:
        """解析执行器日志行
        
        Args:
            log_line: 日志行
            
        Returns:
            Optional[Dict]: 执行器信息
        """
        try:
            # 示例日志行：Executor added: app-20231225-1 with 4 cores, 8.0 GB memory
            match = re.search(r'Executor added: (.*) with (\d+) cores, ([\d.]+) ([GM])B memory', log_line)
            if match:
                memory_value = float(match.group(3))
                memory_unit = match.group(4)
                memory_mb = memory_value * (1024 if memory_unit == 'G' else 1)
                
                return {
                    'id': match.group(1),
                    'cores': int(match.group(2)),
                    'memory': memory_mb
                }
        except Exception:
            pass
        return None
        
    def get_stage_metrics(self, app_id: str) -> List[Dict]:
        """
        获取应用程序的阶段指标
        
        Args:
            app_id: 应用程序ID
            
        Returns:
            List[Dict]: 阶段指标列表
        """
        try:
            response = requests.get(f"{self.spark_master_url}/api/v1/applications/{app_id}/stages")
            return response.json()
        except Exception as e:
            self.logger.error(f"获取阶段指标失败: {str(e)}")
            return []
            
    def get_job_metrics(self, app_id: str) -> List[Dict]:
        """
        获取应用程序的作业指标
        
        Args:
            app_id: 应用程序ID
            
        Returns:
            List[Dict]: 作业指标列表
        """
        try:
            response = requests.get(f"{self.spark_master_url}/api/v1/applications/{app_id}/jobs")
            return response.json()
        except Exception as e:
            self.logger.error(f"获取作业指标失败: {str(e)}")
            return []
            
    def get_cluster_metrics(self) -> Dict:
        """
        获取集群整体指标
        
        Returns:
            Dict: 集群指标
        """
        try:
            # 首先尝试REST API
            try:
                response = requests.get(f"{self.spark_master_url}/api/v1/cluster")
                return response.json()
            except Exception:
                pass
                
            # 如果REST API失败，使用spark-submit --status
            cmd = [
                str(self.spark_home / 'bin' / 'spark-submit'),
                '--status',
                '--master',
                'status'
            ]
            output = self._execute_spark_command(cmd)
            
            # 解析输出
            metrics = {
                'workers': 0,
                'cores': 0,
                'memory': 0
            }
            
            for line in output.split('\n'):
                if 'Workers:' in line:
                    metrics['workers'] = int(re.search(r'Workers:\s*(\d+)', line).group(1))
                elif 'Cores:' in line:
                    metrics['cores'] = int(re.search(r'Cores:\s*(\d+)', line).group(1))
                elif 'Memory:' in line:
                    memory_match = re.search(r'Memory:\s*([\d.]+)\s*([GM])B', line)
                    if memory_match:
                        value = float(memory_match.group(1))
                        unit = memory_match.group(2)
                        metrics['memory'] = value * (1024 if unit == 'G' else 1)
                        
            return metrics
            
        except Exception as e:
            self.logger.error(f"获取集群指标失败: {str(e)}")
            return {} 