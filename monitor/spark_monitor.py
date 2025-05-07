import json
import requests
from typing import Dict, List, Optional
import logging

class SparkMonitor:
    def __init__(self, spark_master_url: str):
        """
        初始化Spark监控器
        
        Args:
            spark_master_url: Spark Master的URL地址
        """
        self.spark_master_url = spark_master_url
        self.logger = logging.getLogger(__name__)
        
    def get_running_applications(self) -> List[Dict]:
        """
        获取正在运行的Spark应用程序信息
        
        Returns:
            List[Dict]: 应用程序信息列表
        """
        try:
            response = requests.get(f"{self.spark_master_url}/api/v1/applications")
            return response.json()
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
            response = requests.get(f"{self.spark_master_url}/api/v1/applications/{app_id}")
            return response.json()
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
            response = requests.get(f"{self.spark_master_url}/api/v1/applications/{app_id}/executors")
            return response.json()
        except Exception as e:
            self.logger.error(f"获取执行器指标失败: {str(e)}")
            return []
            
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
            response = requests.get(f"{self.spark_master_url}/api/v1/cluster")
            return response.json()
        except Exception as e:
            self.logger.error(f"获取集群指标失败: {str(e)}")
            return {} 