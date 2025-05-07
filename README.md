# DRL Energy Saving

基于深度强化学习的Spark集群能源优化系统。该系统使用PPO（Proximal Policy Optimization）算法来优化Spark工作负载的执行，通过动态调整CPU频率来平衡性能和能源消耗。

## 功能特点

- 支持多种Spark工作负载（WordCount、Sort、TeraSort等）
- 使用PPO算法进行策略优化
- 实时监控系统资源使用情况
- 动态调整CPU频率以优化能源效率
- 支持训练和推理两种模式
- 提供详细的性能指标和能源消耗统计

## 系统要求

- Python 3.8+
- PyTorch 1.8+
- Apache Spark 3.0+
- Apache Hadoop 3.0+
- HiBench 7.0+
- Linux操作系统（支持CPU频率调节）

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/HonmaMeiko1/DRL_energy_saving.git
cd DRL_energy_saving
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境：
   - 修改 `config/config.yaml` 中的路径配置
   - 确保HiBench、Spark和Hadoop正确安装并配置

## 使用方法

### 训练模式

```bash
python run.py --mode train --config config/config.yaml
```

### 测试模式

```bash
python run.py --mode test --config config/config.yaml --model_path models/ppo_final.pt
```

## 配置说明

主要配置项（config/config.yaml）：

- `training`: 训练相关配置
  - `episodes`: 训练回合数
  - `max_steps`: 每回合最大步数
  - `learning_rate`: 学习率
  - `gamma`: 折扣因子
  - `clip_ratio`: PPO裁剪比例

- `spark`: Spark配置
  - `executor_cores`: 执行器核心数
  - `executor_memory`: 执行器内存
  - `driver_memory`: 驱动器内存

- `monitoring`: 监控配置
  - `metrics_interval`: 指标收集间隔
  - `cpu_metrics`: 是否收集CPU指标
  - `memory_metrics`: 是否收集内存指标
  - `power_metrics`: 是否收集功耗指标

- `control`: 控制配置
  - `cpu_freq_min`: 最小CPU频率
  - `cpu_freq_max`: 最大CPU频率
  - `memory_limit_min`: 最小内存限制
  - `memory_limit_max`: 最大内存限制

## 项目结构

```
DRL_energy_saving/
├── agent/                 # 智能体相关代码
│   ├── actor_critic.py    # Actor-Critic网络
│   └── ppo_trainer.py     # PPO训练器
├── env/                   # 环境相关代码
│   ├── spark_env.py       # Spark环境
│   ├── state_space.py     # 状态空间
│   ├── action_space.py    # 动作空间
│   └── reward.py          # 奖励计算
├── monitor/              # 监控相关代码
│   ├── spark_monitor.py   # Spark监控
│   ├── system_monitor.py  # 系统监控
│   └── hibench_monitor.py # HiBench监控
├── control/              # 控制相关代码
│   ├── cpu_control.py     # CPU控制
│   └── memory_control.py  # 内存控制
├── utils/                # 工具函数
├── config/               # 配置文件
├── models/               # 模型保存目录
├── results/              # 结果保存目录
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── run.py               # 主运行脚本
└── requirements.txt      # 依赖列表
```

## 注意事项

1. 确保系统支持CPU频率调节
2. 需要root权限来修改CPU频率
3. 建议在测试环境中先进行小规模测试
4. 监控系统资源使用情况，避免过度消耗