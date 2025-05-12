# CPUAgent

基于深度强化学习的Linux系统CPU能源优化智能体。该系统使用PPO（Proximal Policy Optimization）算法来优化Spark工作负载的执行，通过动态调整CPU频率和内存电源模式来平衡性能和能源消耗。

## 功能特点

- 支持多种Spark机器学习工作负载（KMeans、LR、Random Forest等）
- 使用PPO算法进行CPU频率和内存模式的动态调整
- 实时监控系统资源使用和能源消耗
- 支持Linux系统CPU频率和内存电源模式管理
- 提供详细的性能指标和能源消耗统计
- 支持优雅退出和资源清理
- 自动保存训练/测试配置和结果

## 系统要求

- Linux操作系统（支持cpufreq和内存电源管理）
- Python 3.8+
- PyTorch 1.8+
- Apache Spark 3.0+
- Apache Hadoop 3.0+
- HiBench 7.0+

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/CPUAgent.git
cd CPUAgent
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境：
   - 修改 `config/config.yaml` 中的集群配置
   - 确保HiBench正确安装并配置
   - 确保有修改CPU频率和内存电源模式的权限

## 使用方法

### 训练模式

```bash
python run.py --mode train --config config/config.yaml
```

### 测试模式

```bash
python run.py --mode test --config config/config.yaml --model_path models/final_model.pt
```

## 配置说明

主要配置项（config/config.yaml）：

- `spark`: Spark集群配置
  - `master_url`: Spark主节点URL
  - `nodes`: 集群节点配置
  - `hibench`: HiBench配置

- `training`: 训练配置
  - `n_episodes`: 训练回合数
  - `max_steps`: 每回合最大步数
  - `save_interval`: 模型保存间隔
  - `hibench_training`: HiBench训练工作负载配置

- `testing`: 测试配置
  - `n_episodes`: 测试回合数
  - `max_steps`: 每回合最大步数
  - `hibench_testing`: HiBench测试工作负载配置

- `ppo`: PPO智能体配置
  - `network`: 网络结构配置
  - `learning_rate`: 学习率
  - `gamma`: 折扣因子
  - `clip_ratio`: PPO裁剪参数

- `environment`: 环境配置
  - `state_space`: 状态空间配置
  - `action_space`: 动作空间配置
  - `reward`: 奖励配置

- `control`: 控制配置
  - `cpu`: CPU频率控制参数
  - `memory`: 内存电源模式配置

## 项目结构

```
CPUAgent/
├── agent/                 # 智能体实现
│   ├── agents/           # 智能体类
│   │   ├── base_agent.py # 基础智能体
│   │   └── ppo_agent.py  # PPO智能体
│   ├── trainers/         # 训练器
│   │   ├── base_trainer.py # 基础训练器
│   │   └── ppo_trainer.py  # PPO训练器
│   ├── networks/         # 神经网络
│   └── utils/            # 智能体工具
├── env/                   # 环境实现
│   ├── spark_env.py      # Spark环境
│   ├── state_space.py    # 状态空间
│   ├── action_space.py   # 动作空间
│   └── reward.py         # 奖励计算
├── monitor/              # 监控实现
│   ├── spark_monitor.py  # Spark监控
│   ├── system_monitor.py # 系统监控
│   └── hibench_monitor.py # HiBench监控
├── control/              # 控制实现
│   ├── cpu_control.py    # CPU控制
│   └── memory_control.py # 内存控制
├── utils/                # 工具函数
│   └── plot_utils.py     # 绘图工具
├── config/               # 配置文件
│   └── config.yaml      # 主配置文件
├── models/               # 模型保存目录
├── results/              # 结果保存目录
├── logs/                 # 日志保存目录
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── run.py               # 主运行脚本
└── requirements.txt      # 依赖列表
```

## 注意事项

1. 系统要求：
   - 确保Linux系统支持cpufreq调频
   - 确保有修改CPU频率和内存电源模式的权限
   - 推荐使用支持RAPL（Running Average Power Limit）的CPU

2. 运行建议：
   - 首次运行时建议使用较小的回合数进行测试
   - 监控系统资源使用情况，避免过度消耗
   - 使用`Ctrl+C`可以退出训练/测试过程

3. 性能优化：
   - 根据实际硬件配置调整CPU频率范围
   - 适当调整奖励权重以平衡性能和能耗
   - 可以通过配置文件调整网络结构和训练参数

4. 数据收集：
   - 所有训练和测试结果会自动保存到results目录
   - 日志文件包含详细的运行信息和错误记录
   - 可以使用plot_utils进行结果可视化