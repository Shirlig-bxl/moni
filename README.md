# 主动故障注入与多源异常数据收集系统 (MONI)

## 项目概述
本项目实现了一个完整的"观测器-执行器-注入器"系统，用于主动制造并捕获多源异常，生成带有精确 Ground Truth 标注的数据集，用于训练和验证 TSE-Matrix 和 MLH-AD 框架。

## 系统架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   观测器模块     │    │   执行器模块     │    │   注入器模块     │
│   (Observer)    │    │   (Executor)    │    │   (Injector)    │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • GPU监控       │    │ • BERT训练      │    │ • 故障注入控制   │
│ • 系统监控      │◄──►│ • 日志生成      │◄──►│ • 时间调度      │
│ • 日志解析      │    │ • 指标输出      │    │ • 资源压力      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │      数据聚合模块        │
                    │     (Aggregator)       │
                    ├─────────────────────────┤
                    │ • 时间戳对齐            │
                    │ • 多源数据融合          │
                    │ • TSE-Matrix构建       │
                    │ • Ground Truth标注     │
                    └─────────────────────────┘
```

## 项目结构
```
moni/
├── src/
│   ├── executor/           # 执行器模块
│   │   ├── train.py       # 主训练脚本 (BERT-IMDB微调)
│   │   └── config.py      # 训练配置管理
│   ├── observer/          # 观测器模块
│   │   ├── gpu_monitor.py # GPU指标监控
│   │   ├── system_monitor.py # 系统指标监控
│   │   └── log_parser.py  # 训练日志解析器
│   ├── injector/          # 注入器模块
│   │   ├── fault_injector.py # 故障注入控制器
│   │   ├── io_stress.py   # I/O压力测试
│   │   └── resource_competitor.py # 资源争用模拟
│   ├── aggregator/        # 数据聚合模块
│   │   ├── data_aggregator.py # 多源数据聚合器
│   │   └── tse_matrix_builder.py # TSE矩阵构建器
│   └── utils/             # 工具模块
│       └── config_loader.py # 配置加载器
├── experiments/           # 实验脚本
│   ├── baseline_run.py    # 基线运行脚本
│   ├── fault_experiments.py # 故障注入实验
│   └── orchestrator.py    # 实验编排器
├── configs/               # 配置文件
│   ├── training_config.yaml # 训练配置
│   ├── fault_injection_config.yaml # 故障注入配置
│   └── monitoring_config.yaml # 监控配置
├── example_usage.py       # 使用示例
├── requirements.txt       # 依赖包
└── setup.py              # 安装脚本
```

## 支持的故障类型
1. **NaN Loss (梯度爆炸)** - 学习率过大导致的训练不稳定
2. **OOM (显存溢出)** - 批次大小过大导致的显存不足
3. **模型不收敛** - 超参数设置不当导致的收敛失败
4. **I/O 瓶颈** - 磁盘读写压力导致的训练延迟
5. **资源争用** - 多进程竞争GPU资源
6. **进程终止** - 模拟调度器强制终止训练任务

## 数据收集维度
- **GPU指标**: 利用率、显存使用、显存总量、温度、功耗
- **系统指标**: CPU使用率、内存使用率、磁盘I/O、网络I/O、负载均衡
- **训练指标**: Loss、Accuracy、学习率、训练步数、Epoch
- **事件指标**: OOM事件、NaN事件、错误事件、警告事件

## 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository_url>
cd moni

# 安装依赖
pip install -r requirements.txt

# 查看系统演示
python3 example_usage.py
```

### 2. 运行实验

#### 方式一：完整自动化流程
```bash
# 运行完整的实验流程（推荐）
python3 experiments/orchestrator.py
```

#### 方式二：分步执行
```bash
# 1. 运行基线实验
python3 experiments/baseline_run.py

# 2. 运行特定故障实验
python3 experiments/fault_experiments.py --fault_type nan_loss
python3 experiments/fault_experiments.py --fault_type oom
python3 experiments/fault_experiments.py --fault_type io_bottleneck

# 3. 数据聚合和TSE-Matrix构建（自动完成）
```

### 3. 配置自定义
```bash
# 编辑训练配置
vim configs/training_config.yaml

# 编辑故障注入配置
vim configs/fault_injection_config.yaml

# 编辑监控配置
vim configs/monitoring_config.yaml
```

## 输出数据格式

### 监控数据
- `monitoring_data/gpu_metrics.csv` - GPU指标时序数据
- `monitoring_data/system_metrics.csv` - 系统指标时序数据
- `monitoring_data/training_metrics.csv` - 训练指标时序数据

### 实验结果
- `fault_experiments/` - 各类故障实验的原始数据
- `tse_matrix/` - TSE-Matrix格式的聚合数据
- `experiment_reports/` - 实验报告和元数据

### TSE-Matrix格式
最终生成的TSE-Matrix包含：
- **时间戳对齐**: 1秒粒度的时间戳对齐
- **多源指标**: GPU、系统、训练指标的融合
- **Ground Truth标注**: 精确的异常时间窗口标记
- **特征工程**: 移动平均、标准差、变化率等衍生特征
- **标准格式**: 支持TSE-Matrix和MLH-AD框架的数据格式

## 技术特性

### 高精度时间同步
- 所有监控组件使用统一的1秒采样间隔
- 基于系统时间戳的精确对齐
- 支持亚秒级的事件标注

### 多源数据融合
- GPU指标通过nvidia-smi和pynvml采集
- 系统指标通过psutil库采集
- 训练指标通过日志解析提取
- 事件数据通过正则表达式识别

### 故障注入精确控制
- 基于训练步数的精确注入
- 基于时间的定时注入
- 支持故障持续时间控制
- 可配置的故障强度参数

### 可扩展架构
- 模块化设计，易于添加新的故障类型
- 配置驱动，支持灵活的实验设计
- 标准化接口，便于集成其他ML框架

## 使用场景

1. **异常检测算法验证** - 为TSE-Matrix和MLH-AD提供标准测试数据
2. **故障诊断研究** - 分析不同故障模式的多源指标特征
3. **系统监控优化** - 评估监控系统的异常检测能力
4. **ML训练稳定性研究** - 研究各类故障对训练过程的影响

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目。在贡献代码前，请确保：
1. 代码符合项目的编码规范
2. 添加适当的测试用例
3. 更新相关文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。