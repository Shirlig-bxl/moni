# 主动故障注入与多源异常数据收集系统 (MONI)

## 🎯 项目概述
完整的"观测器-执行器-注入器"系统，用于主动制造并捕获多源异常，生成带有精确Ground Truth标注的数据集，用于训练和验证TSE-Matrix和MLH-AD框架。

## 🏗️ 系统架构
```
观测器 (Observer) ←→ 执行器 (Executor) ←→ 注入器 (Injector)
       ↓                    ↓                    ↓
              数据聚合模块 (Aggregator)
                     ↓
              TSE-Matrix + Ground Truth
```

## 📁 项目结构
```
moni/
├── src/                    # 核心源代码
│   ├── observer/          # 观测器：GPU/系统监控、日志解析
│   ├── executor/          # 执行器：BERT训练脚本
│   ├── injector/          # 注入器：故障注入控制
│   ├── aggregator/        # 聚合器：数据融合、TSE-Matrix构建
│   └── utils/             # 工具：配置加载器
├── configs/               # YAML配置文件
├── experiments/           # 实验编排脚本
├── run_local_experiment.py   # 本地CPU实验（推荐）
└── run_gpu_experiment.py     # GPU实验（Colab专用）
```

## 🚀 快速开始

### 方案1：本地运行（推荐）
```bash
# 克隆项目
git clone https://github.com/Shirlig-bxl/moni.git
cd moni

# 安装依赖
pip install pandas numpy psutil

# 运行完整实验
python3 run_local_experiment.py
```

### 方案2：Google Colab GPU实验
```python
# 在Colab中运行
!git clone https://github.com/Shirlig-bxl/moni.git
%cd moni
!python3 run_gpu_experiment.py
```

## 📊 支持的故障类型
1. **NaN Loss** - 梯度爆炸导致的训练不稳定
2. **I/O瓶颈** - 数据加载延迟
3. **资源争用** - CPU/GPU资源竞争

## 📈 输出数据
- **系统监控数据** - CPU、内存、磁盘使用率
- **训练指标数据** - Loss、Accuracy、学习率
- **聚合数据集** - 时间对齐的多源数据
- **Ground Truth标注** - 精确的异常时间窗口标记

## 🎯 研究应用
- TSE-Matrix框架验证
- MLH-AD算法训练
- 异常检测研究
- 系统监控优化

## 📝 许可证
MIT License