# GPU环境完整实验指南

## 🎯 方案选择建议

### 💰 成本效益分析

| 方案 | 成本 | 时间限制 | GPU性能 | 推荐指数 |
|------|------|----------|---------|----------|
| **Google Colab** | 免费 | 12小时/次 | Tesla T4 | ⭐⭐⭐⭐⭐ |
| **Kaggle Notebooks** | 免费 | 30小时/周 | Tesla P100 | ⭐⭐⭐⭐⭐ |
| **阿里云ECS GPU** | ¥2-3/小时 | 无限制 | Tesla T4 | ⭐⭐⭐⭐ |
| **腾讯云GPU** | ¥8-12/小时 | 无限制 | Tesla V100 | ⭐⭐⭐ |

## 🆓 方案一：Google Colab（推荐新手）

### 步骤1：准备Colab环境
```python
# 在Colab新建笔记本，运行以下代码

# 1. 检查GPU
!nvidia-smi

# 2. 克隆项目
!git clone https://github.com/your-username/moni.git
%cd moni

# 3. 安装依赖
!pip install -r requirements.txt

# 4. 验证环境
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU名称: {torch.cuda.get_device_name(0)}")
```

### 步骤2：运行完整实验
```python
# 运行完整的故障注入实验
!python3 experiments/orchestrator.py

# 或者分步运行
!python3 experiments/baseline_run.py
!python3 experiments/fault_experiments.py --fault_type nan_loss
!python3 experiments/fault_experiments.py --fault_type oom
```

### 步骤3：下载结果
```python
# 打包实验结果
!tar -czf experiment_results.tar.gz monitoring_data/ fault_experiments/ tse_matrix/ experiment_reports/

# 下载到本地
from google.colab import files
files.download('experiment_results.tar.gz')
```

## 💰 方案二：阿里云ECS GPU实例

### 步骤1：创建GPU实例

1. **登录阿里云控制台**
   - 进入ECS控制台
   - 选择"创建实例"

2. **选择GPU实例规格**
   ```
   推荐配置：
   - 实例规格：ecs.gn6i-c4g1.xlarge
   - GPU：NVIDIA Tesla T4 (16GB)
   - CPU：4核
   - 内存：15GB
   - 系统盘：40GB SSD
   - 操作系统：Ubuntu 20.04
   ```

3. **配置网络和安全组**
   - 开放SSH端口(22)
   - 开放Jupyter端口(8888)（可选）

### 步骤2：连接并部署

```bash
# SSH连接到实例
ssh root@your-instance-ip

# 上传部署脚本
scp deploy_aliyun_gpu.sh root@your-instance-ip:~/

# 运行部署脚本
chmod +x deploy_aliyun_gpu.sh
./deploy_aliyun_gpu.sh
```

### 步骤3：上传项目代码

```bash
# 方法1：使用scp上传
scp -r /Users/bytedance/Downloads/trae_projects/moni root@your-instance-ip:~/

# 方法2：使用Git（推荐）
git clone https://github.com/your-username/moni.git
```

### 步骤4：运行实验

```bash
# 激活虚拟环境
source moni_env/bin/activate
cd moni

# 运行完整实验
python3 experiments/orchestrator.py

# 监控GPU使用情况
watch -n 1 nvidia-smi
```

## 🚀 当前CPU环境测试

在等待GPU资源的同时，您可以先运行我为您准备的CPU完整实验：

<function_calls>
<invoke name="RunCommand">
<parameter name="command">python3 run_cpu_experiment.py