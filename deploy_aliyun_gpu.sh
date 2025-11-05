#!/bin/bash
# 阿里云GPU实例部署脚本
# 适用于Ubuntu 20.04 + NVIDIA GPU环境

echo "🚀 开始在阿里云GPU实例上部署故障注入系统"
echo "================================================"

# 1. 系统更新
echo "📦 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 2. 安装Python和基础工具
echo "🐍 安装Python环境..."
sudo apt install -y python3 python3-pip python3-venv git wget curl

# 3. 检查GPU驱动
echo "🔍 检查GPU状态..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "❌ GPU驱动未正确安装，请检查实例配置"
    exit 1
fi

# 4. 安装CUDA工具包（如果需要）
echo "⚡ 检查CUDA环境..."
nvcc --version
if [ $? -ne 0 ]; then
    echo "📥 安装CUDA工具包..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
fi

# 5. 创建Python虚拟环境
echo "🏗️ 创建Python虚拟环境..."
python3 -m venv moni_env
source moni_env/bin/activate

# 6. 克隆项目（假设您已经上传到Git仓库）
echo "📥 下载项目代码..."
if [ ! -d "moni" ]; then
    # 如果您还没有Git仓库，可以使用scp上传
    echo "请将项目文件上传到当前目录"
    echo "或者从Git仓库克隆: git clone <your-repo-url> moni"
    # git clone <your-repo-url> moni
fi

cd moni

# 7. 安装Python依赖
echo "📦 安装Python依赖包..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate evaluate
pip install pandas numpy scipy scikit-learn
pip install psutil pynvml nvidia-ml-py3
pip install pyyaml loguru tqdm click
pip install matplotlib seaborn

# 8. 验证GPU环境
echo "🔍 验证GPU环境..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 9. 运行GPU环境测试
echo "🧪 运行GPU环境测试..."
python3 -c "
import torch
import transformers
print('✅ 所有依赖包安装成功')
print(f'✅ PyTorch GPU支持: {torch.cuda.is_available()}')
print(f'✅ Transformers版本: {transformers.__version__}')
"

echo "🎉 阿里云GPU环境部署完成！"
echo ""
echo "📋 下一步操作："
echo "1. 激活虚拟环境: source moni_env/bin/activate"
echo "2. 进入项目目录: cd moni"
echo "3. 运行完整实验: python3 experiments/orchestrator.py"
echo "4. 或运行基线实验: python3 experiments/baseline_run.py"
echo ""
echo "💡 实验建议："
echo "- Tesla T4 (16GB): 适合batch_size=16-32"
echo "- Tesla V100 (32GB): 适合batch_size=32-64"
echo "- 预计实验时间: 30-60分钟"
echo "- 预计费用: ¥2-5 (取决于实例规格和运行时间)"