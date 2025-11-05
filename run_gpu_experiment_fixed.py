#!/usr/bin/env python3
"""
修复版GPU实验脚本 - 确保GPU得到充分利用
专门解决GPU利用率为0%的问题
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

def check_and_fix_gpu_environment():
    """检查并修复GPU环境"""
    print("🔧 检查并修复GPU环境...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        if not torch.cuda.is_available():
            print("❌ CUDA不可用！")
            print("💡 请在Google Colab中:")
            print("   1. 运行时 → 更改运行时类型")
            print("   2. 硬件加速器 → GPU")
            print("   3. 保存并重新连接")
            return False
        
        print(f"✅ CUDA版本: {torch.version.cuda}")
        print(f"✅ GPU数量: {torch.cuda.device_count()}")
        print(f"✅ GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 测试GPU基本功能
        device = torch.device('cuda')
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"✅ GPU基础计算测试通过")
        
        return True
        
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
        return False

def create_intensive_gpu_training_script():
    """创建GPU密集型训练脚本"""
    
    training_script = '''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from datetime import datetime

def log_with_timestamp(message):
    """带时间戳的日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def create_large_model():
    """创建一个较大的模型以充分利用GPU"""
    model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)  # 10分类
    )
    return model

def main():
    log_with_timestamp("开始GPU密集型训练")
    
    # 强制使用GPU
    if not torch.cuda.is_available():
        log_with_timestamp("ERROR: CUDA不可用，无法继续")
        return
    
    device = torch.device('cuda')
    log_with_timestamp(f"使用设备: {device}")
    log_with_timestamp(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 创建大模型
    log_with_timestamp("创建大型神经网络...")
    model = create_large_model().to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    log_with_timestamp(f"模型参数总数: {total_params:,}")
    
    # 验证模型在GPU上
    model_device = next(model.parameters()).device
    log_with_timestamp(f"模型设备: {model_device}")
    
    # 创建大批量数据
    batch_size = 256  # 增大batch size
    input_size = 2048
    num_batches = 100
    
    log_with_timestamp(f"批次大小: {batch_size}")
    log_with_timestamp(f"输入维度: {input_size}")
    log_with_timestamp(f"总批次数: {num_batches}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    log_with_timestamp("开始训练循环...")
    
    # 训练循环
    for epoch in range(5):  # 5个epoch
        log_with_timestamp(f"Epoch {epoch+1}/5")
        
        epoch_start_time = time.time()
        total_loss = 0
        
        for batch_idx in range(num_batches):
            # 创建随机数据（在GPU上）
            inputs = torch.randn(batch_size, input_size, device=device)
            targets = torch.randint(0, 10, (batch_size,), device=device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每10个batch输出一次
            if batch_idx % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                log_with_timestamp(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, GPU内存: {gpu_memory:.1f}MB")
            
            # 添加一些额外的GPU计算来增加负载
            if batch_idx % 5 == 0:
                # 额外的矩阵运算
                extra_computation = torch.matmul(
                    torch.randn(512, 512, device=device),
                    torch.randn(512, 512, device=device)
                )
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        log_with_timestamp(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}秒")
        
        # 显示GPU内存使用情况
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
        gpu_cached = torch.cuda.memory_reserved() / 1024**2
        log_with_timestamp(f"GPU内存 - 已分配: {gpu_memory:.1f}MB, 已缓存: {gpu_cached:.1f}MB")
    
    log_with_timestamp("训练完成！")
    
    # 最终GPU状态
    final_memory = torch.cuda.memory_allocated() / 1024**2
    log_with_timestamp(f"最终GPU内存使用: {final_memory:.1f}MB")

if __name__ == "__main__":
    main()
'''
    
    return training_script

def run_gpu_monitoring():
    """运行GPU监控"""
    monitoring_script = '''
import time
import pynvml
from datetime import datetime

def monitor_gpu():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("开始GPU监控...")
        
        for i in range(120):  # 监控2分钟
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # GPU利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # GPU内存
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # GPU温度
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # GPU功耗
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            except:
                power = 0
            
            print(f"[{timestamp}] GPU: {util.gpu:3d}% | "
                  f"内存: {mem_info.used//1024//1024:4d}MB/{mem_info.total//1024//1024:4d}MB "
                  f"({mem_info.used/mem_info.total*100:5.1f}%) | "
                  f"温度: {temp:2d}°C | "
                  f"功耗: {power:5.1f}W")
            
            time.sleep(1)
            
    except Exception as e:
        print(f"GPU监控错误: {e}")

if __name__ == "__main__":
    monitor_gpu()
'''
    
    return monitoring_script

def main():
    """主函数"""
    print("🚀 修复版GPU实验 - 解决GPU利用率0%问题")
    print("=" * 60)
    
    # 1. 检查GPU环境
    if not check_and_fix_gpu_environment():
        print("❌ GPU环境检查失败，请先解决GPU问题")
        return
    
    # 2. 创建训练脚本
    print("📝 创建GPU密集型训练脚本...")
    training_script = create_intensive_gpu_training_script()
    
    with open("intensive_gpu_training.py", "w") as f:
        f.write(training_script)
    
    # 3. 创建监控脚本
    print("📊 创建GPU监控脚本...")
    monitoring_script = run_gpu_monitoring()
    
    with open("gpu_monitor.py", "w") as f:
        f.write(monitoring_script)
    
    print("✅ 脚本创建完成")
    print("\n" + "=" * 60)
    print("🎯 使用说明")
    print("=" * 60)
    print("在Google Colab中运行以下命令:")
    print()
    print("# 1. 首先运行GPU诊断")
    print("!python gpu_diagnostic.py")
    print()
    print("# 2. 在一个cell中启动GPU监控")
    print("!python gpu_monitor.py &")
    print()
    print("# 3. 在另一个cell中运行GPU训练")
    print("!python intensive_gpu_training.py")
    print()
    print("💡 这样您应该能看到GPU利用率上升到50-90%")
    print("=" * 60)

if __name__ == "__main__":
    main()