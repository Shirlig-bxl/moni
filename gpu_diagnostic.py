#!/usr/bin/env python3
"""
GPU诊断脚本 - 检查GPU使用情况
专门诊断为什么GPU利用率为0%
"""

import torch
import time
import psutil
from datetime import datetime

def check_gpu_basic():
    """基础GPU检查"""
    print("=" * 60)
    print("🔍 基础GPU环境检查")
    print("=" * 60)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - 总内存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - 计算能力: {props.major}.{props.minor}")
        
        print(f"当前GPU设备: {torch.cuda.current_device()}")
        return True
    else:
        print("❌ CUDA不可用！")
        print("💡 请检查:")
        print("   1. Colab运行时是否设置为GPU")
        print("   2. 运行时 → 更改运行时类型 → 硬件加速器 → GPU")
        return False

def test_gpu_computation():
    """测试GPU计算能力"""
    print("\n" + "=" * 60)
    print("🧪 GPU计算能力测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ 跳过GPU测试 - CUDA不可用")
        return False
    
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 测试1: 简单矩阵运算
    print("\n📊 测试1: 矩阵运算")
    try:
        # 创建大矩阵进行计算
        size = 2000
        print(f"创建 {size}x{size} 矩阵...")
        
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print("开始矩阵乘法...")
        start_time = time.time()
        
        # 执行多次计算以产生GPU负载
        for i in range(10):
            c = torch.matmul(a, b)
            if i % 2 == 0:
                print(f"  计算轮次 {i+1}/10...")
            time.sleep(0.5)  # 给监控时间观察
        
        end_time = time.time()
        print(f"✅ 矩阵运算完成，耗时: {end_time - start_time:.2f}秒")
        print(f"结果矩阵形状: {c.shape}")
        print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
        return False

def test_model_gpu_usage():
    """测试模型GPU使用"""
    print("\n" + "=" * 60)
    print("🤖 模型GPU使用测试")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ 跳过模型测试 - CUDA不可用")
        return False
    
    try:
        device = torch.device('cuda')
        print(f"使用设备: {device}")
        
        # 创建一个简单的神经网络
        print("创建神经网络...")
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        ).to(device)
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print(f"模型设备: {next(model.parameters()).device}")
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("\n开始训练循环...")
        batch_size = 64
        
        for epoch in range(5):
            print(f"Epoch {epoch+1}/5")
            
            for step in range(20):  # 20个步骤
                # 创建随机数据
                inputs = torch.randn(batch_size, 1000, device=device)
                targets = torch.randint(0, 2, (batch_size,), device=device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if step % 5 == 0:
                    print(f"  Step {step+1}/20, Loss: {loss.item():.4f}")
                    print(f"  GPU内存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                
                time.sleep(0.2)  # 给监控时间观察
        
        print("✅ 模型训练测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 模型GPU测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_gpu_during_test():
    """在测试期间监控GPU"""
    print("\n" + "=" * 60)
    print("📊 GPU监控测试")
    print("=" * 60)
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("开始GPU监控（10秒）...")
        
        for i in range(10):
            # 获取GPU利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取GPU内存
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 获取GPU温度
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            print(f"[{i+1:2d}/10] GPU利用率: {util.gpu:3d}%, "
                  f"内存: {mem_info.used//1024//1024:4d}/{mem_info.total//1024//1024:4d}MB "
                  f"({mem_info.used/mem_info.total*100:5.1f}%), "
                  f"温度: {temp:2d}°C")
            
            time.sleep(1)
        
        return True
        
    except ImportError:
        print("❌ pynvml未安装，无法监控GPU")
        print("💡 安装命令: pip install nvidia-ml-py3")
        return False
    except Exception as e:
        print(f"❌ GPU监控失败: {e}")
        return False

def comprehensive_gpu_test():
    """综合GPU测试"""
    print("🧪 GPU诊断 - 综合测试")
    print("这个测试将帮助诊断为什么GPU利用率为0%")
    print("=" * 60)
    
    # 1. 基础检查
    gpu_available = check_gpu_basic()
    
    if not gpu_available:
        print("\n❌ GPU不可用，请先解决GPU环境问题")
        return
    
    # 2. 启动GPU监控（后台）
    import threading
    
    def background_monitor():
        time.sleep(2)  # 等待测试开始
        monitor_gpu_during_test()
    
    monitor_thread = threading.Thread(target=background_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 3. GPU计算测试
    print("\n🚀 开始GPU负载测试...")
    time.sleep(1)
    
    compute_success = test_gpu_computation()
    
    # 4. 模型测试
    model_success = test_model_gpu_usage()
    
    # 等待监控完成
    monitor_thread.join(timeout=15)
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 诊断总结")
    print("=" * 60)
    
    print(f"✅ GPU环境: {'正常' if gpu_available else '异常'}")
    print(f"✅ GPU计算: {'正常' if compute_success else '异常'}")
    print(f"✅ 模型训练: {'正常' if model_success else '异常'}")
    
    if gpu_available and compute_success and model_success:
        print("\n🎉 GPU功能正常！")
        print("💡 如果您的训练仍然GPU利用率为0%，可能的原因:")
        print("   1. 训练数据太小，GPU处理太快")
        print("   2. batch_size太小，无法充分利用GPU")
        print("   3. 模型没有正确移动到GPU")
        print("   4. 数据没有移动到GPU")
    else:
        print("\n⚠️ 发现GPU问题，请检查上述失败的测试项")

if __name__ == "__main__":
    comprehensive_gpu_test()