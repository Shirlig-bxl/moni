#!/usr/bin/env python3
"""
GPU监控脚本 - 实时监控GPU使用情况
"""

import time
import pynvml
from datetime import datetime

def monitor_gpu():
    """监控GPU使用情况"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        print("🔍 开始GPU监控...")
        print("=" * 80)
        
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
            
            # 格式化输出
            gpu_util_bar = "█" * (util.gpu // 5) + "░" * (20 - util.gpu // 5)
            mem_percent = mem_info.used / mem_info.total * 100
            mem_bar = "█" * int(mem_percent // 5) + "░" * (20 - int(mem_percent // 5))
            
            print(f"[{timestamp}] GPU: {util.gpu:3d}% |{gpu_util_bar}| "
                  f"内存: {mem_info.used//1024//1024:4d}MB/{mem_info.total//1024//1024:4d}MB "
                  f"|{mem_bar}| ({mem_percent:5.1f}%) "
                  f"温度: {temp:2d}°C 功耗: {power:5.1f}W")
            
            time.sleep(1)
            
    except ImportError:
        print("❌ nvidia-ml-py3未安装")
        print("💡 安装命令: pip install nvidia-ml-py3")
    except Exception as e:
        print(f"❌ GPU监控错误: {e}")
        print("💡 请确保:")
        print("   1. 运行在GPU环境中")
        print("   2. 已安装nvidia-ml-py3")
        print("   3. GPU驱动正常")

if __name__ == "__main__":
    monitor_gpu()