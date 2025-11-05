"""
GPU监控模块
使用nvidia-ml-py3监控GPU指标，每秒记录一次数据
"""

import time
import csv
import os
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
import threading
import argparse

try:
    import nvidia_ml_py as pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    try:
        import pynvml
        PYNVML_AVAILABLE = True
    except ImportError:
        PYNVML_AVAILABLE = False
        print("警告: nvidia-ml-py和pynvml都不可用，将尝试使用nvidia-smi命令")

import subprocess
import json


class GPUMonitor:
    """GPU监控器类"""
    
    def __init__(self, output_file: str = "gpu_metrics.csv", interval: int = 1):
        self.output_file = output_file
        self.interval = interval
        self.running = False
        self.thread = None
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        # 初始化NVIDIA ML
        self.use_pynvml = PYNVML_AVAILABLE
        if self.use_pynvml:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                print(f"检测到 {self.device_count} 个GPU设备")
            except Exception as e:
                print(f"PYNVML初始化失败: {e}")
                self.use_pynvml = False
        
        if not self.use_pynvml:
            print("使用nvidia-smi命令进行监控")
    
    def get_gpu_info_pynvml(self) -> List[Dict[str, Any]]:
        """使用pynvml获取GPU信息"""
        gpu_info = []
        
        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # 获取基本信息
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # 获取内存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // 1024 // 1024  # MB
                memory_used = mem_info.used // 1024 // 1024   # MB
                memory_free = mem_info.free // 1024 // 1024   # MB
                memory_percent = (memory_used / memory_total) * 100
                
                # 获取利用率
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    memory_util = util.memory
                except:
                    gpu_util = 0
                    memory_util = 0
                
                # 获取温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0
                
                # 获取功耗
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # W
                except:
                    power = 0
                
                gpu_info.append({
                    'gpu_id': i,
                    'name': name,
                    'utilization_gpu': gpu_util,
                    'utilization_memory': memory_util,
                    'memory_total': memory_total,
                    'memory_used': memory_used,
                    'memory_free': memory_free,
                    'memory_percent': memory_percent,
                    'temperature': temperature,
                    'power_draw': power
                })
                
            except Exception as e:
                print(f"获取GPU {i} 信息失败: {e}")
                
        return gpu_info
    
    def get_gpu_info_nvidia_smi(self) -> List[Dict[str, Any]]:
        """使用nvidia-smi命令获取GPU信息"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"nvidia-smi命令执行失败: {result.stderr}")
                return []
            
            gpu_info = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 9:
                        try:
                            memory_total = int(parts[4])
                            memory_used = int(parts[5])
                            memory_percent = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                            
                            gpu_info.append({
                                'gpu_id': int(parts[0]),
                                'name': parts[1],
                                'utilization_gpu': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                                'utilization_memory': int(parts[3]) if parts[3] != '[Not Supported]' else 0,
                                'memory_total': memory_total,
                                'memory_used': memory_used,
                                'memory_free': int(parts[6]),
                                'memory_percent': memory_percent,
                                'temperature': int(parts[7]) if parts[7] != '[Not Supported]' else 0,
                                'power_draw': float(parts[8]) if parts[8] != '[Not Supported]' else 0
                            })
                        except (ValueError, IndexError) as e:
                            print(f"解析GPU信息失败: {e}, line: {line}")
            
            return gpu_info
            
        except subprocess.TimeoutExpired:
            print("nvidia-smi命令超时")
            return []
        except Exception as e:
            print(f"执行nvidia-smi失败: {e}")
            return []
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """获取GPU信息"""
        if self.use_pynvml:
            return self.get_gpu_info_pynvml()
        else:
            return self.get_gpu_info_nvidia_smi()
    
    def write_csv_header(self):
        """写入CSV文件头"""
        fieldnames = [
            'timestamp', 'gpu_id', 'name', 'utilization_gpu', 'utilization_memory',
            'memory_total', 'memory_used', 'memory_free', 'memory_percent',
            'temperature', 'power_draw'
        ]
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def monitor_loop(self):
        """监控循环"""
        print(f"开始GPU监控，输出文件: {self.output_file}")
        print(f"监控间隔: {self.interval}秒")
        
        # 写入CSV头
        self.write_csv_header()
        
        while self.running:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                gpu_info_list = self.get_gpu_info()
                
                if gpu_info_list:
                    # 写入数据
                    with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                        fieldnames = [
                            'timestamp', 'gpu_id', 'name', 'utilization_gpu', 'utilization_memory',
                            'memory_total', 'memory_used', 'memory_free', 'memory_percent',
                            'temperature', 'power_draw'
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        
                        for gpu_info in gpu_info_list:
                            row = {'timestamp': timestamp}
                            row.update(gpu_info)
                            writer.writerow(row)
                    
                    # 打印当前状态
                    for gpu_info in gpu_info_list:
                        print(f"[{timestamp}] GPU{gpu_info['gpu_id']}: "
                              f"Util={gpu_info['utilization_gpu']}%, "
                              f"Mem={gpu_info['memory_used']}/{gpu_info['memory_total']}MB "
                              f"({gpu_info['memory_percent']:.1f}%), "
                              f"Temp={gpu_info['temperature']}°C")
                else:
                    print(f"[{timestamp}] 无法获取GPU信息")
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """启动监控"""
        if self.running:
            print("监控已在运行中")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print("GPU监控已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            print("监控未在运行")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("GPU监控已停止")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n接收到信号 {signum}，正在停止监控...")
    sys.exit(0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU监控工具")
    parser.add_argument("--output", "-o", default="gpu_metrics.csv", help="输出CSV文件路径")
    parser.add_argument("--interval", "-i", type=int, default=1, help="监控间隔（秒）")
    parser.add_argument("--duration", "-d", type=int, help="监控持续时间（秒），不指定则持续运行")
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor = GPUMonitor(output_file=args.output, interval=args.interval)
    
    try:
        monitor.start()
        
        if args.duration:
            print(f"将运行 {args.duration} 秒")
            time.sleep(args.duration)
        else:
            print("按 Ctrl+C 停止监控")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()