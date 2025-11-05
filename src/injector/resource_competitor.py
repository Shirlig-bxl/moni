"""
资源竞争模拟器
模拟"吵闹的邻居"，与训练任务竞争GPU、CPU、内存等资源
"""

import os
import time
import threading
import subprocess
import multiprocessing
import argparse
from typing import List, Optional
import psutil


class ResourceCompetitor:
    """资源竞争器"""
    
    def __init__(self):
        self.running = False
        self.competitor_processes = []
        self.competitor_threads = []
        
        print("资源竞争器初始化完成")
    
    def gpu_competition(self, duration: float = 60, intensity: int = 2):
        """
        GPU资源竞争
        
        Args:
            duration: 持续时间（秒）
            intensity: 竞争强度（并发进程数）
        """
        print(f"启动GPU资源竞争，强度: {intensity}, 持续: {duration}秒")
        
        gpu_stress_script = '''
import torch
import time
import sys

def gpu_stress_worker(worker_id, duration):
    print(f"GPU竞争进程 {worker_id} 开始")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU模拟")
        # CPU密集型计算作为备选
        start_time = time.time()
        while time.time() - start_time < duration:
            # 大矩阵计算
            import numpy as np
            a = np.random.randn(1000, 1000)
            b = np.random.randn(1000, 1000)
            c = np.dot(a, b)
            time.sleep(0.01)
        return
    
    device = torch.cuda.current_device()
    print(f"使用GPU设备: {device}")
    
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < duration:
            # 创建大张量进行计算
            size = 1000 + (iteration % 500)  # 动态大小
            
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # 矩阵乘法
            c = torch.matmul(a, b)
            
            # 一些其他GPU操作
            d = torch.relu(c)
            e = torch.softmax(d, dim=1)
            f = torch.sum(e)
            
            # 强制同步
            torch.cuda.synchronize()
            
            iteration += 1
            
            if iteration % 10 == 0:
                print(f"GPU竞争进程 {worker_id} - 迭代 {iteration}")
            
            # 短暂休息
            time.sleep(0.05)
            
    except Exception as e:
        print(f"GPU竞争进程 {worker_id} 错误: {e}")
    
    print(f"GPU竞争进程 {worker_id} 结束")

if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    duration = float(sys.argv[2])
    gpu_stress_worker(worker_id, duration)
'''
        
        # 启动多个GPU竞争进程
        for i in range(intensity):
            try:
                proc = subprocess.Popen([
                    'python', '-c', gpu_stress_script, str(i), str(duration)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.competitor_processes.append(proc)
                print(f"启动GPU竞争进程 {i}: PID={proc.pid}")
                
            except Exception as e:
                print(f"启动GPU竞争进程 {i} 失败: {e}")
    
    def cpu_competition(self, duration: float = 60, intensity: int = None):
        """
        CPU资源竞争
        
        Args:
            duration: 持续时间（秒）
            intensity: 竞争强度（进程数，默认为CPU核心数）
        """
        if intensity is None:
            intensity = multiprocessing.cpu_count()
        
        print(f"启动CPU资源竞争，强度: {intensity}, 持续: {duration}秒")
        
        cpu_stress_script = '''
import time
import sys
import math

def cpu_intensive_work(worker_id, duration):
    print(f"CPU竞争进程 {worker_id} 开始")
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration:
        # CPU密集型计算
        result = 0
        for i in range(10000):
            result += math.sqrt(i) * math.sin(i) * math.cos(i)
        
        # 一些字符串操作
        text = "stress test " * 1000
        text = text.upper().lower().replace("stress", "STRESS")
        
        iteration += 1
        
        if iteration % 1000 == 0:
            print(f"CPU竞争进程 {worker_id} - 迭代 {iteration}")
    
    print(f"CPU竞争进程 {worker_id} 结束")

if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    duration = float(sys.argv[2])
    cpu_intensive_work(worker_id, duration)
'''
        
        # 启动多个CPU竞争进程
        for i in range(intensity):
            try:
                proc = subprocess.Popen([
                    'python', '-c', cpu_stress_script, str(i), str(duration)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                self.competitor_processes.append(proc)
                print(f"启动CPU竞争进程 {i}: PID={proc.pid}")
                
            except Exception as e:
                print(f"启动CPU竞争进程 {i} 失败: {e}")
    
    def memory_competition(self, duration: float = 60, memory_gb: float = 2.0):
        """
        内存资源竞争
        
        Args:
            duration: 持续时间（秒）
            memory_gb: 要分配的内存量（GB）
        """
        print(f"启动内存资源竞争，分配: {memory_gb}GB, 持续: {duration}秒")
        
        memory_stress_script = f'''
import time
import sys
import numpy as np
import gc

def memory_intensive_work(memory_gb, duration):
    print(f"内存竞争进程开始，分配 {{memory_gb}}GB")
    
    arrays = []
    chunk_size_mb = 100  # 每次分配100MB
    total_chunks = int(memory_gb * 1024 / chunk_size_mb)
    
    start_time = time.time()
    
    try:
        # 分配内存
        for i in range(total_chunks):
            if time.time() - start_time >= duration:
                break
                
            # 分配100MB数组
            arr = np.random.randn(chunk_size_mb * 1024 * 1024 // 8)  # 8 bytes per float64
            arrays.append(arr)
            
            if i % 10 == 0:
                print(f"已分配 {{i * chunk_size_mb}}MB 内存")
            
            time.sleep(0.1)
        
        print(f"内存分配完成，总计: {{len(arrays) * chunk_size_mb}}MB")
        
        # 保持内存占用
        remaining_time = duration - (time.time() - start_time)
        if remaining_time > 0:
            print(f"保持内存占用 {{remaining_time:.1f}} 秒")
            
            # 定期访问内存以防止被交换
            while time.time() - start_time < duration:
                for i, arr in enumerate(arrays[::10]):  # 每10个数组访问一次
                    _ = arr[0]  # 简单访问
                    if i % 100 == 0:
                        print(f"内存访问检查: {{i}}")
                
                time.sleep(1)
    
    except MemoryError:
        print("内存分配失败：内存不足")
    except Exception as e:
        print(f"内存竞争错误: {{e}}")
    finally:
        # 清理内存
        arrays.clear()
        gc.collect()
        print("内存竞争进程结束")

if __name__ == "__main__":
    memory_gb = {memory_gb}
    duration = {duration}
    memory_intensive_work(memory_gb, duration)
'''
        
        try:
            proc = subprocess.Popen([
                'python', '-c', memory_stress_script
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.competitor_processes.append(proc)
            print(f"启动内存竞争进程: PID={proc.pid}")
            
        except Exception as e:
            print(f"启动内存竞争进程失败: {e}")
    
    def network_competition(self, duration: float = 60, bandwidth_mbps: int = 100):
        """
        网络资源竞争（模拟）
        
        Args:
            duration: 持续时间（秒）
            bandwidth_mbps: 模拟带宽（Mbps）
        """
        print(f"启动网络资源竞争，带宽: {bandwidth_mbps}Mbps, 持续: {duration}秒")
        
        network_stress_script = f'''
import time
import socket
import threading
import sys

def network_stress_worker(duration, bandwidth_mbps):
    print("网络竞争进程开始")
    
    # 创建大量网络连接（模拟）
    sockets = []
    
    start_time = time.time()
    
    try:
        # 创建多个socket连接
        for i in range(10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                # 尝试连接到本地端口（通常会失败，但会消耗资源）
                try:
                    sock.connect(('127.0.0.1', 12345 + i))
                except:
                    pass
                sockets.append(sock)
            except:
                pass
        
        # 模拟网络数据传输
        data = b'x' * (bandwidth_mbps * 1024 * 1024 // 8 // 10)  # 每次传输1/10的带宽数据
        
        while time.time() - start_time < duration:
            # 模拟数据发送/接收
            for sock in sockets:
                try:
                    sock.send(data[:1024])  # 发送小块数据
                except:
                    pass
            
            time.sleep(0.1)
            
            if int(time.time() - start_time) % 10 == 0:
                print(f"网络竞争运行中: {{int(time.time() - start_time)}}s")
    
    except Exception as e:
        print(f"网络竞争错误: {{e}}")
    finally:
        # 清理socket
        for sock in sockets:
            try:
                sock.close()
            except:
                pass
        print("网络竞争进程结束")

if __name__ == "__main__":
    duration = {duration}
    bandwidth_mbps = {bandwidth_mbps}
    network_stress_worker(duration, bandwidth_mbps)
'''
        
        try:
            proc = subprocess.Popen([
                'python', '-c', network_stress_script
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.competitor_processes.append(proc)
            print(f"启动网络竞争进程: PID={proc.pid}")
            
        except Exception as e:
            print(f"启动网络竞争进程失败: {e}")
    
    def start_competition(self, resource_types: List[str], duration: float = 60, **kwargs):
        """
        启动资源竞争
        
        Args:
            resource_types: 资源类型列表 ["gpu", "cpu", "memory", "network"]
            duration: 持续时间（秒）
            **kwargs: 各资源类型的特定参数
        """
        print(f"启动资源竞争: {resource_types}, 持续时间: {duration}秒")
        
        self.running = True
        
        # 启动各种资源竞争
        if "gpu" in resource_types:
            intensity = kwargs.get('gpu_intensity', 2)
            self.gpu_competition(duration, intensity)
        
        if "cpu" in resource_types:
            intensity = kwargs.get('cpu_intensity', None)
            self.cpu_competition(duration, intensity)
        
        if "memory" in resource_types:
            memory_gb = kwargs.get('memory_gb', 2.0)
            self.memory_competition(duration, memory_gb)
        
        if "network" in resource_types:
            bandwidth_mbps = kwargs.get('bandwidth_mbps', 100)
            self.network_competition(duration, bandwidth_mbps)
        
        # 监控竞争进程
        self.monitor_competition(duration)
    
    def monitor_competition(self, duration: float):
        """监控竞争进程"""
        print("开始监控资源竞争...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration and self.running:
            active_processes = []
            
            for proc in self.competitor_processes:
                if proc.poll() is None:  # 进程仍在运行
                    active_processes.append(proc.pid)
            
            if active_processes:
                print(f"活跃竞争进程: {len(active_processes)} 个")
            
            time.sleep(5)  # 每5秒检查一次
        
        # 停止所有竞争进程
        self.stop_competition()
    
    def stop_competition(self):
        """停止资源竞争"""
        print("停止资源竞争...")
        
        self.running = False
        
        # 终止所有竞争进程
        for proc in self.competitor_processes:
            try:
                if proc.poll() is None:  # 进程仍在运行
                    print(f"终止进程: PID={proc.pid}")
                    proc.terminate()
                    
                    # 等待进程结束
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"强制杀死进程: PID={proc.pid}")
                        proc.kill()
                        
            except Exception as e:
                print(f"终止进程失败: {e}")
        
        # 清理进程列表
        self.competitor_processes.clear()
        
        print("资源竞争已停止")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_competition()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="资源竞争模拟器")
    parser.add_argument("--resources", nargs="+", 
                       choices=["gpu", "cpu", "memory", "network"],
                       default=["gpu", "cpu"],
                       help="竞争的资源类型")
    parser.add_argument("--duration", type=float, default=60, help="持续时间（秒）")
    parser.add_argument("--gpu-intensity", type=int, default=2, help="GPU竞争强度")
    parser.add_argument("--cpu-intensity", type=int, help="CPU竞争强度（默认为CPU核心数）")
    parser.add_argument("--memory-gb", type=float, default=2.0, help="内存分配量（GB）")
    parser.add_argument("--bandwidth-mbps", type=int, default=100, help="网络带宽（Mbps）")
    
    args = parser.parse_args()
    
    competitor = ResourceCompetitor()
    
    try:
        competitor.start_competition(
            resource_types=args.resources,
            duration=args.duration,
            gpu_intensity=args.gpu_intensity,
            cpu_intensity=args.cpu_intensity,
            memory_gb=args.memory_gb,
            bandwidth_mbps=args.bandwidth_mbps
        )
    except KeyboardInterrupt:
        print("\n用户中断")
        competitor.stop_competition()


if __name__ == "__main__":
    main()