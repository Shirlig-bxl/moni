"""
故障注入控制器
负责在训练过程中主动触发各种故障
"""

import os
import time
import signal
import subprocess
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import psutil
import argparse


class FaultInjector:
    """故障注入器类"""
    
    def __init__(self):
        self.running = False
        self.injection_thread = None
        self.fault_schedule = []  # 故障注入计划
        self.start_time = None
        
        print("故障注入器初始化完成")
    
    def schedule_fault(self, fault_type: str, delay: float, duration: float = 0, **kwargs):
        """
        调度故障注入
        
        Args:
            fault_type: 故障类型 ("io_stress", "resource_competition", "process_kill")
            delay: 延迟时间（秒）
            duration: 持续时间（秒），0表示一次性操作
            **kwargs: 故障特定参数
        """
        fault_config = {
            'fault_type': fault_type,
            'delay': delay,
            'duration': duration,
            'params': kwargs,
            'executed': False
        }
        
        self.fault_schedule.append(fault_config)
        print(f"已调度故障: {fault_type}, 延迟: {delay}s, 持续: {duration}s")
    
    def inject_io_stress(self, target_dir: str = "/tmp", duration: float = 60, 
                        read_rate: str = "100M", write_rate: str = "100M"):
        """
        注入I/O压力
        
        Args:
            target_dir: 目标目录
            duration: 持续时间（秒）
            read_rate: 读取速率
            write_rate: 写入速率
        """
        print(f"[FAULT INJECTION] 开始I/O压力测试: {target_dir}, 持续 {duration}s")
        
        try:
            # 使用dd命令创建I/O压力
            stress_processes = []
            
            # 写入压力
            write_cmd = [
                'dd', 'if=/dev/zero', f'of={target_dir}/stress_write_test',
                f'bs=1M', 'count=1000', 'oflag=direct'
            ]
            
            # 读取压力 (如果文件存在)
            read_cmd = [
                'dd', f'if={target_dir}/stress_write_test', 'of=/dev/null',
                f'bs=1M', 'iflag=direct'
            ]
            
            # 启动写入压力
            write_proc = subprocess.Popen(write_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stress_processes.append(write_proc)
            
            # 等待一段时间后启动读取压力
            time.sleep(2)
            
            # 启动多个读取进程
            for i in range(3):
                try:
                    read_proc = subprocess.Popen(read_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    stress_processes.append(read_proc)
                except:
                    pass
            
            # 等待指定时间
            time.sleep(duration)
            
            # 终止所有压力进程
            for proc in stress_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            
            # 清理测试文件
            try:
                os.remove(f'{target_dir}/stress_write_test')
            except:
                pass
            
            print(f"[FAULT INJECTION] I/O压力测试结束")
            
        except Exception as e:
            print(f"[FAULT INJECTION] I/O压力测试失败: {e}")
    
    def inject_resource_competition(self, target_process_name: str = "python", 
                                  duration: float = 60, competitor_type: str = "gpu"):
        """
        注入资源竞争
        
        Args:
            target_process_name: 目标进程名称
            duration: 持续时间（秒）
            competitor_type: 竞争类型 ("gpu", "cpu", "memory")
        """
        print(f"[FAULT INJECTION] 开始资源竞争: {competitor_type}, 持续 {duration}s")
        
        competitor_processes = []
        
        try:
            if competitor_type == "gpu":
                # GPU竞争：启动GPU密集型任务
                gpu_stress_cmd = [
                    'python', '-c',
                    '''
import torch
import time
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    # 创建大矩阵进行计算
    for i in range(1000):
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        time.sleep(0.1)
else:
    print("CUDA not available")
                    '''
                ]
                
                # 启动多个GPU竞争进程
                for i in range(2):
                    try:
                        proc = subprocess.Popen(gpu_stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        competitor_processes.append(proc)
                    except Exception as e:
                        print(f"启动GPU竞争进程失败: {e}")
            
            elif competitor_type == "cpu":
                # CPU竞争：启动CPU密集型任务
                cpu_stress_cmd = [
                    'python', '-c',
                    '''
import time
import multiprocessing
def cpu_intensive():
    while True:
        sum(i*i for i in range(10000))

processes = []
for i in range(multiprocessing.cpu_count()):
    p = multiprocessing.Process(target=cpu_intensive)
    p.start()
    processes.append(p)

time.sleep(60)
for p in processes:
    p.terminate()
                    '''
                ]
                
                proc = subprocess.Popen(cpu_stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                competitor_processes.append(proc)
            
            elif competitor_type == "memory":
                # 内存竞争：分配大量内存
                memory_stress_cmd = [
                    'python', '-c',
                    '''
import time
import numpy as np
arrays = []
try:
    for i in range(10):
        # 分配1GB内存
        arr = np.zeros((1024, 1024, 128), dtype=np.float64)
        arrays.append(arr)
        time.sleep(1)
    time.sleep(60)
except MemoryError:
    print("Memory allocation failed")
                    '''
                ]
                
                proc = subprocess.Popen(memory_stress_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                competitor_processes.append(proc)
            
            # 等待指定时间
            time.sleep(duration)
            
        except Exception as e:
            print(f"[FAULT INJECTION] 资源竞争注入失败: {e}")
        
        finally:
            # 终止所有竞争进程
            for proc in competitor_processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            
            print(f"[FAULT INJECTION] 资源竞争结束")
    
    def inject_process_kill(self, target_process_name: str = "python", 
                           target_cmdline_pattern: str = "train.py"):
        """
        注入进程终止
        
        Args:
            target_process_name: 目标进程名称
            target_cmdline_pattern: 命令行模式匹配
        """
        print(f"[FAULT INJECTION] 查找并终止进程: {target_process_name}, 模式: {target_cmdline_pattern}")
        
        killed_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    proc_info = proc.info
                    if (proc_info['name'] and target_process_name in proc_info['name'].lower() and
                        proc_info['cmdline'] and any(target_cmdline_pattern in cmd for cmd in proc_info['cmdline'])):
                        
                        print(f"[FAULT INJECTION] 找到目标进程: PID={proc.pid}, CMD={' '.join(proc_info['cmdline'])}")
                        
                        # 发送SIGTERM信号
                        proc.terminate()
                        killed_processes.append(proc.pid)
                        
                        # 等待3秒，如果还没结束就强制杀死
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            print(f"[FAULT INJECTION] 强制杀死进程: PID={proc.pid}")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            if killed_processes:
                print(f"[FAULT INJECTION] 已终止进程: {killed_processes}")
            else:
                print(f"[FAULT INJECTION] 未找到匹配的进程")
                
        except Exception as e:
            print(f"[FAULT INJECTION] 进程终止失败: {e}")
    
    def execute_fault(self, fault_config: Dict[str, Any]):
        """执行单个故障注入"""
        fault_type = fault_config['fault_type']
        params = fault_config['params']
        
        print(f"[FAULT INJECTION] 执行故障: {fault_type}")
        
        try:
            if fault_type == "io_stress":
                self.inject_io_stress(**params)
            elif fault_type == "resource_competition":
                self.inject_resource_competition(**params)
            elif fault_type == "process_kill":
                self.inject_process_kill(**params)
            else:
                print(f"[FAULT INJECTION] 未知故障类型: {fault_type}")
        
        except Exception as e:
            print(f"[FAULT INJECTION] 故障执行失败: {e}")
    
    def injection_loop(self):
        """故障注入循环"""
        print("故障注入器开始运行")
        self.start_time = time.time()
        
        while self.running:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # 检查是否有需要执行的故障
            for fault_config in self.fault_schedule:
                if not fault_config['executed'] and elapsed_time >= fault_config['delay']:
                    fault_config['executed'] = True
                    
                    # 在新线程中执行故障注入
                    fault_thread = threading.Thread(
                        target=self.execute_fault,
                        args=(fault_config,)
                    )
                    fault_thread.daemon = True
                    fault_thread.start()
            
            time.sleep(1)  # 每秒检查一次
        
        print("故障注入器停止运行")
    
    def start(self):
        """启动故障注入器"""
        if self.running:
            print("故障注入器已在运行中")
            return
        
        if not self.fault_schedule:
            print("没有调度的故障，故障注入器不会启动")
            return
        
        self.running = True
        self.injection_thread = threading.Thread(target=self.injection_loop)
        self.injection_thread.daemon = True
        self.injection_thread.start()
        print("故障注入器已启动")
    
    def stop(self):
        """停止故障注入器"""
        if not self.running:
            print("故障注入器未在运行")
            return
        
        self.running = False
        if self.injection_thread:
            self.injection_thread.join(timeout=5)
        print("故障注入器已停止")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_fault_schedule_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从配置创建故障调度"""
    schedule = []
    
    for fault_name, fault_config in config.items():
        schedule.append({
            'fault_type': fault_config['type'],
            'delay': fault_config['delay'],
            'duration': fault_config.get('duration', 0),
            'params': fault_config.get('params', {}),
            'executed': False
        })
    
    return schedule


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="故障注入工具")
    parser.add_argument("--fault-type", choices=["io_stress", "resource_competition", "process_kill"], 
                       help="故障类型")
    parser.add_argument("--delay", type=float, default=0, help="延迟时间（秒）")
    parser.add_argument("--duration", type=float, default=60, help="持续时间（秒）")
    parser.add_argument("--target-dir", default="/tmp", help="I/O压力目标目录")
    parser.add_argument("--competitor-type", choices=["gpu", "cpu", "memory"], default="gpu", 
                       help="资源竞争类型")
    parser.add_argument("--target-process", default="python", help="目标进程名称")
    parser.add_argument("--target-pattern", default="train.py", help="目标进程命令行模式")
    
    args = parser.parse_args()
    
    if not args.fault_type:
        print("请指定故障类型")
        return
    
    injector = FaultInjector()
    
    # 根据参数调度故障
    if args.fault_type == "io_stress":
        injector.schedule_fault(
            "io_stress", 
            delay=args.delay, 
            duration=args.duration,
            target_dir=args.target_dir
        )
    elif args.fault_type == "resource_competition":
        injector.schedule_fault(
            "resource_competition",
            delay=args.delay,
            duration=args.duration,
            competitor_type=args.competitor_type
        )
    elif args.fault_type == "process_kill":
        injector.schedule_fault(
            "process_kill",
            delay=args.delay,
            target_process_name=args.target_process,
            target_cmdline_pattern=args.target_pattern
        )
    
    try:
        injector.start()
        print("按 Ctrl+C 停止故障注入器")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        injector.stop()


if __name__ == "__main__":
    main()