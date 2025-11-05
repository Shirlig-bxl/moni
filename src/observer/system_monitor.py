"""
系统监控模块
使用psutil监控CPU、内存、磁盘I/O等系统指标
"""

import time
import csv
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import argparse

import psutil


class SystemMonitor:
    """系统监控器类"""
    
    def __init__(self, output_file: str = "system_metrics.csv", interval: int = 1):
        self.output_file = output_file
        self.interval = interval
        self.running = False
        self.thread = None
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        # 初始化磁盘I/O计数器
        self.last_disk_io = None
        
        print(f"系统监控器初始化完成")
        print(f"CPU核心数: {psutil.cpu_count(logical=False)} 物理核心, {psutil.cpu_count(logical=True)} 逻辑核心")
        print(f"内存总量: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU信息"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            current_freq = cpu_freq.current if cpu_freq else 0
            
            # 负载平均值 (仅在Unix系统上可用)
            try:
                load_avg = os.getloadavg()
                load_1min, load_5min, load_15min = load_avg
            except (AttributeError, OSError):
                load_1min = load_5min = load_15min = 0
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_percent_max': max(cpu_percent_per_core) if cpu_percent_per_core else 0,
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq_current': current_freq,
                'load_avg_1min': load_1min,
                'load_avg_5min': load_5min,
                'load_avg_15min': load_15min
            }
        except Exception as e:
            print(f"获取CPU信息失败: {e}")
            return {}
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            # 虚拟内存
            virtual_mem = psutil.virtual_memory()
            
            # 交换内存
            swap_mem = psutil.swap_memory()
            
            return {
                'memory_total': virtual_mem.total,
                'memory_available': virtual_mem.available,
                'memory_used': virtual_mem.used,
                'memory_free': virtual_mem.free,
                'memory_percent': virtual_mem.percent,
                'memory_cached': getattr(virtual_mem, 'cached', 0),
                'memory_buffers': getattr(virtual_mem, 'buffers', 0),
                'swap_total': swap_mem.total,
                'swap_used': swap_mem.used,
                'swap_free': swap_mem.free,
                'swap_percent': swap_mem.percent
            }
        except Exception as e:
            print(f"获取内存信息失败: {e}")
            return {}
    
    def get_disk_info(self) -> Dict[str, Any]:
        """获取磁盘信息"""
        try:
            # 磁盘使用情况
            disk_usage = psutil.disk_usage('/')
            
            # 磁盘I/O统计
            disk_io = psutil.disk_io_counters()
            
            # 计算I/O速率
            io_read_rate = 0
            io_write_rate = 0
            
            if self.last_disk_io and disk_io:
                time_diff = self.interval
                read_diff = disk_io.read_bytes - self.last_disk_io.read_bytes
                write_diff = disk_io.write_bytes - self.last_disk_io.write_bytes
                
                io_read_rate = read_diff / time_diff if time_diff > 0 else 0
                io_write_rate = write_diff / time_diff if time_diff > 0 else 0
            
            self.last_disk_io = disk_io
            
            result = {
                'disk_total': disk_usage.total,
                'disk_used': disk_usage.used,
                'disk_free': disk_usage.free,
                'disk_percent': (disk_usage.used / disk_usage.total) * 100 if disk_usage.total > 0 else 0
            }
            
            if disk_io:
                result.update({
                    'disk_read_count': disk_io.read_count,
                    'disk_write_count': disk_io.write_count,
                    'disk_read_bytes': disk_io.read_bytes,
                    'disk_write_bytes': disk_io.write_bytes,
                    'disk_read_rate': io_read_rate,
                    'disk_write_rate': io_write_rate,
                    'disk_read_time': disk_io.read_time,
                    'disk_write_time': disk_io.write_time
                })
            
            return result
            
        except Exception as e:
            print(f"获取磁盘信息失败: {e}")
            return {}
    
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        try:
            net_io = psutil.net_io_counters()
            
            if net_io:
                return {
                    'network_bytes_sent': net_io.bytes_sent,
                    'network_bytes_recv': net_io.bytes_recv,
                    'network_packets_sent': net_io.packets_sent,
                    'network_packets_recv': net_io.packets_recv,
                    'network_errin': net_io.errin,
                    'network_errout': net_io.errout,
                    'network_dropin': net_io.dropin,
                    'network_dropout': net_io.dropout
                }
            else:
                return {}
                
        except Exception as e:
            print(f"获取网络信息失败: {e}")
            return {}
    
    def get_process_info(self) -> Dict[str, Any]:
        """获取进程信息"""
        try:
            # 进程数量
            process_count = len(psutil.pids())
            
            # 运行中的进程数量
            running_processes = 0
            sleeping_processes = 0
            
            for proc in psutil.process_iter(['status']):
                try:
                    status = proc.info['status']
                    if status == psutil.STATUS_RUNNING:
                        running_processes += 1
                    elif status == psutil.STATUS_SLEEPING:
                        sleeping_processes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'process_count': process_count,
                'process_running': running_processes,
                'process_sleeping': sleeping_processes
            }
            
        except Exception as e:
            print(f"获取进程信息失败: {e}")
            return {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取完整系统信息"""
        system_info = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        # 合并所有信息
        system_info.update(self.get_cpu_info())
        system_info.update(self.get_memory_info())
        system_info.update(self.get_disk_info())
        system_info.update(self.get_network_info())
        system_info.update(self.get_process_info())
        
        return system_info
    
    def write_csv_header(self):
        """写入CSV文件头"""
        # 获取一次系统信息来确定字段名
        sample_info = self.get_system_info()
        fieldnames = list(sample_info.keys())
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def monitor_loop(self):
        """监控循环"""
        print(f"开始系统监控，输出文件: {self.output_file}")
        print(f"监控间隔: {self.interval}秒")
        
        # 写入CSV头
        self.write_csv_header()
        
        while self.running:
            try:
                system_info = self.get_system_info()
                
                # 写入数据
                with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                    fieldnames = list(system_info.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(system_info)
                
                # 打印当前状态
                timestamp = system_info['timestamp']
                cpu_percent = system_info.get('cpu_percent', 0)
                memory_percent = system_info.get('memory_percent', 0)
                disk_percent = system_info.get('disk_percent', 0)
                
                print(f"[{timestamp}] CPU: {cpu_percent:.1f}%, "
                      f"Memory: {memory_percent:.1f}%, "
                      f"Disk: {disk_percent:.1f}%")
                
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
        print("系统监控已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            print("监控未在运行")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("系统监控已停止")
    
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
    parser = argparse.ArgumentParser(description="系统监控工具")
    parser.add_argument("--output", "-o", default="system_metrics.csv", help="输出CSV文件路径")
    parser.add_argument("--interval", "-i", type=int, default=1, help="监控间隔（秒）")
    parser.add_argument("--duration", "-d", type=int, help="监控持续时间（秒），不指定则持续运行")
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor = SystemMonitor(output_file=args.output, interval=args.interval)
    
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