"""
I/O压力测试模块
专门用于创建磁盘I/O瓶颈
"""

import os
import time
import threading
import subprocess
import tempfile
import argparse
from typing import List, Optional
import psutil


class IOStressTester:
    """I/O压力测试器"""
    
    def __init__(self, target_dir: str = None, block_size: str = "1M", 
                 num_files: int = 5, file_size_mb: int = 100):
        self.target_dir = target_dir or tempfile.gettempdir()
        self.block_size = block_size
        self.num_files = num_files
        self.file_size_mb = file_size_mb
        self.running = False
        self.stress_processes = []
        self.stress_threads = []
        
        # 确保目标目录存在
        os.makedirs(self.target_dir, exist_ok=True)
        
        print(f"I/O压力测试器初始化完成")
        print(f"目标目录: {self.target_dir}")
        print(f"文件数量: {num_files}, 文件大小: {file_size_mb}MB")
    
    def create_write_stress(self, file_index: int):
        """创建写入压力"""
        filename = os.path.join(self.target_dir, f"stress_write_{file_index}.tmp")
        
        try:
            # 使用dd命令创建写入压力
            cmd = [
                'dd',
                'if=/dev/zero',
                f'of={filename}',
                f'bs={self.block_size}',
                f'count={self.file_size_mb}',
                'conv=fdatasync'  # 强制同步写入
            ]
            
            while self.running:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.stress_processes.append(proc)
                
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                
                # 删除文件后重新创建
                try:
                    os.remove(filename)
                except:
                    pass
                
                if not self.running:
                    break
                    
        except Exception as e:
            print(f"写入压力测试失败: {e}")
        finally:
            # 清理文件
            try:
                os.remove(filename)
            except:
                pass
    
    def create_read_stress(self, file_index: int):
        """创建读取压力"""
        filename = os.path.join(self.target_dir, f"stress_read_{file_index}.tmp")
        
        try:
            # 先创建一个文件用于读取
            create_cmd = [
                'dd',
                'if=/dev/zero',
                f'of={filename}',
                f'bs={self.block_size}',
                f'count={self.file_size_mb}',
                'conv=fdatasync'
            ]
            
            subprocess.run(create_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 持续读取文件
            read_cmd = [
                'dd',
                f'if={filename}',
                'of=/dev/null',
                f'bs={self.block_size}'
            ]
            
            while self.running:
                proc = subprocess.Popen(read_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.stress_processes.append(proc)
                
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                
                if not self.running:
                    break
                    
        except Exception as e:
            print(f"读取压力测试失败: {e}")
        finally:
            # 清理文件
            try:
                os.remove(filename)
            except:
                pass
    
    def create_random_io_stress(self, file_index: int):
        """创建随机I/O压力"""
        filename = os.path.join(self.target_dir, f"stress_random_{file_index}.tmp")
        
        try:
            # 创建一个大文件
            create_cmd = [
                'dd',
                'if=/dev/zero',
                f'of={filename}',
                f'bs={self.block_size}',
                f'count={self.file_size_mb * 2}',  # 更大的文件用于随机访问
                'conv=fdatasync'
            ]
            
            subprocess.run(create_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 使用Python进行随机I/O
            import random
            
            while self.running:
                try:
                    with open(filename, 'r+b') as f:
                        file_size = os.path.getsize(filename)
                        
                        for _ in range(100):  # 100次随机访问
                            if not self.running:
                                break
                                
                            # 随机位置
                            pos = random.randint(0, max(0, file_size - 1024))
                            f.seek(pos)
                            
                            # 随机读写
                            if random.choice([True, False]):
                                # 读取
                                f.read(1024)
                            else:
                                # 写入
                                f.write(b'x' * 1024)
                                f.flush()
                                os.fsync(f.fileno())
                        
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"随机I/O错误: {e}")
                    break
                    
        except Exception as e:
            print(f"随机I/O压力测试失败: {e}")
        finally:
            # 清理文件
            try:
                os.remove(filename)
            except:
                pass
    
    def monitor_io_stats(self):
        """监控I/O统计"""
        print("开始监控I/O统计...")
        
        last_io = psutil.disk_io_counters()
        
        while self.running:
            try:
                time.sleep(1)
                current_io = psutil.disk_io_counters()
                
                if last_io and current_io:
                    read_rate = (current_io.read_bytes - last_io.read_bytes) / 1024 / 1024  # MB/s
                    write_rate = (current_io.write_bytes - last_io.write_bytes) / 1024 / 1024  # MB/s
                    
                    print(f"I/O速率 - 读取: {read_rate:.2f} MB/s, 写入: {write_rate:.2f} MB/s")
                
                last_io = current_io
                
            except Exception as e:
                print(f"I/O监控错误: {e}")
                break
    
    def start_stress_test(self, duration: float = 60, stress_types: List[str] = None):
        """
        启动压力测试
        
        Args:
            duration: 持续时间（秒）
            stress_types: 压力类型列表 ["write", "read", "random"]
        """
        if stress_types is None:
            stress_types = ["write", "read", "random"]
        
        print(f"启动I/O压力测试，持续时间: {duration}秒")
        print(f"压力类型: {stress_types}")
        
        self.running = True
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self.monitor_io_stats)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 启动压力测试线程
        for i in range(self.num_files):
            if "write" in stress_types:
                thread = threading.Thread(target=self.create_write_stress, args=(i,))
                thread.daemon = True
                thread.start()
                self.stress_threads.append(thread)
            
            if "read" in stress_types:
                thread = threading.Thread(target=self.create_read_stress, args=(i,))
                thread.daemon = True
                thread.start()
                self.stress_threads.append(thread)
            
            if "random" in stress_types:
                thread = threading.Thread(target=self.create_random_io_stress, args=(i,))
                thread.daemon = True
                thread.start()
                self.stress_threads.append(thread)
        
        # 等待指定时间
        time.sleep(duration)
        
        # 停止压力测试
        self.stop_stress_test()
    
    def stop_stress_test(self):
        """停止压力测试"""
        print("停止I/O压力测试...")
        
        self.running = False
        
        # 终止所有进程
        for proc in self.stress_processes:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except:
                try:
                    proc.kill()
                except:
                    pass
        
        # 等待线程结束
        for thread in self.stress_threads:
            try:
                thread.join(timeout=2)
            except:
                pass
        
        # 清理临时文件
        self.cleanup_temp_files()
        
        print("I/O压力测试已停止")
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for filename in os.listdir(self.target_dir):
                if filename.startswith('stress_') and filename.endswith('.tmp'):
                    filepath = os.path.join(self.target_dir, filename)
                    try:
                        os.remove(filepath)
                        print(f"清理文件: {filepath}")
                    except:
                        pass
        except Exception as e:
            print(f"清理临时文件失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_stress_test()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="I/O压力测试工具")
    parser.add_argument("--target-dir", help="目标目录")
    parser.add_argument("--duration", type=float, default=60, help="持续时间（秒）")
    parser.add_argument("--num-files", type=int, default=5, help="文件数量")
    parser.add_argument("--file-size", type=int, default=100, help="文件大小（MB）")
    parser.add_argument("--block-size", default="1M", help="块大小")
    parser.add_argument("--stress-types", nargs="+", 
                       choices=["write", "read", "random"], 
                       default=["write", "read", "random"],
                       help="压力类型")
    
    args = parser.parse_args()
    
    tester = IOStressTester(
        target_dir=args.target_dir,
        block_size=args.block_size,
        num_files=args.num_files,
        file_size_mb=args.file_size
    )
    
    try:
        tester.start_stress_test(
            duration=args.duration,
            stress_types=args.stress_types
        )
    except KeyboardInterrupt:
        print("\n用户中断")
        tester.stop_stress_test()


if __name__ == "__main__":
    main()