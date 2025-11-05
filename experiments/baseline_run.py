"""
基线实验运行脚本
运行正常的训练任务，收集基线数据
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from observer import GPUMonitor, SystemMonitor, LogParser
from executor import FaultConfigFactory


def run_baseline_experiment(output_dir: str = "./data/raw/baseline", duration: int = 300):
    """
    运行基线实验
    
    Args:
        output_dir: 输出目录
        duration: 实验持续时间（秒）
    """
    print("=" * 60)
    print("开始基线实验")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置文件路径
    gpu_metrics_file = os.path.join(output_dir, "gpu_metrics.csv")
    system_metrics_file = os.path.join(output_dir, "system_metrics.csv")
    training_log_file = os.path.join(output_dir, "train.log")
    training_metrics_file = os.path.join(output_dir, "training_metrics.csv")
    
    # 创建基线配置
    config = FaultConfigFactory.create_baseline_config()
    config.output_dir = output_dir
    config.logging_dir = output_dir
    
    print(f"输出目录: {output_dir}")
    print(f"预计持续时间: {duration}秒")
    
    # 启动监控
    monitors = []
    
    try:
        # 1. 启动GPU监控
        print("启动GPU监控...")
        gpu_monitor = GPUMonitor(output_file=gpu_metrics_file, interval=1)
        gpu_monitor.start()
        monitors.append(gpu_monitor)
        
        # 2. 启动系统监控
        print("启动系统监控...")
        system_monitor = SystemMonitor(output_file=system_metrics_file, interval=1)
        system_monitor.start()
        monitors.append(system_monitor)
        
        # 3. 启动训练任务
        print("启动训练任务...")
        train_cmd = [
            sys.executable, "-m", "src.executor.train",
            "--config-name", "baseline"
        ]
        
        # 重定向训练日志
        with open(training_log_file, 'w') as log_file:
            train_process = subprocess.Popen(
                train_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent)
            )
        
        # 4. 启动日志解析
        print("启动日志解析...")
        time.sleep(2)  # 等待日志文件创建
        log_parser = LogParser(
            log_file=training_log_file,
            output_file=training_metrics_file,
            interval=1
        )
        log_parser.start()
        monitors.append(log_parser)
        
        # 5. 等待训练完成或超时
        print("等待训练完成...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 检查训练进程状态
            if train_process.poll() is not None:
                print(f"训练进程已结束，退出码: {train_process.returncode}")
                break
            
            time.sleep(5)
            elapsed = time.time() - start_time
            print(f"实验进行中... {elapsed:.0f}/{duration}s")
        
        # 如果训练还在运行，等待其完成
        if train_process.poll() is None:
            print("等待训练进程完成...")
            try:
                train_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("训练进程超时，强制终止")
                train_process.kill()
        
        print("基线实验完成")
        
    except Exception as e:
        print(f"实验过程中发生错误: {e}")
        
    finally:
        # 停止所有监控
        print("停止监控...")
        for monitor in monitors:
            try:
                monitor.stop()
            except:
                pass
        
        # 确保训练进程已终止
        try:
            if train_process.poll() is None:
                train_process.terminate()
                train_process.wait(timeout=10)
        except:
            pass
    
    # 检查生成的文件
    print("\n生成的文件:")
    for file_path in [gpu_metrics_file, system_metrics_file, training_log_file, training_metrics_file]:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file_path}: {size} bytes")
        else:
            print(f"  {file_path}: 未生成")
    
    print(f"\n基线实验数据已保存到: {output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="基线实验运行脚本")
    parser.add_argument("--output-dir", default="./data/raw/baseline", help="输出目录")
    parser.add_argument("--duration", type=int, default=300, help="实验持续时间（秒）")
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        output_dir=args.output_dir,
        duration=args.duration
    )


if __name__ == "__main__":
    main()