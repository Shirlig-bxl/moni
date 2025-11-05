"""
故障注入实验脚本
运行各种故障注入实验，收集异常数据
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import yaml

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from observer import GPUMonitor, SystemMonitor, LogParser
from executor import FaultConfigFactory
from injector import FaultInjector


def run_fault_experiment(fault_name: str, output_dir: str = None, duration: int = 300):
    """
    运行单个故障注入实验
    
    Args:
        fault_name: 故障名称
        output_dir: 输出目录
        duration: 实验持续时间（秒）
    """
    if output_dir is None:
        output_dir = f"./data/raw/{fault_name}"
    
    print("=" * 60)
    print(f"开始故障实验: {fault_name}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置文件路径
    gpu_metrics_file = os.path.join(output_dir, "gpu_metrics.csv")
    system_metrics_file = os.path.join(output_dir, "system_metrics.csv")
    training_log_file = os.path.join(output_dir, "train.log")
    training_metrics_file = os.path.join(output_dir, "training_metrics.csv")
    fault_config_file = os.path.join(output_dir, "fault_config.yaml")
    
    # 获取故障配置
    all_configs = FaultConfigFactory.get_all_fault_configs()
    if fault_name not in all_configs:
        print(f"未知故障类型: {fault_name}")
        return
    
    config = all_configs[fault_name]
    config.output_dir = output_dir
    config.logging_dir = output_dir
    
    print(f"故障类型: {fault_name}")
    print(f"输出目录: {output_dir}")
    print(f"预计持续时间: {duration}秒")
    
    # 创建故障注入配置
    fault_schedule = {}
    
    if fault_name in ["nan_loss", "oom", "no_convergence_low", "no_convergence_high"]:
        # 这些故障通过训练配置注入，不需要外部故障注入器
        pass
    else:
        # 需要外部故障注入的情况
        if "io_stress" in fault_name:
            fault_schedule["io_bottleneck"] = {
                "type": "io_stress",
                "delay": 30,  # 训练开始30秒后注入
                "duration": 60,
                "params": {
                    "target_dir": "/tmp",
                    "duration": 60
                }
            }
        elif "resource_competition" in fault_name:
            fault_schedule["resource_competition"] = {
                "type": "resource_competition",
                "delay": 30,
                "duration": 60,
                "params": {
                    "competitor_type": "gpu",
                    "duration": 60
                }
            }
        elif "process_kill" in fault_name:
            fault_schedule["process_termination"] = {
                "type": "process_kill",
                "delay": 120,  # 训练2分钟后终止
                "duration": 0,
                "params": {
                    "target_process_name": "python",
                    "target_cmdline_pattern": "train.py"
                }
            }
    
    # 保存故障配置
    with open(fault_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(fault_schedule, f, default_flow_style=False, allow_unicode=True)
    
    # 启动监控和实验
    monitors = []
    fault_injector = None
    train_process = None
    
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
        
        # 3. 启动故障注入器（如果需要）
        if fault_schedule:
            print("启动故障注入器...")
            fault_injector = FaultInjector()
            for fault_id, fault_config in fault_schedule.items():
                fault_injector.schedule_fault(
                    fault_type=fault_config["type"],
                    delay=fault_config["delay"],
                    duration=fault_config["duration"],
                    **fault_config["params"]
                )
            fault_injector.start()
        
        # 4. 启动训练任务
        print("启动训练任务...")
        train_cmd = [
            sys.executable, "-m", "src.executor.train",
            "--config-name", fault_name
        ]
        
        # 重定向训练日志
        with open(training_log_file, 'w') as log_file:
            train_process = subprocess.Popen(
                train_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent)
            )
        
        # 5. 启动日志解析
        print("启动日志解析...")
        time.sleep(2)  # 等待日志文件创建
        log_parser = LogParser(
            log_file=training_log_file,
            output_file=training_metrics_file,
            interval=1
        )
        log_parser.start()
        monitors.append(log_parser)
        
        # 6. 等待实验完成
        print("等待实验完成...")
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
        if train_process and train_process.poll() is None:
            print("等待训练进程完成...")
            try:
                train_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("训练进程超时，强制终止")
                train_process.kill()
        
        print(f"故障实验 {fault_name} 完成")
        
    except Exception as e:
        print(f"实验过程中发生错误: {e}")
        
    finally:
        # 停止故障注入器
        if fault_injector:
            try:
                fault_injector.stop()
            except:
                pass
        
        # 停止所有监控
        print("停止监控...")
        for monitor in monitors:
            try:
                monitor.stop()
            except:
                pass
        
        # 确保训练进程已终止
        if train_process:
            try:
                if train_process.poll() is None:
                    train_process.terminate()
                    train_process.wait(timeout=10)
            except:
                pass
    
    # 检查生成的文件
    print("\n生成的文件:")
    for file_path in [gpu_metrics_file, system_metrics_file, training_log_file, training_metrics_file, fault_config_file]:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file_path}: {size} bytes")
        else:
            print(f"  {file_path}: 未生成")
    
    print(f"\n故障实验数据已保存到: {output_dir}")


def run_all_fault_experiments(base_output_dir: str = "./data/raw", duration: int = 300):
    """
    运行所有故障实验
    
    Args:
        base_output_dir: 基础输出目录
        duration: 每个实验的持续时间（秒）
    """
    print("开始运行所有故障实验")
    
    # 获取所有故障配置
    all_configs = FaultConfigFactory.get_all_fault_configs()
    
    # 添加外部故障注入实验
    external_faults = [
        "io_stress",
        "resource_competition_gpu",
        "process_kill"
    ]
    
    all_fault_names = list(all_configs.keys()) + external_faults
    
    print(f"将运行 {len(all_fault_names)} 个故障实验:")
    for i, fault_name in enumerate(all_fault_names, 1):
        print(f"  {i}. {fault_name}")
    
    # 逐个运行实验
    for i, fault_name in enumerate(all_fault_names, 1):
        print(f"\n{'='*60}")
        print(f"运行实验 {i}/{len(all_fault_names)}: {fault_name}")
        print(f"{'='*60}")
        
        output_dir = os.path.join(base_output_dir, fault_name)
        
        try:
            run_fault_experiment(
                fault_name=fault_name,
                output_dir=output_dir,
                duration=duration
            )
            
            # 实验间隔
            if i < len(all_fault_names):
                print(f"等待 30 秒后开始下一个实验...")
                time.sleep(30)
                
        except Exception as e:
            print(f"实验 {fault_name} 失败: {e}")
            continue
    
    print(f"\n所有故障实验完成！数据保存在: {base_output_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="故障注入实验脚本")
    parser.add_argument("--fault-name", help="单个故障名称")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--duration", type=int, default=300, help="实验持续时间（秒）")
    parser.add_argument("--run-all", action="store_true", help="运行所有故障实验")
    
    args = parser.parse_args()
    
    if args.run_all:
        base_dir = args.output_dir or "./data/raw"
        run_all_fault_experiments(
            base_output_dir=base_dir,
            duration=args.duration
        )
    elif args.fault_name:
        run_fault_experiment(
            fault_name=args.fault_name,
            output_dir=args.output_dir,
            duration=args.duration
        )
    else:
        print("请指定 --fault-name 或使用 --run-all")
        
        # 显示可用的故障类型
        all_configs = FaultConfigFactory.get_all_fault_configs()
        external_faults = ["io_stress", "resource_competition_gpu", "process_kill"]
        all_fault_names = list(all_configs.keys()) + external_faults
        
        print("\n可用的故障类型:")
        for fault_name in all_fault_names:
            print(f"  - {fault_name}")


if __name__ == "__main__":
    main()