#!/usr/bin/env python3
"""
系统演示脚本
在没有GPU的环境中演示故障注入和监控系统的功能
"""

import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.observer.system_monitor import SystemMonitor
from src.observer.log_parser import LogParser
from src.injector.fault_injector import FaultInjector
from src.aggregator.data_aggregator import DataAggregator
from src.aggregator.tse_matrix_builder import TSEMatrixBuilder
from src.utils.config_loader import ConfigLoader


def simulate_training_logs():
    """模拟训练日志生成"""
    log_file = "demo_training.log"
    
    # 模拟正常训练日志
    normal_logs = [
        "2025-11-05 15:30:00 - INFO - Training started",
        "2025-11-05 15:30:01 - INFO - Epoch 1/3, Step 1/100, Loss: 0.693, Accuracy: 0.520",
        "2025-11-05 15:30:02 - INFO - Epoch 1/3, Step 2/100, Loss: 0.680, Accuracy: 0.535",
        "2025-11-05 15:30:03 - INFO - Epoch 1/3, Step 3/100, Loss: 0.665, Accuracy: 0.548",
        "2025-11-05 15:30:04 - INFO - Epoch 1/3, Step 4/100, Loss: 0.650, Accuracy: 0.562",
        "2025-11-05 15:30:05 - INFO - Epoch 1/3, Step 5/100, Loss: 0.635, Accuracy: 0.575",
    ]
    
    # 模拟异常日志
    anomaly_logs = [
        "2025-11-05 15:30:06 - WARNING - Learning rate too high, loss unstable",
        "2025-11-05 15:30:07 - ERROR - Loss: nan, Accuracy: 0.000",
        "2025-11-05 15:30:08 - CRITICAL - NaN detected in loss computation",
        "2025-11-05 15:30:09 - ERROR - Training failed due to gradient explosion",
    ]
    
    with open(log_file, 'w') as f:
        # 写入正常日志
        for log in normal_logs:
            f.write(log + '\n')
            f.flush()
            time.sleep(1)
        
        # 写入异常日志
        for log in anomaly_logs:
            f.write(log + '\n')
            f.flush()
            time.sleep(1)
    
    print(f"✅ 模拟训练日志已生成: {log_file}")


def demo_monitoring():
    """演示监控功能"""
    print("\n🔍 启动系统监控...")
    
    # 启动系统监控
    system_monitor = SystemMonitor(output_file="demo_system_metrics.csv", interval=1)
    monitor_thread = threading.Thread(target=system_monitor.start)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 启动日志解析
    log_parser = LogParser(log_file="demo_training.log", output_file="demo_training_metrics.csv")
    parser_thread = threading.Thread(target=log_parser.start)
    parser_thread.daemon = True
    parser_thread.start()
    
    print("✅ 系统监控已启动")
    print("✅ 日志解析器已启动")
    
    return system_monitor, log_parser


def demo_fault_injection():
    """演示故障注入功能"""
    print("\n⚡ 演示故障注入...")
    
    # 创建故障注入器
    fault_injector = FaultInjector()
    
    # 模拟不同类型的故障
    fault_types = ["nan_loss", "oom", "io_bottleneck"]
    
    for fault_type in fault_types:
        print(f"🔥 模拟故障: {fault_type}")
        
        # 模拟故障注入调度
        fault_injector.schedule_fault(
            fault_type=fault_type,
            delay=1.0,  # 1秒后执行
            duration=3.0,  # 持续3秒
            intensity="medium"
        )
        time.sleep(2)
    
    print("✅ 故障注入演示完成")


def demo_data_aggregation():
    """演示数据聚合功能"""
    print("\n📊 演示数据聚合...")
    
    # 等待一些数据生成
    time.sleep(3)
    
    try:
        # 创建数据聚合器
        aggregator = DataAggregator(time_granularity=1)
        
        # 聚合数据
        aggregated_data = aggregator.aggregate_data(
            gpu_file="demo_gpu_metrics.csv",  # 不存在，会被跳过
            system_file="demo_system_metrics.csv",
            training_file="demo_training_metrics.csv"
        )
        
        if aggregated_data is not None and not aggregated_data.empty:
            print(f"✅ 数据聚合完成，共 {len(aggregated_data)} 条记录")
            print(f"📈 数据列: {list(aggregated_data.columns)}")
            
            # 保存聚合数据
            output_file = "demo_aggregated_data.csv"
            aggregated_data.to_csv(output_file, index=False)
            print(f"💾 聚合数据已保存: {output_file}")
            
            return aggregated_data
        else:
            print("⚠️ 没有足够的数据进行聚合")
            return None
            
    except Exception as e:
        print(f"❌ 数据聚合失败: {e}")
        return None


def demo_tse_matrix():
    """演示TSE-Matrix构建"""
    print("\n🏗️ 演示TSE-Matrix构建...")
    
    try:
        # 创建TSE-Matrix构建器
        tse_builder = TSEMatrixBuilder()
        
        # 模拟故障配置
        fault_config = {
            "fault_type": "nan_loss",
            "injection_time": "2025-11-05 15:30:06",
            "duration": 3,
            "end_time": "2025-11-05 15:30:09"
        }
        
        # 构建TSE-Matrix
        if os.path.exists("demo_aggregated_data.csv"):
            # 加载聚合数据
            aggregated_data = pd.read_csv("demo_aggregated_data.csv")
            aggregated_data['timestamp'] = pd.to_datetime(aggregated_data['timestamp'])
            
            tse_matrix, ground_truth = tse_builder.build_tse_matrix(
                aggregated_data=aggregated_data,
                fault_config=fault_config,
                experiment_start_time=datetime.strptime("2025-11-05 15:30:00", "%Y-%m-%d %H:%M:%S")
            )
            
            # 保存TSE-Matrix
            if tse_matrix is not None:
                tse_matrix.to_csv("demo_tse_matrix.csv", index=False)
            
            if tse_matrix is not None:
                print(f"✅ TSE-Matrix构建完成，形状: {tse_matrix.shape}")
                print(f"🏷️ Ground Truth标注: {tse_matrix['is_anomaly'].sum()} 个异常点")
                print(f"💾 TSE-Matrix已保存: demo_tse_matrix.csv")
            else:
                print("⚠️ TSE-Matrix构建失败")
        else:
            print("⚠️ 没有聚合数据文件，跳过TSE-Matrix构建")
            
    except Exception as e:
        print(f"❌ TSE-Matrix构建失败: {e}")


def main():
    """主演示函数"""
    print("🚀 主动故障注入与多源异常数据收集系统 - 演示模式")
    print("=" * 70)
    
    try:
        # 1. 启动监控
        system_monitor, log_parser = demo_monitoring()
        
        # 2. 模拟训练日志生成
        log_thread = threading.Thread(target=simulate_training_logs)
        log_thread.start()
        
        # 3. 演示故障注入
        demo_fault_injection()
        
        # 等待日志生成完成
        log_thread.join()
        
        # 4. 演示数据聚合
        demo_data_aggregation()
        
        # 5. 演示TSE-Matrix构建
        demo_tse_matrix()
        
        # 停止监控
        system_monitor.stop()
        log_parser.stop()
        
        print("\n🎉 演示完成！")
        print("\n📁 生成的文件:")
        demo_files = [
            "demo_training.log",
            "demo_system_metrics.csv", 
            "demo_training_metrics.csv",
            "demo_aggregated_data.csv",
            "demo_tse_matrix.csv"
        ]
        
        for file in demo_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  ✅ {file} ({size} bytes)")
            else:
                print(f"  ❌ {file} (未生成)")
        
        print("\n💡 提示: 在有GPU的环境中运行完整实验:")
        print("  python3 experiments/orchestrator.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()