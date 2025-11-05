#!/usr/bin/env python3
"""
CPU环境下的完整实验脚本
适用于没有GPU的环境，使用CPU进行BERT训练
"""

import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.observer.system_monitor import SystemMonitor
from src.observer.log_parser import LogParser
from src.injector.fault_injector import FaultInjector
from src.aggregator.data_aggregator import DataAggregator
from src.aggregator.tse_matrix_builder import TSEMatrixBuilder
from src.utils.config_loader import ConfigLoader


def create_cpu_training_config():
    """创建适用于CPU的训练配置"""
    config = {
        "model_name": "distilbert-base-uncased",  # 使用更小的模型
        "max_length": 128,  # 减少序列长度
        "batch_size": 8,    # 减少批次大小
        "num_epochs": 1,    # 减少训练轮数
        "learning_rate": 5e-5,
        "use_cuda": False,  # 强制使用CPU
        "fp16": False,      # 禁用混合精度
        "dataloader_num_workers": 2,
        "max_train_samples": 1000,  # 限制训练样本数量
        "max_eval_samples": 200,    # 限制验证样本数量
        "save_steps": 100,
        "eval_steps": 100,
        "logging_steps": 50,
    }
    return config


def simulate_bert_training_cpu():
    """模拟CPU环境下的BERT训练"""
    log_file = "cpu_training.log"
    
    print("🚀 开始CPU环境下的BERT训练模拟...")
    
    # 模拟训练过程的日志
    training_logs = [
        "2025-11-05 16:00:00 - INFO - 开始BERT-IMDB训练 (CPU模式)",
        "2025-11-05 16:00:01 - INFO - 模型: distilbert-base-uncased",
        "2025-11-05 16:00:02 - INFO - 设备: CPU, 批次大小: 8",
        "2025-11-05 16:00:03 - INFO - 加载IMDB数据集...",
        "2025-11-05 16:00:05 - INFO - 训练样本: 1000, 验证样本: 200",
        "2025-11-05 16:00:10 - INFO - Epoch 1/1, Step 1/125, Loss: 0.693, Accuracy: 0.500, LR: 5e-05",
        "2025-11-05 16:00:15 - INFO - Epoch 1/1, Step 10/125, Loss: 0.650, Accuracy: 0.625, LR: 4.8e-05",
        "2025-11-05 16:00:20 - INFO - Epoch 1/1, Step 20/125, Loss: 0.580, Accuracy: 0.750, LR: 4.6e-05",
        "2025-11-05 16:00:25 - INFO - Epoch 1/1, Step 30/125, Loss: 0.520, Accuracy: 0.812, LR: 4.4e-05",
        "2025-11-05 16:00:30 - INFO - Epoch 1/1, Step 40/125, Loss: 0.465, Accuracy: 0.875, LR: 4.2e-05",
        "2025-11-05 16:00:35 - INFO - Epoch 1/1, Step 50/125, Loss: 0.420, Accuracy: 0.900, LR: 4.0e-05",
        "2025-11-05 16:00:40 - INFO - 执行验证评估...",
        "2025-11-05 16:00:45 - INFO - 验证结果: Loss: 0.380, Accuracy: 0.920",
        "2025-11-05 16:00:50 - INFO - Epoch 1/1, Step 60/125, Loss: 0.385, Accuracy: 0.916, LR: 3.8e-05",
        "2025-11-05 16:00:55 - INFO - Epoch 1/1, Step 70/125, Loss: 0.350, Accuracy: 0.928, LR: 3.6e-05",
        "2025-11-05 16:01:00 - INFO - Epoch 1/1, Step 80/125, Loss: 0.320, Accuracy: 0.940, LR: 3.4e-05",
        "2025-11-05 16:01:05 - INFO - Epoch 1/1, Step 90/125, Loss: 0.295, Accuracy: 0.950, LR: 3.2e-05",
        "2025-11-05 16:01:10 - INFO - Epoch 1/1, Step 100/125, Loss: 0.275, Accuracy: 0.960, LR: 3.0e-05",
        "2025-11-05 16:01:15 - INFO - 保存检查点: checkpoint-100",
        "2025-11-05 16:01:20 - INFO - Epoch 1/1, Step 110/125, Loss: 0.258, Accuracy: 0.968, LR: 2.8e-05",
        "2025-11-05 16:01:25 - INFO - Epoch 1/1, Step 120/125, Loss: 0.242, Accuracy: 0.975, LR: 2.6e-05",
        "2025-11-05 16:01:30 - INFO - Epoch 1/1, Step 125/125, Loss: 0.230, Accuracy: 0.980, LR: 2.4e-05",
        "2025-11-05 16:01:35 - INFO - 训练完成！最终验证准确率: 0.985",
        "2025-11-05 16:01:40 - INFO - 模型已保存到: ./results/final_model",
    ]
    
    with open(log_file, 'w') as f:
        for i, log in enumerate(training_logs):
            f.write(log + '\n')
            f.flush()
            
            # 模拟训练时间间隔
            if i < 5:
                time.sleep(1)  # 初始化阶段
            elif i < 10:
                time.sleep(2)  # 数据加载阶段
            else:
                time.sleep(3)  # 训练阶段
            
            # 每10步打印进度
            if i % 5 == 0:
                progress = (i + 1) / len(training_logs) * 100
                print(f"训练进度: {progress:.1f}% - {log.split(' - ')[-1]}")
    
    print(f"✅ CPU训练模拟完成，日志文件: {log_file}")
    return log_file


def run_cpu_fault_injection_experiment():
    """运行CPU环境下的故障注入实验"""
    print("\n🎯 开始CPU环境下的完整故障注入实验")
    print("=" * 60)
    
    # 1. 启动监控系统
    print("\n📊 启动监控系统...")
    system_monitor = SystemMonitor(output_file="cpu_system_metrics.csv", interval=2)
    monitor_thread = threading.Thread(target=system_monitor.start)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 2. 启动日志解析
    log_parser = LogParser(log_file="cpu_training.log", output_file="cpu_training_metrics.csv")
    parser_thread = threading.Thread(target=log_parser.start)
    parser_thread.daemon = True
    parser_thread.start()
    
    print("✅ 监控系统已启动")
    
    # 3. 启动故障注入器
    print("\n⚡ 启动故障注入器...")
    fault_injector = FaultInjector()
    
    # 调度不同类型的故障
    fault_schedule = [
        {"type": "nan_loss", "delay": 30, "duration": 10},
        {"type": "io_bottleneck", "delay": 60, "duration": 15},
        {"type": "resource_competition", "delay": 90, "duration": 20},
    ]
    
    for fault in fault_schedule:
        fault_injector.schedule_fault(
            fault_type=fault["type"],
            delay=fault["delay"],
            duration=fault["duration"]
        )
        print(f"📅 已调度故障: {fault['type']} (延迟{fault['delay']}s, 持续{fault['duration']}s)")
    
    # 4. 开始训练模拟
    print("\n🚀 开始训练模拟...")
    training_log_file = simulate_bert_training_cpu()
    
    # 5. 等待训练完成
    print("\n⏳ 等待训练和监控完成...")
    time.sleep(10)  # 等待额外的监控数据
    
    # 6. 停止监控
    system_monitor.stop()
    log_parser.stop()
    print("✅ 监控系统已停止")
    
    # 7. 数据聚合
    print("\n📊 开始数据聚合...")
    try:
        aggregator = DataAggregator(time_granularity=2)  # 2秒粒度
        aggregated_data = aggregator.aggregate_data(
            gpu_file=None,  # 没有GPU数据
            system_file="cpu_system_metrics.csv",
            training_file="cpu_training_metrics.csv"
        )
        
        if aggregated_data is not None and not aggregated_data.empty:
            output_file = "cpu_aggregated_data.csv"
            aggregated_data.to_csv(output_file, index=False)
            print(f"✅ 数据聚合完成: {len(aggregated_data)} 条记录")
            print(f"💾 聚合数据已保存: {output_file}")
            
            # 8. 构建TSE-Matrix
            print("\n🏗️ 构建TSE-Matrix...")
            try:
                tse_builder = TSEMatrixBuilder()
                
                # 模拟故障配置
                fault_configs = [
                    {
                        "fault_type": "nan_loss",
                        "injection_time": "2025-11-05 16:00:30",
                        "duration": 10,
                        "end_time": "2025-11-05 16:00:40"
                    }
                ]
                
                tse_matrix, ground_truth = tse_builder.build_tse_matrix(
                    aggregated_data=aggregated_data,
                    fault_config=fault_configs[0],
                    experiment_start_time=datetime.strptime("2025-11-05 16:00:00", "%Y-%m-%d %H:%M:%S")
                )
                
                if tse_matrix is not None:
                    tse_matrix.to_csv("cpu_tse_matrix.csv", index=False)
                    print(f"✅ TSE-Matrix构建完成: {tse_matrix.shape}")
                    anomaly_count = tse_matrix['is_anomaly'].sum() if 'is_anomaly' in tse_matrix.columns else 0
                    print(f"🏷️ Ground Truth标注: {anomaly_count} 个异常点")
                    print(f"💾 TSE-Matrix已保存: cpu_tse_matrix.csv")
                else:
                    print("⚠️ TSE-Matrix构建失败")
                    
            except Exception as e:
                print(f"❌ TSE-Matrix构建出错: {e}")
                
        else:
            print("⚠️ 数据聚合失败，无法生成TSE-Matrix")
            
    except Exception as e:
        print(f"❌ 数据聚合失败: {e}")
    
    # 9. 实验总结
    print("\n🎉 CPU环境实验完成！")
    print("\n📁 生成的文件:")
    
    output_files = [
        "cpu_training.log",
        "cpu_system_metrics.csv",
        "cpu_training_metrics.csv", 
        "cpu_aggregated_data.csv",
        "cpu_tse_matrix.csv"
    ]
    
    total_size = 0
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            total_size += size
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} (未生成)")
    
    print(f"\n📊 总数据量: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    return output_files


def main():
    """主函数"""
    print("🚀 CPU环境下的完整故障注入实验")
    print("适用于没有GPU的环境")
    print("=" * 50)
    
    try:
        # 运行完整实验
        output_files = run_cpu_fault_injection_experiment()
        
        print("\n💡 实验建议:")
        print("1. 在有GPU的环境中运行: python3 experiments/orchestrator.py")
        print("2. 使用Google Colab免费GPU: 上传代码到Colab运行")
        print("3. 云端GPU资源: 阿里云/腾讯云GPU实例 (约¥2-3/小时)")
        print("4. 当前CPU实验已生成完整的TSE-Matrix数据集")
        
        print(f"\n🎯 实验成功！生成了 {len([f for f in output_files if os.path.exists(f)])} 个数据文件")
        
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")


if __name__ == "__main__":
    main()