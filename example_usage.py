#!/usr/bin/env python3
"""
主动故障注入与多源异常数据收集系统使用示例
Example Usage of Active Fault Injection and Multi-source Anomaly Data Collection System
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader, ExperimentConfigManager


def main():
    """主函数：演示系统使用方法"""
    
    print("🚀 主动故障注入与多源异常数据收集系统")
    print("=" * 60)
    
    # 1. 初始化配置管理器
    print("\n📋 1. 初始化配置管理器")
    config_loader = ConfigLoader()
    experiment_manager = ExperimentConfigManager(config_loader)
    
    # 2. 加载配置文件
    print("\n📁 2. 加载配置文件")
    try:
        training_config = config_loader.load_training_config()
        fault_config = config_loader.load_fault_injection_config()
        monitoring_config = config_loader.load_monitoring_config()
        
        print(f"✅ 训练配置加载成功: {len(training_config)} 个配置项")
        print(f"✅ 故障注入配置加载成功: {len(fault_config)} 个配置项")
        print(f"✅ 监控配置加载成功: {len(monitoring_config)} 个配置项")
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return
    
    # 3. 创建实验配置
    print("\n🧪 3. 创建实验配置")
    
    # 基线实验配置
    baseline_config = experiment_manager.create_experiment_config(
        experiment_name="bert_imdb_baseline",
        fault_type=None
    )
    print(f"✅ 基线实验配置: {baseline_config['experiment_name']}")
    
    # 故障注入实验配置
    fault_types = ["nan_loss", "oom", "non_convergence", "io_bottleneck", "resource_competition", "process_termination"]
    
    for fault_type in fault_types:
        fault_config = experiment_manager.create_experiment_config(
            experiment_name=f"bert_imdb_{fault_type}",
            fault_type=fault_type
        )
        print(f"✅ {fault_type} 故障实验配置: {fault_config['experiment_name']}")
    
    # 4. 获取监控指标配置
    print("\n📊 4. 监控指标配置")
    metrics = experiment_manager.get_monitoring_metrics()
    
    print(f"GPU 监控指标: {metrics['gpu']}")
    print(f"系统监控指标数量: {len(metrics['system'])}")
    print(f"训练日志指标: {metrics['training']}")
    
    # 5. 获取故障注入计划
    print("\n⚡ 5. 故障注入计划示例")
    for fault_type in ["nan_loss", "io_bottleneck"]:
        schedule = experiment_manager.get_fault_injection_schedule(fault_type)
        step_based = len(schedule['step_based'])
        time_based = len(schedule['time_based'])
        print(f"{fault_type}: {step_based} 个基于步数的注入, {time_based} 个基于时间的注入")
    
    # 6. 系统使用指南
    print("\n📖 6. 系统使用指南")
    print("""
使用步骤:
1. 基线实验: python experiments/baseline_run.py
2. 故障实验: python experiments/fault_experiments.py --fault_type nan_loss
3. 完整流程: python experiments/orchestrator.py
4. 数据聚合: 自动在 orchestrator.py 中完成
5. TSE-Matrix构建: 自动生成带Ground Truth标注的数据集

输出文件:
- monitoring_data/: 监控数据 (GPU, 系统, 训练指标)
- fault_experiments/: 故障实验结果
- tse_matrix/: TSE-Matrix 和 Ground Truth 标注
- experiment_reports/: 实验报告和元数据
    """)
    
    print("\n🎉 系统初始化完成！准备开始实验。")


if __name__ == "__main__":
    main()