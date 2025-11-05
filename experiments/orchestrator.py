"""
实验编排器
自动化运行完整的实验流程：基线实验 -> 故障实验 -> 数据聚合 -> TSE-Matrix构建
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from .baseline_run import run_baseline_experiment
from .fault_experiments import run_all_fault_experiments
from aggregator.data_aggregator import DataAggregator
from aggregator.tse_matrix_builder import TSEMatrixBuilder


def run_complete_experiment_pipeline(
    output_base_dir: str = "./data",
    experiment_name: str = None,
    duration_per_experiment: int = 300,
    skip_baseline: bool = False,
    skip_faults: bool = False,
    skip_aggregation: bool = False
):
    """
    运行完整的实验流程
    
    Args:
        output_base_dir: 输出基础目录
        experiment_name: 实验名称
        duration_per_experiment: 每个实验的持续时间（秒）
        skip_baseline: 跳过基线实验
        skip_faults: 跳过故障实验
        skip_aggregation: 跳过数据聚合
    """
    
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 80)
    print("主动故障注入与多源异常数据收集系统")
    print("完整实验流程")
    print("=" * 80)
    print(f"实验名称: {experiment_name}")
    print(f"输出目录: {output_base_dir}")
    print(f"每个实验持续时间: {duration_per_experiment}秒")
    print("=" * 80)
    
    # 创建目录结构
    raw_data_dir = os.path.join(output_base_dir, "raw", experiment_name)
    processed_data_dir = os.path.join(output_base_dir, "processed", experiment_name)
    datasets_dir = os.path.join(output_base_dir, "datasets", experiment_name)
    
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    
    experiment_start_time = datetime.now()
    
    try:
        # 阶段1: 基线实验
        if not skip_baseline:
            print("\n" + "=" * 60)
            print("阶段1: 基线实验")
            print("=" * 60)
            
            baseline_dir = os.path.join(raw_data_dir, "baseline")
            run_baseline_experiment(
                output_dir=baseline_dir,
                duration=duration_per_experiment
            )
            
            print("基线实验完成")
            time.sleep(10)  # 短暂休息
        
        # 阶段2: 故障注入实验
        if not skip_faults:
            print("\n" + "=" * 60)
            print("阶段2: 故障注入实验")
            print("=" * 60)
            
            run_all_fault_experiments(
                base_output_dir=raw_data_dir,
                duration=duration_per_experiment
            )
            
            print("所有故障实验完成")
            time.sleep(10)  # 短暂休息
        
        # 阶段3: 数据聚合与TSE-Matrix构建
        if not skip_aggregation:
            print("\n" + "=" * 60)
            print("阶段3: 数据聚合与TSE-Matrix构建")
            print("=" * 60)
            
            # 获取所有实验目录
            experiment_dirs = []
            if os.path.exists(raw_data_dir):
                for item in os.listdir(raw_data_dir):
                    item_path = os.path.join(raw_data_dir, item)
                    if os.path.isdir(item_path):
                        experiment_dirs.append((item, item_path))
            
            print(f"找到 {len(experiment_dirs)} 个实验目录")
            
            # 为每个实验构建TSE-Matrix
            for exp_name, exp_dir in experiment_dirs:
                print(f"\n处理实验: {exp_name}")
                
                # 检查必要文件是否存在
                gpu_file = os.path.join(exp_dir, "gpu_metrics.csv")
                system_file = os.path.join(exp_dir, "system_metrics.csv")
                training_file = os.path.join(exp_dir, "training_metrics.csv")
                fault_config_file = os.path.join(exp_dir, "fault_config.yaml")
                
                missing_files = []
                if not os.path.exists(gpu_file):
                    missing_files.append("gpu_metrics.csv")
                if not os.path.exists(system_file):
                    missing_files.append("system_metrics.csv")
                if not os.path.exists(training_file):
                    missing_files.append("training_metrics.csv")
                
                if missing_files:
                    print(f"  跳过 {exp_name}，缺少文件: {missing_files}")
                    continue
                
                try:
                    # 数据聚合
                    print(f"  聚合数据...")
                    aggregator = DataAggregator(time_granularity=1)
                    aggregated_data = aggregator.aggregate_data(
                        gpu_file=gpu_file,
                        system_file=system_file,
                        training_file=training_file
                    )
                    
                    if aggregated_data.empty:
                        print(f"  {exp_name} 数据聚合失败")
                        continue
                    
                    # 保存聚合数据
                    aggregated_file = os.path.join(processed_data_dir, f"{exp_name}_aggregated.csv")
                    aggregator.save_aggregated_data(aggregated_file)
                    
                    # 构建TSE-Matrix
                    print(f"  构建TSE-Matrix...")
                    builder = TSEMatrixBuilder()
                    
                    # 加载故障配置（如果存在）
                    fault_config = None
                    if os.path.exists(fault_config_file):
                        fault_config = builder.load_fault_annotations(fault_config_file)
                    
                    # 构建TSE-Matrix
                    tse_matrix, ground_truth = builder.build_tse_matrix(
                        aggregated_data=aggregated_data,
                        fault_config=fault_config,
                        experiment_start_time=experiment_start_time
                    )
                    
                    # 保存TSE-Matrix
                    builder.save_tse_matrix(datasets_dir, exp_name)
                    
                    # 打印统计信息
                    stats = builder.get_statistics()
                    print(f"  TSE-Matrix: {stats['matrix_shape']}")
                    print(f"  异常比例: {stats['anomaly_statistics']['anomaly_ratio']:.3f}")
                    
                except Exception as e:
                    print(f"  处理 {exp_name} 时发生错误: {e}")
                    continue
            
            print("数据聚合与TSE-Matrix构建完成")
        
        # 生成实验报告
        print("\n" + "=" * 60)
        print("生成实验报告")
        print("=" * 60)
        
        generate_experiment_report(
            experiment_name=experiment_name,
            output_base_dir=output_base_dir,
            experiment_start_time=experiment_start_time
        )
        
        print("\n" + "=" * 80)
        print("完整实验流程执行完成！")
        print(f"实验名称: {experiment_name}")
        print(f"数据位置: {output_base_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n实验流程执行失败: {e}")
        raise


def generate_experiment_report(experiment_name: str, output_base_dir: str, experiment_start_time: datetime):
    """
    生成实验报告
    
    Args:
        experiment_name: 实验名称
        output_base_dir: 输出基础目录
        experiment_start_time: 实验开始时间
    """
    
    report_file = os.path.join(output_base_dir, f"{experiment_name}_report.md")
    
    # 收集统计信息
    raw_data_dir = os.path.join(output_base_dir, "raw", experiment_name)
    datasets_dir = os.path.join(output_base_dir, "datasets", experiment_name)
    
    experiment_count = 0
    total_data_size = 0
    tse_matrices = []
    
    # 统计原始数据
    if os.path.exists(raw_data_dir):
        for item in os.listdir(raw_data_dir):
            item_path = os.path.join(raw_data_dir, item)
            if os.path.isdir(item_path):
                experiment_count += 1
                
                # 计算目录大小
                for root, dirs, files in os.walk(item_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_data_size += os.path.getsize(file_path)
                        except:
                            pass
    
    # 统计TSE-Matrix
    if os.path.exists(datasets_dir):
        for file in os.listdir(datasets_dir):
            if file.endswith("_metadata.yaml"):
                tse_matrices.append(file.replace("_metadata.yaml", ""))
    
    # 生成报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# 实验报告: {experiment_name}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**实验开始时间**: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**实验持续时间**: {(datetime.now() - experiment_start_time).total_seconds():.0f} 秒\n\n")
        
        f.write("## 实验概览\n\n")
        f.write(f"- **实验数量**: {experiment_count}\n")
        f.write(f"- **原始数据大小**: {total_data_size / 1024 / 1024:.2f} MB\n")
        f.write(f"- **TSE-Matrix数量**: {len(tse_matrices)}\n\n")
        
        f.write("## 目录结构\n\n")
        f.write("```\n")
        f.write(f"{experiment_name}/\n")
        f.write("├── raw/                    # 原始监控数据\n")
        f.write("│   ├── baseline/          # 基线实验数据\n")
        f.write("│   ├── nan_loss/          # NaN Loss故障数据\n")
        f.write("│   ├── oom/               # OOM故障数据\n")
        f.write("│   └── ...\n")
        f.write("├── processed/             # 聚合后数据\n")
        f.write("└── datasets/              # TSE-Matrix数据集\n")
        f.write("```\n\n")
        
        f.write("## TSE-Matrix列表\n\n")
        for matrix_name in tse_matrices:
            f.write(f"- `{matrix_name}`\n")
        
        f.write("\n## 使用说明\n\n")
        f.write("1. **原始数据**: 位于 `raw/` 目录，包含GPU、系统监控和训练日志\n")
        f.write("2. **聚合数据**: 位于 `processed/` 目录，时间戳对齐的多源数据\n")
        f.write("3. **TSE-Matrix**: 位于 `datasets/` 目录，可直接用于异常检测算法训练\n\n")
        
        f.write("## 数据格式\n\n")
        f.write("- **GPU指标**: 利用率、显存使用、温度等\n")
        f.write("- **系统指标**: CPU、内存、磁盘I/O等\n")
        f.write("- **训练指标**: Loss、Accuracy、吞吐量等\n")
        f.write("- **事件指标**: 异常事件标记（OOM、NaN等）\n")
        f.write("- **Ground Truth**: 精确的异常标注\n\n")
    
    print(f"实验报告已生成: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实验编排器")
    parser.add_argument("--output-dir", default="./data", help="输出基础目录")
    parser.add_argument("--experiment-name", help="实验名称")
    parser.add_argument("--duration", type=int, default=300, help="每个实验持续时间（秒）")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过基线实验")
    parser.add_argument("--skip-faults", action="store_true", help="跳过故障实验")
    parser.add_argument("--skip-aggregation", action="store_true", help="跳过数据聚合")
    
    args = parser.parse_args()
    
    run_complete_experiment_pipeline(
        output_base_dir=args.output_dir,
        experiment_name=args.experiment_name,
        duration_per_experiment=args.duration,
        skip_baseline=args.skip_baseline,
        skip_faults=args.skip_faults,
        skip_aggregation=args.skip_aggregation
    )


if __name__ == "__main__":
    main()