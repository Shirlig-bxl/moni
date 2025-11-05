"""
TSE-Matrix构建器
将聚合后的多源数据转换为TSE-Matrix格式，并添加Ground Truth标注
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import argparse
import yaml

from .data_aggregator import DataAggregator


class TSEMatrixBuilder:
    """TSE-Matrix构建器"""
    
    def __init__(self):
        self.tse_matrix = None
        self.ground_truth = None
        
        print("TSE-Matrix构建器初始化完成")
    
    def load_fault_annotations(self, fault_config_file: str) -> Dict[str, Any]:
        """
        加载故障注入配置，用于生成Ground Truth
        
        Args:
            fault_config_file: 故障配置文件路径
            
        Returns:
            故障配置字典
        """
        try:
            with open(fault_config_file, 'r', encoding='utf-8') as f:
                fault_config = yaml.safe_load(f)
            
            print(f"加载故障配置: {len(fault_config)} 个故障")
            return fault_config
            
        except Exception as e:
            print(f"加载故障配置失败: {e}")
            return {}
    
    def create_ground_truth_labels(self, df: pd.DataFrame, fault_config: Dict[str, Any], 
                                 experiment_start_time: datetime) -> pd.Series:
        """
        根据故障配置创建Ground Truth标签
        
        Args:
            df: 聚合数据DataFrame
            fault_config: 故障配置
            experiment_start_time: 实验开始时间
            
        Returns:
            Ground Truth标签Series
        """
        print("创建Ground Truth标签...")
        
        # 初始化标签为0（正常）
        labels = pd.Series(0, index=df.index, name='is_anomaly')
        
        for fault_name, fault_info in fault_config.items():
            fault_start = experiment_start_time + timedelta(seconds=fault_info['delay'])
            fault_end = fault_start + timedelta(seconds=fault_info.get('duration', 0))
            
            # 找到故障时间窗口内的记录
            if fault_info.get('duration', 0) > 0:
                # 持续性故障
                mask = (df['timestamp'] >= fault_start) & (df['timestamp'] <= fault_end)
            else:
                # 瞬时故障（如进程终止）
                mask = (df['timestamp'] >= fault_start) & (df['timestamp'] <= fault_start + timedelta(seconds=10))
            
            labels[mask] = 1
            
            print(f"故障 {fault_name}: {mask.sum()} 个时间点标记为异常")
        
        # 基于事件列的自动标注
        event_cols = [col for col in df.columns if col.startswith('event_')]
        for col in event_cols:
            if col in df.columns:
                event_mask = df[col] > 0
                labels[event_mask] = 1
                print(f"事件 {col}: {event_mask.sum()} 个时间点标记为异常")
        
        anomaly_count = (labels == 1).sum()
        normal_count = (labels == 0).sum()
        
        print(f"Ground Truth标签创建完成: {anomaly_count} 异常, {normal_count} 正常")
        
        return labels
    
    def build_tse_matrix(self, aggregated_data: pd.DataFrame, 
                        fault_config: Optional[Dict[str, Any]] = None,
                        experiment_start_time: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        构建TSE-Matrix
        
        Args:
            aggregated_data: 聚合后的数据
            fault_config: 故障配置（可选）
            experiment_start_time: 实验开始时间（可选）
            
        Returns:
            (TSE-Matrix DataFrame, Ground Truth Series)
        """
        print("构建TSE-Matrix...")
        
        # 复制数据
        tse_matrix = aggregated_data.copy()
        
        # 选择特征列（排除时间戳和非数值列）
        feature_cols = []
        
        # GPU特征
        gpu_cols = [col for col in tse_matrix.columns if col.startswith('gpu_') and 
                   tse_matrix[col].dtype in ['int64', 'float64']]
        feature_cols.extend(gpu_cols)
        
        # 系统特征
        sys_cols = [col for col in tse_matrix.columns if col.startswith('sys_') and 
                   tse_matrix[col].dtype in ['int64', 'float64']]
        feature_cols.extend(sys_cols)
        
        # 训练特征
        train_cols = [col for col in tse_matrix.columns if col.startswith('train_') and 
                     tse_matrix[col].dtype in ['int64', 'float64']]
        feature_cols.extend(train_cols)
        
        # 评估特征
        eval_cols = [col for col in tse_matrix.columns if col.startswith('eval_') and 
                    tse_matrix[col].dtype in ['int64', 'float64']]
        feature_cols.extend(eval_cols)
        
        # 事件特征
        event_cols = [col for col in tse_matrix.columns if col.startswith('event_')]
        feature_cols.extend(event_cols)
        
        # 派生特征
        derived_cols = [col for col in tse_matrix.columns if col in [
            'gpu_memory_utilization_ratio', 'train_step_rate', 'train_loss_change_rate',
            'train_loss_moving_avg', 'total_anomaly_events', 'elapsed_seconds'
        ]]
        feature_cols.extend(derived_cols)
        
        # 确保特征列存在
        feature_cols = [col for col in feature_cols if col in tse_matrix.columns]
        
        print(f"选择了 {len(feature_cols)} 个特征")
        
        # 构建特征矩阵
        feature_matrix = tse_matrix[['timestamp'] + feature_cols].copy()
        
        # 处理无穷值和NaN
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
        feature_matrix = feature_matrix.fillna(0)
        
        # 创建Ground Truth标签
        if fault_config and experiment_start_time:
            ground_truth = self.create_ground_truth_labels(
                tse_matrix, fault_config, experiment_start_time
            )
        else:
            # 基于事件列创建简单标签
            ground_truth = pd.Series(0, index=tse_matrix.index, name='is_anomaly')
            
            event_cols = [col for col in tse_matrix.columns if col.startswith('event_')]
            if event_cols:
                event_mask = tse_matrix[event_cols].sum(axis=1) > 0
                ground_truth[event_mask] = 1
                
                print(f"基于事件创建Ground Truth: {event_mask.sum()} 个异常点")
        
        self.tse_matrix = feature_matrix
        self.ground_truth = ground_truth
        
        print(f"TSE-Matrix构建完成: {len(feature_matrix)} × {len(feature_cols)} 矩阵")
        
        return feature_matrix, ground_truth
    
    def save_tse_matrix(self, output_dir: str, experiment_name: str = "experiment"):
        """
        保存TSE-Matrix和Ground Truth
        
        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        if self.tse_matrix is None or self.ground_truth is None:
            print("没有TSE-Matrix数据可保存")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存TSE-Matrix
        matrix_file = os.path.join(output_dir, f"{experiment_name}_tse_matrix.csv")
        self.tse_matrix.to_csv(matrix_file, index=False)
        print(f"TSE-Matrix已保存到: {matrix_file}")
        
        # 保存Ground Truth
        gt_file = os.path.join(output_dir, f"{experiment_name}_ground_truth.csv")
        gt_df = pd.DataFrame({
            'timestamp': self.tse_matrix['timestamp'],
            'is_anomaly': self.ground_truth
        })
        gt_df.to_csv(gt_file, index=False)
        print(f"Ground Truth已保存到: {gt_file}")
        
        # 保存合并数据（用于分析）
        combined_file = os.path.join(output_dir, f"{experiment_name}_combined.csv")
        combined_df = self.tse_matrix.copy()
        combined_df['is_anomaly'] = self.ground_truth
        combined_df.to_csv(combined_file, index=False)
        print(f"合并数据已保存到: {combined_file}")
        
        # 保存元数据
        metadata = {
            'experiment_name': experiment_name,
            'total_records': len(self.tse_matrix),
            'total_features': len(self.tse_matrix.columns) - 1,  # 减去timestamp
            'anomaly_count': int(self.ground_truth.sum()),
            'normal_count': int((self.ground_truth == 0).sum()),
            'anomaly_ratio': float(self.ground_truth.mean()),
            'time_range': {
                'start': str(self.tse_matrix['timestamp'].min()),
                'end': str(self.tse_matrix['timestamp'].max()),
                'duration_seconds': (
                    self.tse_matrix['timestamp'].max() - 
                    self.tse_matrix['timestamp'].min()
                ).total_seconds()
            },
            'features': self.tse_matrix.columns.tolist()
        }
        
        metadata_file = os.path.join(output_dir, f"{experiment_name}_metadata.yaml")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
        print(f"元数据已保存到: {metadata_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取TSE-Matrix统计信息
        
        Returns:
            统计信息字典
        """
        if self.tse_matrix is None or self.ground_truth is None:
            return {}
        
        stats = {
            'matrix_shape': self.tse_matrix.shape,
            'feature_count': len(self.tse_matrix.columns) - 1,  # 减去timestamp
            'total_records': len(self.tse_matrix),
            'anomaly_statistics': {
                'total_anomalies': int(self.ground_truth.sum()),
                'total_normal': int((self.ground_truth == 0).sum()),
                'anomaly_ratio': float(self.ground_truth.mean())
            },
            'feature_statistics': {}
        }
        
        # 特征统计
        numeric_cols = self.tse_matrix.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'timestamp':
                stats['feature_statistics'][col] = {
                    'mean': float(self.tse_matrix[col].mean()),
                    'std': float(self.tse_matrix[col].std()),
                    'min': float(self.tse_matrix[col].min()),
                    'max': float(self.tse_matrix[col].max()),
                    'missing_count': int(self.tse_matrix[col].isnull().sum())
                }
        
        return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="TSE-Matrix构建工具")
    parser.add_argument("--gpu-file", required=True, help="GPU监控CSV文件")
    parser.add_argument("--system-file", required=True, help="系统监控CSV文件")
    parser.add_argument("--training-file", required=True, help="训练指标CSV文件")
    parser.add_argument("--fault-config", help="故障配置YAML文件")
    parser.add_argument("--experiment-start", help="实验开始时间 (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--output-dir", "-o", required=True, help="输出目录")
    parser.add_argument("--experiment-name", default="experiment", help="实验名称")
    parser.add_argument("--time-granularity", type=int, default=1, help="时间粒度（秒）")
    
    args = parser.parse_args()
    
    # 数据聚合
    print("步骤1: 数据聚合")
    aggregator = DataAggregator(time_granularity=args.time_granularity)
    aggregated_data = aggregator.aggregate_data(
        gpu_file=args.gpu_file,
        system_file=args.system_file,
        training_file=args.training_file
    )
    
    if aggregated_data.empty:
        print("数据聚合失败")
        return
    
    # TSE-Matrix构建
    print("\n步骤2: TSE-Matrix构建")
    builder = TSEMatrixBuilder()
    
    # 加载故障配置
    fault_config = None
    experiment_start_time = None
    
    if args.fault_config and os.path.exists(args.fault_config):
        fault_config = builder.load_fault_annotations(args.fault_config)
    
    if args.experiment_start:
        try:
            experiment_start_time = datetime.strptime(args.experiment_start, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            print("实验开始时间格式错误，应为: YYYY-MM-DD HH:MM:SS")
    
    # 构建TSE-Matrix
    tse_matrix, ground_truth = builder.build_tse_matrix(
        aggregated_data=aggregated_data,
        fault_config=fault_config,
        experiment_start_time=experiment_start_time
    )
    
    # 保存结果
    print("\n步骤3: 保存结果")
    builder.save_tse_matrix(args.output_dir, args.experiment_name)
    
    # 打印统计信息
    stats = builder.get_statistics()
    print(f"\nTSE-Matrix统计:")
    print(f"矩阵大小: {stats['matrix_shape']}")
    print(f"特征数量: {stats['feature_count']}")
    print(f"异常记录: {stats['anomaly_statistics']['total_anomalies']}")
    print(f"正常记录: {stats['anomaly_statistics']['total_normal']}")
    print(f"异常比例: {stats['anomaly_statistics']['anomaly_ratio']:.3f}")


if __name__ == "__main__":
    main()