"""
日志解析模块
实时解析训练日志，提取Loss、Accuracy、Throughput等指标
识别异常事件 (OOM, NaN, WARNING)
"""

import re
import time
import csv
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, TextIO
import threading
import argparse
from pathlib import Path


class LogParser:
    """日志解析器类"""
    
    def __init__(self, log_file: str, output_file: str = "training_metrics.csv", interval: int = 1):
        self.log_file = log_file
        self.output_file = output_file
        self.interval = interval
        self.running = False
        self.thread = None
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        
        # 编译正则表达式模式
        self._compile_patterns()
        
        # 文件位置跟踪
        self.file_position = 0
        
        print(f"日志解析器初始化完成")
        print(f"监控日志文件: {self.log_file}")
        print(f"输出文件: {self.output_file}")
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        
        # 训练指标模式
        self.patterns = {
            # Step信息: Step 10 | Loss: 0.693147 | LR: 2.00e-05 | Throughput: 16.50 samples/s | Step Time: 0.970s
            'step_info': re.compile(
                r'Step\s+(\d+)\s*\|\s*Loss:\s*([\d\.\-e]+)\s*\|\s*LR:\s*([\d\.\-e]+)\s*\|\s*Throughput:\s*([\d\.\-e]+)\s*samples/s\s*\|\s*Step Time:\s*([\d\.\-e]+)s'
            ),
            
            # 评估结果: eval_loss: 0.693147 | eval_accuracy: 0.500000
            'eval_metrics': re.compile(
                r'eval_(\w+):\s*([\d\.\-e]+)'
            ),
            
            # 时间戳: 2023-11-05 14:30:25
            'timestamp': re.compile(
                r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
            ),
            
            # 异常事件模式
            'nan_loss': re.compile(
                r'\[ANOMALY DETECTED\]\s*NaN/Inf Loss.*?step\s+(\d+).*?:\s*([\d\.\-e]*)',
                re.IGNORECASE
            ),
            
            'oom_error': re.compile(
                r'(CUDA.*out of memory|OutOfMemoryError|OOM)',
                re.IGNORECASE
            ),
            
            'high_loss': re.compile(
                r'\[ANOMALY DETECTED\]\s*High Loss.*?step\s+(\d+).*?:\s*([\d\.\-e]+)',
                re.IGNORECASE
            ),
            
            # 训练阶段识别
            'train_begin': re.compile(r'训练开始|Training.*begin', re.IGNORECASE),
            'train_end': re.compile(r'训练结束|Training.*end|Training.*complete', re.IGNORECASE),
            'eval_begin': re.compile(r'评估.*开始|Evaluation.*begin', re.IGNORECASE),
            'eval_end': re.compile(r'评估.*结束|Evaluation.*end', re.IGNORECASE),
            
            # 错误和警告
            'error': re.compile(r'ERROR|Error|error', re.IGNORECASE),
            'warning': re.compile(r'WARNING|Warning|warning', re.IGNORECASE),
        }
    
    def parse_line(self, line: str, line_timestamp: Optional[str] = None) -> Dict[str, Any]:
        """解析单行日志"""
        result = {
            'timestamp': line_timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'raw_line': line.strip()
        }
        
        # 提取时间戳
        timestamp_match = self.patterns['timestamp'].search(line)
        if timestamp_match:
            result['timestamp'] = timestamp_match.group(1)
        
        # 解析训练步骤信息
        step_match = self.patterns['step_info'].search(line)
        if step_match:
            result.update({
                'step': int(step_match.group(1)),
                'loss': float(step_match.group(2)),
                'learning_rate': float(step_match.group(3)),
                'throughput': float(step_match.group(4)),
                'step_time': float(step_match.group(5)),
                'metric_type': 'training_step'
            })
        
        # 解析评估指标
        eval_matches = self.patterns['eval_metrics'].findall(line)
        if eval_matches:
            result['metric_type'] = 'evaluation'
            for metric_name, metric_value in eval_matches:
                result[f'eval_{metric_name}'] = float(metric_value)
        
        # 检测异常事件
        self._detect_anomalies(line, result)
        
        # 检测训练阶段
        self._detect_phases(line, result)
        
        return result
    
    def _detect_anomalies(self, line: str, result: Dict[str, Any]):
        """检测异常事件"""
        
        # NaN Loss异常
        nan_match = self.patterns['nan_loss'].search(line)
        if nan_match:
            result.update({
                'event_nan_loss': 1,
                'anomaly_step': int(nan_match.group(1)) if nan_match.group(1) else None,
                'anomaly_value': nan_match.group(2) if nan_match.group(2) else 'NaN',
                'anomaly_type': 'nan_loss'
            })
        
        # OOM错误
        if self.patterns['oom_error'].search(line):
            result.update({
                'event_oom': 1,
                'anomaly_type': 'oom_error'
            })
        
        # 高Loss异常
        high_loss_match = self.patterns['high_loss'].search(line)
        if high_loss_match:
            result.update({
                'event_high_loss': 1,
                'anomaly_step': int(high_loss_match.group(1)),
                'anomaly_value': float(high_loss_match.group(2)),
                'anomaly_type': 'high_loss'
            })
        
        # 一般错误和警告
        if self.patterns['error'].search(line):
            result['event_error'] = 1
        
        if self.patterns['warning'].search(line):
            result['event_warning'] = 1
    
    def _detect_phases(self, line: str, result: Dict[str, Any]):
        """检测训练阶段"""
        
        if self.patterns['train_begin'].search(line):
            result['phase'] = 'train_begin'
        elif self.patterns['train_end'].search(line):
            result['phase'] = 'train_end'
        elif self.patterns['eval_begin'].search(line):
            result['phase'] = 'eval_begin'
        elif self.patterns['eval_end'].search(line):
            result['phase'] = 'eval_end'
    
    def read_new_lines(self) -> List[str]:
        """读取日志文件中的新行"""
        new_lines = []
        
        try:
            if not os.path.exists(self.log_file):
                return new_lines
            
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到上次读取的位置
                f.seek(self.file_position)
                
                # 读取新行
                new_lines = f.readlines()
                
                # 更新文件位置
                self.file_position = f.tell()
                
        except Exception as e:
            print(f"读取日志文件失败: {e}")
        
        return new_lines
    
    def write_csv_header(self):
        """写入CSV文件头"""
        fieldnames = [
            'timestamp', 'step', 'loss', 'learning_rate', 'throughput', 'step_time',
            'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall',
            'metric_type', 'phase',
            'event_nan_loss', 'event_oom', 'event_high_loss', 'event_error', 'event_warning',
            'anomaly_type', 'anomaly_step', 'anomaly_value',
            'raw_line'
        ]
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def monitor_loop(self):
        """监控循环"""
        print(f"开始日志监控，输出文件: {self.output_file}")
        print(f"监控间隔: {self.interval}秒")
        
        # 写入CSV头
        self.write_csv_header()
        
        while self.running:
            try:
                new_lines = self.read_new_lines()
                
                if new_lines:
                    parsed_data = []
                    
                    for line in new_lines:
                        if line.strip():  # 跳过空行
                            parsed_result = self.parse_line(line)
                            parsed_data.append(parsed_result)
                    
                    # 写入解析结果
                    if parsed_data:
                        self._write_parsed_data(parsed_data)
                        
                        # 打印重要事件
                        for data in parsed_data:
                            self._print_important_events(data)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(self.interval)
    
    def _write_parsed_data(self, parsed_data: List[Dict[str, Any]]):
        """写入解析后的数据"""
        fieldnames = [
            'timestamp', 'step', 'loss', 'learning_rate', 'throughput', 'step_time',
            'eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall',
            'metric_type', 'phase',
            'event_nan_loss', 'event_oom', 'event_high_loss', 'event_error', 'event_warning',
            'anomaly_type', 'anomaly_step', 'anomaly_value',
            'raw_line'
        ]
        
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            for data in parsed_data:
                # 填充缺失字段
                row = {field: data.get(field, '') for field in fieldnames}
                writer.writerow(row)
    
    def _print_important_events(self, data: Dict[str, Any]):
        """打印重要事件"""
        timestamp = data.get('timestamp', '')
        
        # 训练指标
        if data.get('metric_type') == 'training_step':
            step = data.get('step', 0)
            loss = data.get('loss', 0)
            throughput = data.get('throughput', 0)
            print(f"[{timestamp}] Step {step}: Loss={loss:.6f}, Throughput={throughput:.2f} samples/s")
        
        # 评估指标
        elif data.get('metric_type') == 'evaluation':
            eval_metrics = {k: v for k, v in data.items() if k.startswith('eval_') and v != ''}
            if eval_metrics:
                metrics_str = ', '.join([f"{k}={v:.6f}" for k, v in eval_metrics.items()])
                print(f"[{timestamp}] Evaluation: {metrics_str}")
        
        # 异常事件
        if data.get('event_nan_loss'):
            print(f"[{timestamp}] ⚠️  ANOMALY: NaN Loss detected at step {data.get('anomaly_step', 'unknown')}")
        
        if data.get('event_oom'):
            print(f"[{timestamp}] ⚠️  ANOMALY: Out of Memory Error detected")
        
        if data.get('event_high_loss'):
            print(f"[{timestamp}] ⚠️  ANOMALY: High Loss detected at step {data.get('anomaly_step', 'unknown')}: {data.get('anomaly_value', 'unknown')}")
        
        # 阶段变化
        if data.get('phase'):
            phase = data.get('phase')
            print(f"[{timestamp}] 📍 Phase: {phase}")
    
    def start(self):
        """启动监控"""
        if self.running:
            print("监控已在运行中")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self.monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print("日志监控已启动")
    
    def stop(self):
        """停止监控"""
        if not self.running:
            print("监控未在运行")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("日志监控已停止")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def parse_log_file(log_file: str, output_file: str = None) -> List[Dict[str, Any]]:
    """解析整个日志文件（非实时）"""
    
    if output_file is None:
        output_file = log_file.replace('.log', '_parsed.csv')
    
    parser = LogParser(log_file, output_file)
    
    print(f"解析日志文件: {log_file}")
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        parsed_data = []
        for line in lines:
            if line.strip():
                result = parser.parse_line(line)
                parsed_data.append(result)
        
        # 写入结果
        parser.write_csv_header()
        parser._write_parsed_data(parsed_data)
        
        print(f"解析完成，共处理 {len(parsed_data)} 行")
        print(f"结果保存到: {output_file}")
        
        return parsed_data
        
    except Exception as e:
        print(f"解析日志文件失败: {e}")
        return []


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="日志解析工具")
    parser.add_argument("log_file", help="要监控的日志文件路径")
    parser.add_argument("--output", "-o", help="输出CSV文件路径")
    parser.add_argument("--interval", "-i", type=int, default=1, help="监控间隔（秒）")
    parser.add_argument("--parse-only", action="store_true", help="仅解析现有文件，不进行实时监控")
    
    args = parser.parse_args()
    
    if args.parse_only:
        # 仅解析现有文件
        parse_log_file(args.log_file, args.output)
    else:
        # 实时监控
        output_file = args.output or "training_metrics.csv"
        log_parser = LogParser(args.log_file, output_file, args.interval)
        
        try:
            log_parser.start()
            print("按 Ctrl+C 停止监控")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            log_parser.stop()


if __name__ == "__main__":
    main()