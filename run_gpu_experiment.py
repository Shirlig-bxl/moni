#!/usr/bin/env python3
"""
GPU环境完整实验脚本
专为Google Colab GPU环境设计
"""

import os
import sys
import time
import threading
import csv
import json
from datetime import datetime
from pathlib import Path
import subprocess


def check_gpu_environment():
    """检查GPU环境"""
    print("🔍 检查GPU环境...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            print(f"✅ GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("❌ CUDA不可用，请检查Colab运行时设置")
            print("💡 解决方案: 运行时 → 更改运行时类型 → GPU")
            return False
            
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def install_dependencies():
    """安装必要的依赖"""
    print("📦 安装依赖包...")
    
    packages = [
        "torch torchvision torchaudio",
        "transformers datasets accelerate evaluate",
        "pandas numpy scipy scikit-learn",
        "psutil nvidia-ml-py3",
        "pyyaml loguru tqdm"
    ]
    
    for package in packages:
        print(f"安装: {package}")
        result = subprocess.run(f"pip install {package}", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ 安装失败: {package}")
        else:
            print(f"✅ 安装成功: {package}")


class GPUSystemMonitor:
    """GPU和系统监控器"""
    
    def __init__(self, output_file, interval=1):
        self.output_file = output_file
        self.interval = interval
        self.running = False
        
    def start(self):
        """开始监控"""
        import psutil
        
        # 尝试导入GPU监控
        try:
            import nvidia_ml_py as nvml
            nvml.nvmlInit()
            gpu_available = True
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
        except:
            try:
                import pynvml as nvml
                nvml.nvmlInit()
                gpu_available = True
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
            except:
                gpu_available = False
                print("⚠️ GPU监控不可用，仅监控系统指标")
        
        self.running = True
        
        # 创建CSV文件
        headers = ['timestamp', 'cpu_percent', 'memory_percent', 'disk_percent']
        if gpu_available:
            headers.extend(['gpu_util', 'gpu_memory_used', 'gpu_memory_total', 'gpu_temperature'])
        
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"✅ 监控已启动: {self.output_file}")
        
        while self.running:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 系统指标
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                row = [timestamp, cpu_percent, memory_percent, disk_percent]
                
                # GPU指标
                if gpu_available:
                    try:
                        # GPU利用率
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                        
                        # GPU内存
                        mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_used = mem_info.used // 1024 // 1024  # MB
                        gpu_memory_total = mem_info.total // 1024 // 1024  # MB
                        
                        # GPU温度
                        gpu_temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        
                        row.extend([gpu_util, gpu_memory_used, gpu_memory_total, gpu_temp])
                        
                        print(f"[{timestamp}] GPU0: Util={gpu_util}%, Mem={gpu_memory_used}/{gpu_memory_total}MB ({gpu_memory_used/gpu_memory_total*100:.1f}%), Temp={gpu_temp}°C")
                        print(f"[{timestamp}] CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%")
                        
                    except Exception as e:
                        print(f"GPU监控错误: {e}")
                        row.extend([0, 0, 0, 0])
                else:
                    print(f"[{timestamp}] CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%")
                
                # 写入CSV
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def stop(self):
        """停止监控"""
        self.running = False
        print("✅ 监控已停止")


def run_real_bert_training():
    """运行真实的BERT训练（GPU版本）"""
    print("🚀 开始真实的BERT-IMDB训练（GPU版本）...")
    
    training_script = """
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from datetime import datetime
import sys

def main():
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 开始BERT-IMDB微调训练")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - GPU: {torch.cuda.get_device_name(0)}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载模型和分词器
    model_name = "distilbert-base-uncased"
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 加载数据集
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 加载IMDB数据集...")
    dataset = load_dataset("imdb")
    
    # 限制数据量以加快训练
    train_dataset = dataset["train"].select(range(1000))  # 只用1000个样本
    eval_dataset = dataset["test"].select(range(200))     # 只用200个样本
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 训练样本: {len(train_dataset)}, 验证样本: {len(eval_dataset)}")
    
    # 数据预处理
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./gpu_results",
        num_train_epochs=2,
        per_device_train_batch_size=16,  # 使用较大的batch size来充分利用GPU
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./gpu_logs",
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # 禁用wandb等
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),  # 启用混合精度训练
    )
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 评估函数
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 开始训练...")
    
    # 开始训练
    trainer.train()
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 训练完成！")
    
    # 最终评估
    eval_results = trainer.evaluate()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - 最终评估结果: {eval_results}")

if __name__ == "__main__":
    main()
"""
    
    # 写入训练脚本
    with open("gpu_bert_training.py", "w") as f:
        f.write(training_script)
    
    # 运行训练脚本并捕获输出
    print("执行GPU训练脚本...")
    process = subprocess.Popen(
        ["python", "gpu_bert_training.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 实时输出并保存日志
    log_file = "gpu_training.log"
    with open(log_file, "w") as f:
        for line in process.stdout:
            print(line.strip())
            f.write(line)
            f.flush()
    
    process.wait()
    print(f"✅ GPU训练完成，日志保存到: {log_file}")
    return log_file


def run_gpu_experiment():
    """运行完整的GPU实验"""
    print("🚀 GPU环境完整实验")
    print("=" * 50)
    
    # 1. 检查GPU环境
    if not check_gpu_environment():
        print("❌ GPU环境检查失败，退出实验")
        return
    
    # 2. 安装依赖（在Colab中可能需要）
    try:
        import transformers
        import datasets
        print("✅ 依赖包已安装")
    except ImportError:
        install_dependencies()
    
    # 3. 启动监控
    print("\n📊 启动GPU和系统监控...")
    monitor = GPUSystemMonitor("gpu_system_metrics.csv", interval=2)
    monitor_thread = threading.Thread(target=monitor.start)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    time.sleep(3)  # 等待监控启动
    
    # 4. 运行真实的BERT训练
    print("\n🚀 开始真实的BERT训练...")
    log_file = run_real_bert_training()
    
    # 5. 等待一段时间收集更多监控数据
    print("\n⏳ 等待监控数据收集...")
    time.sleep(10)
    
    # 6. 停止监控
    monitor.stop()
    time.sleep(2)
    
    # 7. 分析结果
    print("\n📊 实验结果分析...")
    
    files = [
        "gpu_training.log",
        "gpu_system_metrics.csv",
        "gpu_bert_training.py"
    ]
    
    total_size = 0
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            total_size += size
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} (未生成)")
    
    print(f"\n📊 总数据量: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    # 8. 检查GPU使用情况
    if os.path.exists("gpu_system_metrics.csv"):
        try:
            import pandas as pd
            df = pd.read_csv("gpu_system_metrics.csv")
            
            if 'gpu_util' in df.columns:
                max_gpu_util = df['gpu_util'].max()
                avg_gpu_util = df['gpu_util'].mean()
                max_gpu_memory = df['gpu_memory_used'].max()
                
                print(f"\n🎯 GPU使用情况分析:")
                print(f"  - 最大GPU利用率: {max_gpu_util:.1f}%")
                print(f"  - 平均GPU利用率: {avg_gpu_util:.1f}%")
                print(f"  - 最大GPU内存使用: {max_gpu_memory:.0f} MB")
                
                if max_gpu_util > 50:
                    print("  ✅ GPU被充分利用！")
                else:
                    print("  ⚠️ GPU利用率较低，可能需要调整batch size")
            else:
                print("  ❌ 未检测到GPU使用数据")
                
        except Exception as e:
            print(f"  ❌ 分析GPU使用情况失败: {e}")
    
    print("\n🎉 GPU实验完成！")
    
    if os.path.exists("gpu_system_metrics.csv"):
        print("\n💡 下一步建议:")
        print("1. 分析gpu_system_metrics.csv中的GPU使用模式")
        print("2. 基于真实GPU训练数据进行异常检测研究")
        print("3. 对比CPU和GPU环境下的系统行为差异")


def main():
    """主函数"""
    try:
        run_gpu_experiment()
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()