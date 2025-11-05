"""
主训练脚本
基于Hugging Face Transformers的BERT-IMDB微调
支持故障注入和详细监控日志
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from transformers.trainer_callback import TrainerCallback
import evaluate

from .config import TrainingConfig, FaultConfigFactory


# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DetailedLoggingCallback(TrainerCallback):
    """详细日志回调，记录训练过程中的关键指标"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        logger.info("=" * 50)
        logger.info("训练开始")
        logger.info(f"总步数: {state.max_steps}")
        logger.info(f"总轮数: {args.num_train_epochs}")
        logger.info(f"批次大小: {args.per_device_train_batch_size}")
        logger.info(f"学习率: {args.learning_rate}")
        logger.info("=" * 50)
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            current_time = time.time()
            if hasattr(self, 'step_start_time'):
                step_duration = current_time - self.step_start_time
                self.step_times.append(step_duration)
                
                # 计算吞吐量 (samples/second)
                throughput = args.per_device_train_batch_size / step_duration if step_duration > 0 else 0
                
                # 记录详细指标
                log_msg = f"Step {state.global_step}"
                if 'loss' in logs:
                    log_msg += f" | Loss: {logs['loss']:.6f}"
                if 'learning_rate' in logs:
                    log_msg += f" | LR: {logs['learning_rate']:.2e}"
                log_msg += f" | Throughput: {throughput:.2f} samples/s"
                log_msg += f" | Step Time: {step_duration:.3f}s"
                
                logger.info(log_msg)
                
                # 检查异常值
                if 'loss' in logs:
                    if np.isnan(logs['loss']) or np.isinf(logs['loss']):
                        logger.error(f"[ANOMALY DETECTED] NaN/Inf Loss at step {state.global_step}: {logs['loss']}")
                    elif logs['loss'] > 100:
                        logger.warning(f"[ANOMALY DETECTED] High Loss at step {state.global_step}: {logs['loss']}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info("=" * 30)
            logger.info("评估结果:")
            for key, value in logs.items():
                if key.startswith('eval_'):
                    logger.info(f"{key}: {value:.6f}")
            logger.info("=" * 30)
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        
        logger.info("=" * 50)
        logger.info("训练结束")
        logger.info(f"总训练时间: {total_time:.2f}s")
        logger.info(f"平均步时间: {avg_step_time:.3f}s")
        logger.info(f"总步数: {len(self.step_times)}")
        logger.info("=" * 50)


def compute_metrics(eval_pred):
    """计算评估指标"""
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def preprocess_function(examples, tokenizer, max_length=512):
    """数据预处理函数"""
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding=False,  # 使用DataCollator进行动态padding
        max_length=max_length
    )


def load_and_prepare_data(config: TrainingConfig):
    """加载和准备IMDB数据集"""
    logger.info("加载IMDB数据集...")
    
    # 加载数据集
    dataset = load_dataset("imdb")
    
    # 为了快速测试，可以使用数据集的子集
    # 在实际实验中可以使用完整数据集
    train_dataset = dataset["train"].select(range(5000))  # 使用5000个训练样本
    eval_dataset = dataset["test"].select(range(1000))    # 使用1000个测试样本
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"测试集大小: {len(eval_dataset)}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 预处理数据
    logger.info("预处理数据...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True
    )
    
    return train_dataset, eval_dataset, tokenizer


def create_model(config: TrainingConfig):
    """创建模型"""
    logger.info(f"加载模型: {config.model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,  # IMDB是二分类任务
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    
    return model


def setup_trainer(model, train_dataset, eval_dataset, tokenizer, config: TrainingConfig):
    """设置Trainer"""
    
    # 创建训练参数
    training_args = TrainingArguments(**config.get_training_args_dict())
    
    # 数据整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 创建回调
    callbacks = [DetailedLoggingCallback()]
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    return trainer


def main(config_name: Optional[str] = None, config_path: Optional[str] = None):
    """主训练函数"""
    
    # 加载配置
    if config_path:
        config = TrainingConfig.from_yaml(config_path)
        logger.info(f"从文件加载配置: {config_path}")
    elif config_name:
        configs = FaultConfigFactory.get_all_fault_configs()
        if config_name not in configs:
            raise ValueError(f"未知配置名称: {config_name}. 可用配置: {list(configs.keys())}")
        config = configs[config_name]
        logger.info(f"使用预定义配置: {config_name}")
    else:
        config = FaultConfigFactory.create_baseline_config()
        logger.info("使用默认基线配置")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # 保存当前配置
    config_save_path = os.path.join(config.output_dir, "training_config.yaml")
    config.to_yaml(config_save_path)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 记录系统信息
    logger.info("=" * 60)
    logger.info("系统信息:")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        logger.info(f"当前CUDA设备: {torch.cuda.current_device()}")
        logger.info(f"GPU名称: {torch.cuda.get_device_name()}")
    logger.info("=" * 60)
    
    try:
        # 加载数据
        train_dataset, eval_dataset, tokenizer = load_and_prepare_data(config)
        
        # 创建模型
        model = create_model(config)
        
        # 设置训练器
        trainer = setup_trainer(model, train_dataset, eval_dataset, tokenizer, config)
        
        # 开始训练
        logger.info("开始训练...")
        start_time = time.time()
        
        trainer.train()
        
        end_time = time.time()
        logger.info(f"训练完成，总耗时: {end_time - start_time:.2f}秒")
        
        # 最终评估
        logger.info("进行最终评估...")
        eval_results = trainer.evaluate()
        logger.info("最终评估结果:")
        for key, value in eval_results.items():
            logger.info(f"{key}: {value}")
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_model()
        
        return eval_results
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        
        # 记录CUDA OOM错误
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.error("[ANOMALY DETECTED] CUDA Out of Memory Error!")
        
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BERT-IMDB训练脚本")
    parser.add_argument("--config-name", type=str, help="预定义配置名称")
    parser.add_argument("--config-path", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    main(config_name=args.config_name, config_path=args.config_path)