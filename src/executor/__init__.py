"""
执行器模块
负责运行ML训练任务，支持故障注入
"""

from .config import TrainingConfig, FaultConfigFactory
from .train import main as train_main

__all__ = ["TrainingConfig", "FaultConfigFactory", "train_main"]