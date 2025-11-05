"""
注入器模块
负责主动触发各种故障，包括I/O压力、资源竞争、进程终止、训练级故障等
"""

from .fault_injector import FaultInjector, create_fault_schedule_from_config
from .io_stress import IOStressTester
from .resource_competitor import ResourceCompetitor

# 尝试导入训练钩子，如果失败则跳过（避免transformers依赖问题）
try:
    from .training_hook import (
        TrainingFaultInjector, 
        NaNLossHook, 
        OOMHook, 
        NonConvergenceHook,
        create_fault_hooks_from_config
    )
    _training_hooks_available = True
except ImportError as e:
    print(f"Warning: Training hooks not available due to missing dependencies: {e}")
    _training_hooks_available = False

__all__ = [
    "FaultInjector", 
    "create_fault_schedule_from_config",
    "IOStressTester", 
    "ResourceCompetitor"
]

# 只有在训练钩子可用时才添加到__all__
if _training_hooks_available:
    __all__.extend([
        "TrainingFaultInjector",
        "NaNLossHook",
        "OOMHook", 
        "NonConvergenceHook",
        "create_fault_hooks_from_config"
    ])