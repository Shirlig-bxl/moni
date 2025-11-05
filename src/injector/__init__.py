"""
注入器模块
负责主动触发各种故障，包括I/O压力、资源竞争、进程终止等
"""

from .fault_injector import FaultInjector, create_fault_schedule_from_config
from .io_stress import IOStressTester
from .resource_competitor import ResourceCompetitor

__all__ = [
    "FaultInjector", 
    "create_fault_schedule_from_config",
    "IOStressTester", 
    "ResourceCompetitor"
]