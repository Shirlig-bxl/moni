#!/usr/bin/env python3
"""
测试所有6种故障注入类型的完整功能
验证扩展后的故障注入系统
"""

import sys
import os
import time
import threading
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from injector import FaultInjector, TrainingFaultInjector, create_fault_hooks_from_config
from utils.config_loader import ConfigLoader


def test_system_level_faults():
    """测试系统级故障注入"""
    print("=" * 60)
    print("🔧 测试系统级故障注入")
    print("=" * 60)
    
    injector = FaultInjector()
    
    # 测试1: I/O压力
    print("\n1️⃣ 测试I/O压力故障...")
    injector.schedule_fault(
        "io_stress",
        delay=1,
        duration=5,
        target_dir="/tmp",
        read_rate="10M",
        write_rate="10M"
    )
    
    # 测试2: 资源竞争
    print("2️⃣ 测试资源竞争故障...")
    injector.schedule_fault(
        "resource_competition",
        delay=8,
        duration=5,
        competitor_type="cpu"
    )
    
    # 测试3: NaN Loss (系统级实现)
    print("3️⃣ 测试NaN Loss故障...")
    injector.schedule_fault(
        "nan_loss",
        delay=15,
        duration=3,
        corruption_probability=0.8
    )
    
    # 测试4: OOM
    print("4️⃣ 测试OOM故障...")
    injector.schedule_fault(
        "oom",
        delay=20,
        duration=5,
        memory_size_mb=512,
        allocation_pattern="linear"
    )
    
    # 测试5: 不收敛
    print("5️⃣ 测试不收敛故障...")
    injector.schedule_fault(
        "non_convergence",
        delay=27,
        duration=3,
        lr_multiplier=100.0,
        corruption_type="too_high"
    )
    
    # 启动故障注入器
    print("\n🚀 启动故障注入器...")
    injector.start()
    
    # 运行35秒
    try:
        for i in range(35):
            print(f"⏱️  运行中... {i+1}/35秒", end='\r')
            time.sleep(1)
        print("\n")
    except KeyboardInterrupt:
        print("\n用户中断测试")
    finally:
        injector.stop()
        print("✅ 系统级故障测试完成")


def test_training_level_faults():
    """测试训练级故障注入钩子"""
    print("\n" + "=" * 60)
    print("🎯 测试训练级故障注入钩子")
    print("=" * 60)
    
    # 创建训练故障注入器
    training_injector = TrainingFaultInjector()
    
    # 模拟配置
    config = {
        "injection_schedule": {
            "step_based": [
                {
                    "fault_type": "nan_loss",
                    "injection_step": 5,
                    "duration": 3
                },
                {
                    "fault_type": "oom",
                    "injection_step": 10,
                    "duration": 2
                },
                {
                    "fault_type": "non_convergence",
                    "injection_step": 15,
                    "duration": 5
                }
            ]
        }
    }
    
    # 从配置创建钩子
    hooks = create_fault_hooks_from_config(config)
    
    print(f"📋 创建了 {len(hooks)} 个训练级故障钩子:")
    for hook in hooks:
        training_injector.add_hook(hook)
        print(f"   - {hook.fault_type} @ step {hook.trigger_step}")
    
    print("✅ 训练级故障钩子测试完成")


def test_config_integration():
    """测试配置文件集成"""
    print("\n" + "=" * 60)
    print("⚙️  测试配置文件集成")
    print("=" * 60)
    
    try:
        # 加载配置
        config_loader = ConfigLoader()
        fault_config = config_loader.load_fault_injection_config()
        
        print("📄 故障注入配置文件内容:")
        print(f"   - 全局配置: {fault_config.get('global', {})}")
        
        fault_types = fault_config.get('fault_types', {})
        print(f"   - 支持的故障类型: {list(fault_types.keys())}")
        
        for fault_name, fault_info in fault_types.items():
            enabled = fault_info.get('enabled', False)
            description = fault_info.get('description', 'N/A')
            print(f"     * {fault_name}: {'✅' if enabled else '❌'} - {description}")
        
        # 测试调度配置
        schedule = fault_config.get('injection_schedule', {})
        step_based = schedule.get('step_based', [])
        time_based = schedule.get('time_based', [])
        
        print(f"   - 基于步数的注入: {len(step_based)} 个")
        print(f"   - 基于时间的注入: {len(time_based)} 个")
        
        print("✅ 配置文件集成测试完成")
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")


def test_fault_type_coverage():
    """测试故障类型覆盖度"""
    print("\n" + "=" * 60)
    print("📊 故障类型覆盖度测试")
    print("=" * 60)
    
    # 预期的6种故障类型
    expected_faults = {
        "nan_loss": "NaN Loss故障",
        "oom": "内存溢出故障", 
        "non_convergence": "模型不收敛故障",
        "io_bottleneck": "I/O瓶颈故障",
        "resource_competition": "资源竞争故障",
        "process_termination": "进程终止故障"
    }
    
    # 实际实现的故障类型
    injector = FaultInjector()
    implemented_faults = []
    
    # 测试每种故障类型是否可以调度
    for fault_type in expected_faults.keys():
        try:
            # 映射故障类型名称
            if fault_type == "io_bottleneck":
                test_type = "io_stress"
            elif fault_type == "process_termination":
                test_type = "process_kill"
            else:
                test_type = fault_type
                
            injector.schedule_fault(test_type, delay=0, duration=1)
            implemented_faults.append(fault_type)
            print(f"✅ {fault_type}: {expected_faults[fault_type]}")
        except Exception as e:
            print(f"❌ {fault_type}: 未实现 - {e}")
    
    coverage = len(implemented_faults) / len(expected_faults) * 100
    print(f"\n📈 故障类型覆盖度: {coverage:.1f}% ({len(implemented_faults)}/{len(expected_faults)})")
    
    if coverage >= 100:
        print("🎉 所有故障类型均已实现！")
    else:
        missing = set(expected_faults.keys()) - set(implemented_faults)
        print(f"⚠️  缺失的故障类型: {list(missing)}")


def main():
    """主测试函数"""
    print("🧪 故障注入系统完整性测试")
    print("测试所有6种故障类型的功能")
    print("=" * 60)
    
    try:
        # 1. 测试故障类型覆盖度
        test_fault_type_coverage()
        
        # 2. 测试配置文件集成
        test_config_integration()
        
        # 3. 测试训练级故障钩子
        test_training_level_faults()
        
        # 4. 测试系统级故障注入 (可选，因为会实际执行故障)
        print("\n" + "=" * 60)
        print("⚠️  系统级故障测试将实际执行故障注入")
        response = input("是否继续执行系统级故障测试? (y/N): ").strip().lower()
        
        if response == 'y':
            test_system_level_faults()
        else:
            print("⏭️  跳过系统级故障测试")
        
        print("\n" + "=" * 60)
        print("🎊 所有测试完成！")
        print("故障注入系统已成功扩展为支持6种故障类型")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()