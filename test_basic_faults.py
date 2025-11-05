#!/usr/bin/env python3
"""
基础故障注入测试 - 不依赖transformers库
测试FaultInjector类的6种故障类型
"""

import sys
import os
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 直接导入故障注入器，避免__init__.py的依赖问题
import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "injector"))
from fault_injector import FaultInjector


def test_fault_type_coverage():
    """测试故障类型覆盖度"""
    print("=" * 60)
    print("📊 故障类型覆盖度测试")
    print("=" * 60)
    
    # 预期的6种故障类型
    expected_faults = {
        "nan_loss": "NaN Loss故障",
        "oom": "内存溢出故障", 
        "non_convergence": "模型不收敛故障",
        "io_stress": "I/O瓶颈故障",
        "resource_competition": "资源竞争故障",
        "process_kill": "进程终止故障"
    }
    
    # 实际实现的故障类型
    injector = FaultInjector()
    implemented_faults = []
    
    # 测试每种故障类型是否可以调度
    for fault_type in expected_faults.keys():
        try:
            injector.schedule_fault(fault_type, delay=0, duration=1)
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
    
    return implemented_faults


def test_fault_methods():
    """测试故障注入方法是否存在"""
    print("\n" + "=" * 60)
    print("🔧 故障注入方法测试")
    print("=" * 60)
    
    injector = FaultInjector()
    
    # 检查所有故障注入方法
    methods = [
        ("inject_io_stress", "I/O压力注入"),
        ("inject_resource_competition", "资源竞争注入"),
        ("inject_process_kill", "进程终止注入"),
        ("inject_nan_loss", "NaN Loss注入"),
        ("inject_oom", "内存溢出注入"),
        ("inject_non_convergence", "不收敛注入")
    ]
    
    implemented_methods = []
    
    for method_name, description in methods:
        if hasattr(injector, method_name):
            method = getattr(injector, method_name)
            if callable(method):
                implemented_methods.append(method_name)
                print(f"✅ {method_name}: {description}")
            else:
                print(f"❌ {method_name}: 不是可调用方法")
        else:
            print(f"❌ {method_name}: 方法不存在")
    
    method_coverage = len(implemented_methods) / len(methods) * 100
    print(f"\n📈 方法覆盖度: {method_coverage:.1f}% ({len(implemented_methods)}/{len(methods)})")
    
    return implemented_methods


def test_execute_fault_method():
    """测试execute_fault方法对新故障类型的支持"""
    print("\n" + "=" * 60)
    print("⚙️  execute_fault方法测试")
    print("=" * 60)
    
    injector = FaultInjector()
    
    # 测试所有故障类型的执行
    test_configs = [
        {
            'fault_type': 'nan_loss',
            'params': {'corruption_probability': 0.1}
        },
        {
            'fault_type': 'oom',
            'params': {'memory_size_mb': 10, 'allocation_pattern': 'linear'}
        },
        {
            'fault_type': 'non_convergence',
            'params': {'lr_multiplier': 10.0, 'corruption_type': 'too_high', 'duration': 1}
        }
    ]
    
    successful_executions = []
    
    for config in test_configs:
        fault_type = config['fault_type']
        try:
            print(f"🧪 测试执行 {fault_type}...")
            injector.execute_fault(config)
            successful_executions.append(fault_type)
            print(f"✅ {fault_type}: 执行成功")
        except Exception as e:
            print(f"❌ {fault_type}: 执行失败 - {e}")
    
    execution_rate = len(successful_executions) / len(test_configs) * 100
    print(f"\n📈 执行成功率: {execution_rate:.1f}% ({len(successful_executions)}/{len(test_configs)})")
    
    return successful_executions


def test_config_integration():
    """测试配置文件集成"""
    print("\n" + "=" * 60)
    print("⚙️  配置文件集成测试")
    print("=" * 60)
    
    try:
        # 检查配置文件是否存在
        config_file = Path("configs/fault_injection_config.yaml")
        if config_file.exists():
            print(f"✅ 配置文件存在: {config_file}")
            
            # 读取配置文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否包含6种故障类型
            fault_types = ["nan_loss", "oom", "non_convergence", "io_bottleneck", "resource_competition", "process_termination"]
            found_types = []
            
            for fault_type in fault_types:
                if fault_type in content:
                    found_types.append(fault_type)
                    print(f"✅ 配置包含: {fault_type}")
                else:
                    print(f"❌ 配置缺失: {fault_type}")
            
            config_coverage = len(found_types) / len(fault_types) * 100
            print(f"\n📈 配置覆盖度: {config_coverage:.1f}% ({len(found_types)}/{len(fault_types)})")
            
        else:
            print(f"❌ 配置文件不存在: {config_file}")
            
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")


def main():
    """主测试函数"""
    print("🧪 基础故障注入系统测试")
    print("测试FaultInjector类的6种故障类型支持")
    print("=" * 60)
    
    try:
        # 1. 测试故障类型覆盖度
        implemented_faults = test_fault_type_coverage()
        
        # 2. 测试故障注入方法
        implemented_methods = test_fault_methods()
        
        # 3. 测试execute_fault方法
        successful_executions = test_execute_fault_method()
        
        # 4. 测试配置文件集成
        test_config_integration()
        
        # 总结
        print("\n" + "=" * 60)
        print("📊 测试总结")
        print("=" * 60)
        
        print(f"✅ 支持的故障类型: {len(implemented_faults)}/6")
        print(f"✅ 实现的注入方法: {len(implemented_methods)}/6")
        print(f"✅ 成功执行的故障: {len(successful_executions)}/3")
        
        if len(implemented_faults) >= 6 and len(implemented_methods) >= 6:
            print("\n🎉 故障注入系统扩展成功！")
            print("✨ 现在支持完整的6种故障类型:")
            print("   1. NaN Loss故障 - 训练过程中的数值异常")
            print("   2. OOM故障 - 内存溢出异常")
            print("   3. 不收敛故障 - 学习率异常导致的收敛问题")
            print("   4. I/O瓶颈故障 - 磁盘读写压力")
            print("   5. 资源竞争故障 - CPU/GPU/内存资源争抢")
            print("   6. 进程终止故障 - 异常进程中断")
        else:
            print("\n⚠️  故障注入系统扩展不完整")
            print("需要进一步检查和修复")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()