"""
训练配置管理模块
支持正常训练和各种故障注入模式的配置
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    # 模型配置
    model_name: str = "bert-base-uncased"
    task_name: str = "imdb"
    
    # 训练超参数
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # 数据配置
    max_seq_length: int = 512
    train_split: str = "train"
    eval_split: str = "test"
    
    # 输出配置
    output_dir: str = "./results"
    logging_dir: str = "./logs"
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    
    # 故障注入配置
    fault_type: Optional[str] = None  # "nan_loss", "oom", "no_convergence", None
    fault_params: Dict[str, Any] = field(default_factory=dict)
    
    # 监控配置
    enable_monitoring: bool = True
    monitor_interval: int = 1  # 秒
    
    def __post_init__(self):
        """配置后处理，应用故障注入参数"""
        if self.fault_type:
            self._apply_fault_config()
    
    def _apply_fault_config(self):
        """根据故障类型应用相应的配置修改"""
        if self.fault_type == "nan_loss":
            # 梯度爆炸：设置极大学习率
            self.learning_rate = self.fault_params.get("learning_rate", 10.0)
            print(f"[FAULT INJECTION] NaN Loss: learning_rate = {self.learning_rate}")
            
        elif self.fault_type == "oom":
            # 显存溢出：设置极大批次大小
            self.per_device_train_batch_size = self.fault_params.get("batch_size", 1024)
            print(f"[FAULT INJECTION] OOM: batch_size = {self.per_device_train_batch_size}")
            
        elif self.fault_type == "no_convergence":
            # 不收敛：设置不合适的学习率
            convergence_type = self.fault_params.get("type", "too_low")
            if convergence_type == "too_low":
                self.learning_rate = self.fault_params.get("learning_rate", 1e-9)
            elif convergence_type == "too_high":
                self.learning_rate = self.fault_params.get("learning_rate", 1e-2)
            print(f"[FAULT INJECTION] No Convergence ({convergence_type}): learning_rate = {self.learning_rate}")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """保存配置到YAML文件"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)
    
    def get_training_args_dict(self) -> Dict[str, Any]:
        """获取Transformers TrainingArguments所需的参数字典"""
        return {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "num_train_epochs": self.num_train_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "logging_dir": self.logging_dir,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_accuracy",
            "greater_is_better": True,
            "report_to": [],  # 禁用wandb等外部报告
        }


class FaultConfigFactory:
    """故障配置工厂类"""
    
    @staticmethod
    def create_baseline_config() -> TrainingConfig:
        """创建基线（正常）配置"""
        return TrainingConfig(
            fault_type=None,
            output_dir="./data/raw/baseline",
            logging_dir="./data/raw/baseline/logs"
        )
    
    @staticmethod
    def create_nan_loss_config() -> TrainingConfig:
        """创建NaN Loss故障配置"""
        return TrainingConfig(
            fault_type="nan_loss",
            fault_params={"learning_rate": 10.0},
            output_dir="./data/raw/nan_loss",
            logging_dir="./data/raw/nan_loss/logs"
        )
    
    @staticmethod
    def create_oom_config() -> TrainingConfig:
        """创建OOM故障配置"""
        return TrainingConfig(
            fault_type="oom",
            fault_params={"batch_size": 1024},
            output_dir="./data/raw/oom",
            logging_dir="./data/raw/oom/logs"
        )
    
    @staticmethod
    def create_no_convergence_config(convergence_type: str = "too_low") -> TrainingConfig:
        """创建不收敛故障配置"""
        lr_map = {
            "too_low": 1e-9,
            "too_high": 1e-2
        }
        return TrainingConfig(
            fault_type="no_convergence",
            fault_params={
                "type": convergence_type,
                "learning_rate": lr_map[convergence_type]
            },
            output_dir=f"./data/raw/no_convergence_{convergence_type}",
            logging_dir=f"./data/raw/no_convergence_{convergence_type}/logs",
            num_train_epochs=5  # 增加训练轮数以观察不收敛现象
        )
    
    @staticmethod
    def get_all_fault_configs() -> Dict[str, TrainingConfig]:
        """获取所有故障配置"""
        return {
            "baseline": FaultConfigFactory.create_baseline_config(),
            "nan_loss": FaultConfigFactory.create_nan_loss_config(),
            "oom": FaultConfigFactory.create_oom_config(),
            "no_convergence_low": FaultConfigFactory.create_no_convergence_config("too_low"),
            "no_convergence_high": FaultConfigFactory.create_no_convergence_config("too_high"),
        }


if __name__ == "__main__":
    # 测试配置创建和保存
    configs = FaultConfigFactory.get_all_fault_configs()
    
    for name, config in configs.items():
        config_path = f"./configs/training_configs/{name}.yaml"
        config.to_yaml(config_path)
        print(f"保存配置: {config_path}")
    
    print("所有配置文件已生成完成！")