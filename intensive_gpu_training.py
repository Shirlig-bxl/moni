
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from datetime import datetime

def log_with_timestamp(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{timestamp}] {message}')

def create_large_model():
    model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def main():
    log_with_timestamp('开始GPU密集型训练')
    
    if not torch.cuda.is_available():
        log_with_timestamp('ERROR: CUDA不可用，无法继续')
        return
    
    device = torch.device('cuda')
    log_with_timestamp(f'使用设备: {device}')
    log_with_timestamp(f'GPU名称: {torch.cuda.get_device_name(0)}')
    
    model = create_large_model().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log_with_timestamp(f'模型参数总数: {total_params:,}')
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 256
    input_size = 2048
    num_batches = 100
    
    for epoch in range(5):
        log_with_timestamp(f'Epoch {epoch+1}/5')
        epoch_start_time = time.time()
        total_loss = 0
        
        for batch_idx in range(num_batches):
            inputs = torch.randn(batch_size, input_size, device=device)
            targets = torch.randint(0, 10, (batch_size,), device=device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                log_with_timestamp(f'  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, GPU内存: {gpu_memory:.1f}MB')
            
            if batch_idx % 5 == 0:
                extra_computation = torch.matmul(
                    torch.randn(512, 512, device=device),
                    torch.randn(512, 512, device=device)
                )
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        log_with_timestamp(f'Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}秒')

if __name__ == '__main__':
    main()
