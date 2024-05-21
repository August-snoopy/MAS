import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class IMUDataset(Dataset):
    """IMU数据集"""

    def __init__(self, data: pd.DataFrame, seq_length: int, target_cols: list):
        """
        初始化IMUDataset

        :param data: 包含IMU数据的DataFrame
        :param seq_length: 输入序列的长度
        :param target_cols: 目标列的名称列表
        """
        self.data = data
        self.seq_length = seq_length
        self.target_cols = target_cols

    def __len__(self) -> int:
        """返回数据集的长度"""
        return len(self.data) - self.seq_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回给定索引的输入序列和目标值

        :param index: 数据索引
        :return: 输入序列和目标值的元组
        """
        input_seq = torch.tensor(self.data.iloc[index:index + self.seq_length].values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[index + self.seq_length][self.target_cols].values, dtype=torch.float32)
        return input_seq, target


class LSTMModel(nn.Module):
    """LSTM模型"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        初始化LSTMModel

        :param input_size: 输入特征的维度
        :param hidden_size: 隐藏状态的维度
        :param num_layers: LSTM层的数量
        :param output_size: 输出的维度
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        :param x: 输入序列,形状为(batch_size, seq_length, input_size)
        :return: 预测输出,形状为(batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
          num_epochs: int, device: str):
    """
    训练模型

    :param model: LSTM模型
    :param dataloader: 数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: 训练的轮数
    :param device: 使用的设备 ('cpu' 或 'cuda')
    """
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def predict(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> np.ndarray:
    """
    使用模型进行预测

    :param model: LSTM模型
    :param dataloader: 数据加载器
    :param device: 使用的设备 ('cpu' 或 'cuda')
    :return: 预测结果的数组
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions)


# 示例用法
if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('imu_data.csv')

    # 定义超参数
    input_size = len(data.columns) - 1  # 减去标签列
    hidden_size = 64
    num_layers = 2
    output_size = 1  # 假设只预测一个标签
    seq_length = 50
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 创建数据集和数据加载器
    target_cols = ['label']  # 假设标签列为 'label'
    dataset = IMUDataset(data, seq_length, target_cols)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 创建模型、损失函数和优化器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train(model, dataloader, criterion, optimizer, num_epochs, device)

    # 使用模型进行预测
    predictions = predict(model, dataloader, device)
    print("Predictions:", predictions)
