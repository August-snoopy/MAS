import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

class IMUDataset(torch.utils.data.Dataset):
    """IMU数据集"""

    def __init__(self, data: Tuple[torch.Tensor, int], seq_length: int):
        """
        初始化IMUDataset

        :param data: IMU数据特征矩阵＋对应类别
        :param seq_length: 输入序列的长度
        """
        self.data = data
        self.seq_length = seq_length


    def __len__(self) -> int:
        """返回数据集的长度"""
        return self.seq_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回给定索引的输入序列和目标值

        :param index: 数据索引
        :return: 输入序列和目标值的元组
        """
        input_seq = torch.tensor(self.data[index][0], dtype=torch.float32)
        target = torch.tensor(self.data[index][1], dtype=torch.float32)
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
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
        for i, (inputs, targets) in enumerate(dataloader):
            targets = torch.stack(targets)  # 将targets中的Tensor合并成一个Tensor
            print('inputs is :', inputs)
            print('targets is :', targets)
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

def collate_fn(train_data):
    # print('train_data is :', train_data)
    # 检查 train_data 中的元素是否为Tensor，如果不是的话就将其转换为Tensor
     # 分离数据和标签
    data = [item[0] for item in train_data]
    labels = [item[1] for item in train_data]

    # 对数据进行排序和填充
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(x) for x in data]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    return data, labels


# 示例用法
if __name__ == '__main__':
    # 读取数据
    data = pd.read_csv('data/20240508/yishi01.csv')
    # 删除特定列中有NaN值的行
    data = data.dropna(subset=['label'])
    # 创建一个 LabelEncoder 对象
    le = LabelEncoder()
    # 使用 LabelEncoder 对标签进行编码
    data['label'] = le.fit_transform(data['label']) 

    # 定义超参数
    input_size = len(data.columns) - 1  # 减去标签列
    # input_size = 2
    hidden_size = 64
    num_layers = 2
    output_size = 1  # 假设只预测一个标签

    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001


    # 初始化一个列表来存储特征矩阵和类别名的元组
    data_list = []
    labels = set()
    # 初始化一个变量来存储当前的特征矩阵和类别名
    current_matrix = []
    # 初始化一个变量来存储当前的标签
    current_label = data['label'].iloc[0]

    # 遍历数据
    for i in range(len(data)):
        if data['label'].iloc[i] not in labels:
            labels.add(data['label'].iloc[i])
        # 如果当前的类别名与上一个不同，那么将当前的特征矩阵和类别名添加到列表中，并开始一个新的特征矩阵
        if data['label'].iloc[i] != current_label:
            if current_matrix:
                data_list.append((torch.tensor(current_matrix), current_label))
            current_matrix = []
            current_label = data['label'].iloc[i]
        # 将当前的行添加到特征矩阵中
        current_matrix.append(data.iloc[i, :-1].tolist())

    # 添加最后一个特征矩阵和类别名
    if current_matrix:
        data_list.append((torch.tensor(current_matrix), current_label))

    seq_length = len(data_list)
    # output_size = len(labels)
    
    # 创建数据集和数据加载器
    dataset = IMUDataset(data_list, seq_length)
    # a,b = dataset[0]
    # print('dataset is :', a)
    # print('label is :', b)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    # for data, length in dataloader:
    #     data = torch.utils.rnn.pack_padded_sequence(data, length, batch_first=True)
    # 使用enumerate访问可遍历的数组对象
    # for step, (input, target) in enumerate(dataloader):
    #     print('step is :', step)
    #     # data, label = input, target
    #     print('data is {}, label is {}'.format(input, target))
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


# class IMUDataset(torch.utils.data.Dataset):
#     """IMU数据集"""

#     def __init__(self, data: pd.DataFrame, seq_lengths: dict, seq_starts: dict, target_cols: list):
#         """
#         初始化IMUDataset

#         :param data: 包含IMU数据的DataFrame
#         :param seq_length: 输入序列的长度
#         :param target_cols: 目标列的名称列表
#         """
#         self.data = data
#         self.seq_lengths = seq_lengths
#         self.seq_starts = seq_starts
#         self.target_cols = target_cols

#     def __len__(self) -> int:
#         """返回数据集的长度"""
#         return sum([len(v) for v in self.seq_lengths.values()])

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         返回给定索引的输入序列和目标值

#         :param index: 数据索引
#         :return: 输入序列和目标值的元组
#         """
#         # 找到对应的标签和序列
#         for label in self.seq_lengths:
#             if index < len(self.seq_lengths[label]):
#                 seq_length = self.seq_lengths[label][index]
#                 start = self.seq_starts[label][index]
#                 break
#             else:
#                 index -= len(self.seq_lengths[label])

#         input_seq = torch.tensor(self.data.iloc[index:index + self.seq_length].values, dtype=torch.float32)
#         target = torch.tensor(self.data.iloc[index + self.seq_length][self.target_cols].values, dtype=torch.float32)
#         return input_seq, target

# 废弃方法：

    # # 初始化一个字典来存储每个标签的序列长度，另一个字典来存储每个标签的序列开始位置
    # seq_lengths = {}
    # seq_starts = {}
    # # 初始化一个变量来存储当前序列的长度
    # current_seq_length = 0

    # # 初始化一个变量来存储当前的标签
    # current_label = data['label'].iloc[0]
    # target_cols.append(current_label)
    # # print("Current label:", current_label)
    # # 初始化一个变量来存储当前序列的开始位置
    # seq_starts[current_label] = [0]
    # for i in range(len(data)):
    #     # 如果当前的标签不在seq_lengths字典中，将当前标签添加到字典中
    #     if current_label not in seq_lengths:
    #         seq_lengths[current_label] = []
    #     # 如果当前的标签不在seq_starts字典中，将当前位置添加到字典中 
    #     if data['label'].iloc[i] not in seq_starts:
    #         # seq_starts[data['label'].iloc[i]] = [seq_starts[data['label'].iloc[i]] + seq_lengths[data['label'].iloc[i]]]
    #         seq_starts[data['label'].iloc[i]] = [i]
    #     # 如果当前的标签与上一个标签相同，那么当前序列的长度加1
    #     if data['label'].iloc[i] == current_label:
    #         current_seq_length += 1
    #     # 如果当前的标签与上一个标签不同
    #     else:
    #         # 将当前序列的长度添加到对应标签的列表中，并重置当前序列的长度和当前的标签
    #         seq_lengths[current_label].append(current_seq_length)
    #         current_seq_length = 1
    #         current_label = data['label'].iloc[i]
    #         if current_label not in target_cols:
    #             target_cols.append(current_label)
    #         if i not in seq_starts[current_label]:
    #             seq_starts[current_label].append(i)
    #         # # 如果当前标签不在seq_starts字典中，将当前位置添加到字典中
    #         # if current_label not in seq_starts:
    #         #     seq_starts[current_label] = i

    # # 添加最后一个序列的长度
    # if current_label not in seq_lengths:
    #     seq_lengths[current_label] = []
    # seq_lengths[current_label].append(current_seq_length)
    # print("Target columns:", target_cols)
    # print("Sequence lengths:", seq_lengths)
    # print("Sequence starts:", seq_starts)