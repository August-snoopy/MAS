import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 定义数据集类
class IMUDataset(Dataset):
    def __init__(self, data_path):
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip().split(',')
                features = [float(x) for x in line[:-1]]
                label = line[-1].strip('"')
                data.append((features, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 定义模型类
class IMUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IMUModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for features, labels in dataloader:
        features = features.float().to(device)
        labels = labels.long().to(device)
        hidden = model.init_hidden(features.size(0))

        optimizer.zero_grad()
        output, _ = model(features, hidden)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()


# 预测函数
def predict(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.float().to(device)
            hidden = model.init_hidden(features.size(0))
            output, _ = model(features, hidden)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions


# 主函数
def main():
    # 数据集路径
    train_data_path = 'train_data.csv'
    test_data_path = 'test_data.csv'

    # 超参数
    input_size = 288  # 24个节点*12个特征
    hidden_size = 128
    output_size = 10  # 假设有10种不同的动作标签
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # 加载数据集
    train_dataset = IMUDataset(train_data_path)
    test_dataset = IMUDataset(test_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IMUModel(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        train(model, train_dataloader, criterion, optimizer, device)

    # 预测结果
    predictions = predict(model, test_dataloader, device)
    print(predictions)


if __name__ == '__main__':
    main()