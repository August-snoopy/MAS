import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class HumanBodyGNN(torch.nn.Module):
    def __init__(self):
        super(HumanBodyGNN, self).__init__()
        self.conv1 = GCNConv(12, 64)  # 输入维度12（9个旋转矩阵 + 3个加速度向量）
        self.conv2 = GATConv(64, 128, heads=4, dropout=0.2)
        self.conv3 = GCNConv(512, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)  # 全局池化
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
