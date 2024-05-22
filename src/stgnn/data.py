import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, InMemoryDataset


class HumanBodyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.le = LabelEncoder()
        super(HumanBodyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # 构建节点及其父节点关系
        # 注意一共21个节点，Root为虚拟节点，不参与计算
        self.nodes = {
            "Hips": "Root",
            "RightUpLeg": "Hips",
            "RightLeg": "RightUpLeg",
            "RightFoot": "RightLeg",
            "LeftUpLeg": "Hips",
            "LeftLeg": "LeftUpLeg",
            "LeftFoot": "LeftLeg",
            "Spine": "Hips",
            "Spine1": "Spine",
            "Spine2": "Spine1",
            "Neck": "Spine2",
            "Neck1": "Neck",
            "Head": "Neck1",
            "RightShoulder": "Spine2",
            "RightArm": "RightShoulder",
            "RightForeArm": "RightArm",
            "RightHand": "RightForeArm",
            "LeftShoulder": "Spine2",
            "LeftArm": "LeftShoulder",
            "LeftForeArm": "LeftArm",
            "LeftHand": "LeftForeArm"
        }

    @property
    def raw_file_names(self):
        # 读取data_path 下的所有csv文件
        files = os.listdir(self.root)
        return list(files)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 数据集下载的逻辑
        pass

    def process(self):
        data_list = []

        # 读取CSV文件并处理
        for raw_path in self.raw_paths:
            df = pd.read_csv(raw_path)
            num_nodes = len(self.nodes)

            # 提取每一时间步的数据
            for i in range(len(df)):
                x = []
                for node in self.nodes.keys():
                    # 提取旋转矩阵
                    rotation_matrix = df[[f"{node}_position_{i}{j}" for i in range(1, 4) for j in range(1, 4)]].iloc[
                        i].values
                    # 提取加速度向量
                    acceleration_vector = df[[f"{node}_acceleration_{i}" for i in range(1, 4)]].iloc[i].values
                    # 将展平的旋转矩阵和加速度向量相拼接
                    x.append(np.concatenate([rotation_matrix, acceleration_vector]))

                # 标签
                y = df.iloc[i, -1]  # 假设最后一列为标签
                y = self.le.fit_transform([y])[0]  # 标签编码

                # 转换为PyG的数据格式
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor([y], dtype=torch.float)
                edge_index = self.get_edge_index()

                datas = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(datas)

        datas, slices = self.collate(data_list)
        torch.save((datas, slices), self.processed_paths[0])

    def get_edge_index(self):
        edge_index = []
        for child, parent in self.nodes.items():
            if parent != "Root":
                edge_index.append([list(self.nodes.keys()).index(parent), list(self.nodes.keys()).index(child)])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '../../'))
    # 读取数据进行预处理
    data_path = os.path.join('data', '20240508')
    # 使用数据集
    dataset = HumanBodyDataset(root=data_path)
    # 获取第一个数据
    data = dataset[0]
    print(data)
