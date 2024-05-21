import pandas as pd
import torch
import numpy as np
import os
import pprint


nodes = {
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
# node = 'Hips'
# 设置根目录
root = os.path.join(os.path.dirname(__file__), '../../')
os.chdir(root)

# 读取数据进行预处理
data_path = os.path.join('data', '20240508')

# 读取data_path 下的所有csv文件
files = os.listdir(data_path)

df = pd.read_csv(os.path.join(data_path, 'yishi01.csv'))

# rotation_matrix = df[[f"{node}_position_{i}{j}" for i in range(1,4) for j in range(1,4)]][0:2]
# # .values.reshape(3, 3)
# print(rotation_matrix.values)
data = []
for i in range(3):
    frame_data = {}
    for node in nodes.keys():
        # 提取旋转矩阵
        rotation_matrix = df[[f"{node}_position_{i}{j}" for i in range(1, 4) for j in range(1, 4)]].iloc[
            i].values.reshape(3, 3)
        # 提取加速度向量
        acceleration_vector = df[[f"{node}_acceleration_{i}" for i in range(1, 4)]].iloc[i].values
        frame_data[node] = (rotation_matrix, acceleration_vector)
    data.append(frame_data)

pprint.pprint(data)
