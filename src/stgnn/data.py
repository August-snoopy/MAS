import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 设置根目录
root = os.path.join(os.path.dirname(__file__), '../../')
os.chdir(root)

# 读取数据进行预处理
data_path = os.path.join('data', '20240508')

# 读取data_path 下的所有csv文件
files = os.listdir(data_path)

df = pd.read_csv(os.path.join(data_path, 'yishi01.csv'))



# 构建节点及其父节点关系
# 注意一共21个节点，Root为虚拟节点，不参与计算
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

# 创建一个空的无向图
G = nx.Graph()

# 添加节点和边到图中
for child, parent in nodes.items():
    G.add_edge(parent, child)

# 定义节点初始位置，使得图像更像一个站立的人
initial_pos = {
    # "Root": np.array([0, 0, 0]),
    "Hips": np.array([0, 0.1, 0]),
    "RightUpLeg": np.array([-0.1, 0.1, 0]),
    "RightLeg": np.array([-0.1, 0, 0]),
    "RightFoot": np.array([-0.1, -0.1, 0]),
    "LeftUpLeg": np.array([0.1, 0.1, 0]),
    "LeftLeg": np.array([0.1, 0, 0]),
    "LeftFoot": np.array([0.1, -0.1, 0]),
    "Spine": np.array([0, 0.2, 0]),
    "Spine1": np.array([0, 0.3, 0]),
    "Spine2": np.array([0, 0.4, 0]),
    "Neck": np.array([0, 0.5, 0]),
    "Neck1": np.array([0, 0.6, 0]),
    "Head": np.array([0, 0.7, 0]),
    "RightShoulder": np.array([-0.1, 0.4, 0]),
    "RightArm": np.array([-0.2, 0.4, 0]),
    "RightForeArm": np.array([-0.3, 0.4, 0]),
    "RightHand": np.array([-0.4, 0.4, 0]),
    "LeftShoulder": np.array([0.1, 0.4, 0]),
    "LeftArm": np.array([0.2, 0.4, 0]),
    "LeftForeArm": np.array([0.3, 0.4, 0]),
    "LeftHand": np.array([0.4, 0.4, 0])
}
# 初始化速度
vel = {node: np.zeros(3) for node in initial_pos}
# 假设每帧的数据格式为 {node: (rotation_matrix, acceleration_vector)}
# 示例数据
data_example = [
    {
        "Hips": np.zeros(9),
        "RightUpLeg": np.zeros(9),
        # 其他节点数据
    },
    # 更多帧数据
]

def extract_data_from_csv(file_path):
    """
    从csv中提取节点数据，已知csv的格式为：每一行为一个时间步，每一列为一个特征。
    特征包括每个人体节点的旋转位置矩阵和加速度向量，但是展平拼接了起来
    而特征名实际就是上述的节点名+特征名，例如：Hips_position_11, Hips_acceleration_1
    """
    df = pd.read_csv(file_path)
    data = []
    # 提取每一个时间步
    for i in range(len(df)):
        frame_data = {}
        for node in nodes.keys():
            # 提取旋转矩阵
            rotation_matrix = df[[f"{node}_position_{i}{j}" for i in range(1,4) for j in range(1,4)]].iloc[i].values
            # 提取加速度向量
            acceleration_vector = df[[f"{node}_acceleration_{i}" for i in range(1,4)]].iloc[i].values
            # 将展平的旋转矩阵和加速度向量相拼接
            frame_data[node] = np.concatenate([rotation_matrix, acceleration_vector])
        data.append(frame_data)
    return data

data = extract_data_from_csv(os.path.join(data_path, 'yishi01.csv'))


# 更新节点位置的函数
def update_positions(frame_data, pos, vel, dt=0.01):
    new_pos = pos.copy()
    new_vel = vel.copy()
    for node, (rot_matrix, accel_vector) in frame_data.items():
        # 应用加速度更新速度
        new_vel[node] += np.array(accel_vector, dtype=float) * dt
        # 使用速度更新位置
        new_pos[node] += new_vel[node][:2] * dt
    return new_pos, new_vel

# 初始化图像
fig, ax = plt.subplots(figsize=(10, 8))
pos = initial_pos.copy()
pos = {node: pos[node][:2] for node in pos}
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")

# 更新函数
def update(frame):
    global pos, vel
    ax.clear()
    pos, vel = update_positions(data[frame], pos, vel)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray")
    ax.set_title(f"Human Body Parts Topology - Frame {frame}")

# 创建动画
ani = FuncAnimation(fig, update, frames=len(data), repeat=True)
plt.show()

# 保存动画
ani.save('human_body_parts_topology.gif', writer='imagemagick', fps=10)