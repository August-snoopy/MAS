import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 解析XML文件并长度单位从厘米为毫米
xml_data = '''<?xml version="1.0" encoding="UTF-8"?>
<body name="yishi01">
    <desc name="Body" length="580"/> <!-- 单位为毫米 -->
    <desc name="Foot length" length="262.4"/> <!-- 单位为毫米 -->
    <desc name="Forearm" length="290"/> <!-- 单位为毫米 -->
    <desc name="Head" length="165"/> <!-- 单位为毫米 -->
    <desc name="Heel height" length="80.8"/> <!-- 单位为毫米 -->
    <desc name="Hip width" length="220"/> <!-- 单位为毫米 -->
    <desc name="Lower leg" length="440"/> <!-- 单位为毫米 -->
    <desc name="Neck" length="160"/> <!-- 单位为毫米 -->
    <desc name="Palm" length="180"/> <!-- 单位为毫米 -->
    <desc name="Shoulder width" length="370"/> <!-- 单位为毫米 -->
    <desc name="Upper arm" length="310"/> <!-- 单位为毫米 -->
    <desc name="Upper leg" length="500"/> <!    -- 单位为毫米 -->
</body>
'''

root = ET.fromstring(xml_data)
body_parts = {desc.attrib['name']: float(desc.attrib['length']) for desc in root.findall('desc')}

# 读取CSV文件
csv_file_path = './test.csv'
df = pd.read_csv(csv_file_path)

# 定义关节点的继承关系
NODES = {
    "Hips": "Root",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    # "Spine": "Hips",
    # "Spine1": "Spine",
    "Spine2": "Hips",
    # "Neck": "Spine2",
    # "Neck1": "Neck",
    "Head": "Spine2",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm"
}

# 初始化骨骼的关节点位置（单位为毫米）
initial_positions = {
    "Hips": np.array([0, 0, 0]),
    "RightUpLeg": np.array([0, -body_parts["Hip width"] / 2, 0]),
    "RightLeg": np.array([0, -body_parts["Upper leg"], 0]),
    "RightFoot": np.array([0, -body_parts["Lower leg"], 0]),
    "LeftUpLeg": np.array([0, body_parts["Hip width"] / 2, 0]),
    "LeftLeg": np.array([0, -body_parts["Upper leg"], 0]),
    "LeftFoot": np.array([0, -body_parts["Lower leg"], 0]),
    "Spine": np.array([0, 0, body_parts["Body"] / 3]),
    "Spine1": np.array([0, 0, body_parts["Body"] / 3]),
    "Spine2": np.array([0, 0, body_parts["Body"] / 3]),
    "Neck": np.array([0, 0, body_parts["Neck"]]),
    "Neck1": np.array([0, 0, body_parts["Neck"]]),
    "Head": np.array([0, 0, body_parts["Head"]]),
    "RightShoulder": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"] / 3]),
    "RightArm": np.array([0, -body_parts["Upper arm"], 0]),
    "RightForeArm": np.array([0, -body_parts["Forearm"], 0]),
    "RightHand": np.array([0, -body_parts["Palm"], 0]),
    "LeftShoulder": np.array([0, body_parts["Shoulder width"] / 2, body_parts["Body"] / 3]),
    "LeftArm": np.array([0, body_parts["Upper arm"], 0]),
    "LeftForeArm": np.array([0, body_parts["Forearm"], 0]),
    "LeftHand": np.array([0, body_parts["Palm"], 0])
}
def rotation_matrix_to_quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()


def apply_quaternion_rotation(position, quaternion):
    r = R.from_quat(quaternion)
    return r.apply(position)


# 初始化速度和位置（单位为毫米）
velocities = {node: np.array([0., 0., 0.]) for node in NODES.keys()}
positions = {node: initial_positions[node] for node in NODES.keys()}

# 时间步长（60Hz，即每行数据对应的时间间隔是1/60秒）
dt = 1 / 60.0

for i in range(len(df)):
    for node, parent in NODES.items():
        if parent != "Root":
            # 更新速度（加速度单位从m/s²为mm/s²，乘以1000）
            acceleration_vector = df[[f"{node}_acceleration_{i}" for i in range(1, 4)]].iloc[i].values * 1000
            velocities[node] += acceleration_vector * dt

            # 更新位置
            positions[node] += velocities[node] * dt

            # 应用旋转矩阵
            rotation_matrix = df[[f"{node}_position_{i}{j}" for i in range(1, 4) for j in range(1, 4)]].iloc[
                i].values.reshape(3, 3)
            quaternion = rotation_matrix_to_quaternion(rotation_matrix)
            positions[node] = apply_quaternion_rotation(positions[node], quaternion)

# print(positions)

def plot_skeleton(positions, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-1000, 1000])

    # 绘制关节点
    for joint, position in positions.items():
        ax.scatter(position[0], position[1], position[2], label=joint)

    # 绘制骨骼（连接线）
    connections = [
        ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
        ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
        ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Neck1"),
        ("Neck1", "Head"),
        ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"),
        ("RightForeArm", "RightHand"),
        ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand")
    ]

    for (joint1, joint2) in connections:
        ax.plot(
            [positions[joint1][0], positions[joint2][0]],
            [positions[joint1][1], positions[joint2][1]],
            [positions[joint1][2], positions[joint2][2]], 'r-'
        )

    ax.legend()
    plt.savefig(filename)
    plt.close()


# 保存绘图为文件
plot_skeleton(positions, 'skeleton_plot.png')

