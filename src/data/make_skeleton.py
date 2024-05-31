import torch
import numpy as np
import xml.etree.ElementTree as ET

skeleton_info = np.load("smpl_skeleton (1).npz")

# 查看文件中的所有键
# print(skeleton_info['p3d0'])

skeleton = {}

skeleton['parents'] = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 7, 9, 10, 11, 7, 13, 14, 15]
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
    <desc name="Upper leg" length="500"/> <!-- 单位为毫米 -->
</body>
'''

root = ET.fromstring(xml_data)
body_parts = {desc.attrib['name']: float(desc.attrib['length']) for desc in root.findall('desc')}
initial_positions = {
    "Hips": np.array([0, 0, 0]),
    "RightUpLeg": np.array([0, -body_parts["Hip width"] / 2, -body_parts["Upper leg"]/2]),
    "RightLeg": np.array([0, -body_parts["Hip width"] / 2, -body_parts["Upper leg"] - body_parts["Lower leg"] / 2]),
    "RightFoot": np.array([body_parts["Foot length"], -body_parts["Hip width"] / 2, -body_parts["Lower leg"] - body_parts["Lower leg"] - body_parts["Heel height"]]),
    "LeftUpLeg": np.array([0, body_parts["Hip width"] / 2, -body_parts["Upper leg"]/2]),
    "LeftLeg": np.array([0, -body_parts["Hip width"] / 2, -body_parts["Upper leg"] - body_parts["Lower leg"] / 2]),
    "LeftFoot": np.array([body_parts["Foot length"], -body_parts["Hip width"] / 2, -body_parts["Lower leg"] - body_parts["Lower leg"] - body_parts["Heel height"]]),
    # "Spine": np.array([0, 0, body_parts["Body"] / 2]),
    # "Spine1": np.array([0, 0, body_parts["Body"] / 3]),
    "Spine2": np.array([0, 0, body_parts["Body"] * 2 / 3]),
    # "Neck": np.array([0, 0, body_parts["Neck"]]),
    # "Neck1": np.array([0, 0, body_parts["Neck"]]),
    "Head": np.array([0, 0, body_parts["Head"] + body_parts["Neck"] + body_parts["Body"]]),
    "RightShoulder": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"]]),
    "RightArm": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"]/2]),
    "RightForeArm": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"] / 2]),
    "RightHand": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"]]),
    "LeftShoulder": np.array([0, body_parts["Shoulder width"] / 2, body_parts["Body"] / 3]),
    "LeftArm": np.array([0, body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"]/2]),
    "LeftForeArm": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"] / 2]),
    "LeftHand": np.array([0, body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"]])
}
skeleton['p3d0'] = (np.array(list(initial_positions.values())) / 1000).reshape(-1, 17, 3)

# print(np.mean(skeleton['p3d0'], axis=1))
# 将skeleton存为npz文件
# np.savez("mad_skeleton.npz", **skeleton)

def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.
    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    p3d = torch.from_numpy(offset).float().to(rotmat.device).unsqueeze(0).repeat(n, 1, 1).clone()
    R = rotmat.view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    return p3d

import pandas as pd

example = pd.read_csv("test.csv", header=0)
# example = df.iloc[1]
example = example.values[:, :-13].reshape(-1, 21, 4, 3)
# .reshape(-1, 21, 4, 3))
# 去除example 的第7，8，10，11列
#
example = np.delete(example, [7, 8, 10, 11], axis=1)
# example = np.concatenate((example[:, 7], example[:, 9:10], example[:, 12:]), axis=1)

example = np.array(example[:, :, :3, :], dtype=np.float32)
# example = torch.from_numpy(example).float()
n = example.data.shape[0]
# print(example.shape)
# p3d = skeleton['p3d0']
p3d0 = np.array(skeleton['p3d0'], dtype=np.float32)
p3d =  torch.from_numpy(p3d0).float().repeat(n, 1, 1).clone()
# print(p3d.shape)
# parent = skeleton["parents"]
# for i in np.arange(1, 17):
    # if parent[i] > 0:
        # example[:, i, :, :] = torch.matmul(example[:, i, :, :], example[:, parent[i], :, :]).clone()
        # p3d[:, i, :] = torch.matmul(p3d[0, i, :], example[:, parent[i], :, :]) + p3d[:, parent[i], :]

# print(p3d.shape)

from scipy.spatial.transform import Rotation as R
rotvecs = np.zeros(example.shape[:3])

for i, matrix in enumerate(example):
    rotation = R.from_matrix(matrix)
    rotvecs[i] = rotation.as_rotvec()

print(rotvecs.shape)


def split_csv_by_last_column(directory):
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)

            # 获取最后一列的值
            last_column = df.columns[-1]
            df[last_column] = df[last_column].fillna('_NA_')
            values = df[last_column].tolist()

            # 用来存储每一个切片的起始位置
            splits = [0]
            nan_value = np.nan

            # 找到切片的分界点
            for i in range(1, len(values)):
                if values[i] != values[i - 1] and values != nan_value:
                    splits.append(i)
                    # print(i)
            splits.append(len(values))
            print(splits)

            # 生成切片并保存为新的CSV文件
            for i in range(len(splits) - 1):
                start, end = splits[i], splits[i + 1]
                slice_df = df.iloc[start:end]
                label = slice_df.iloc[:, -1].values[0]

                sample = slice_df.values[:, :-13].reshape(-1, 21, 4, 3)
                sample = np.delete(sample, [7, 8, 10, 11], axis=1)
                sample = np.array(sample[:, :, :3, :], dtype=np.float32)
                rotvecs = np.zeros(sample.shape[:3])

                for j, matrix in enumerate(sample):
                    rotation = R.from_matrix(matrix)
                    rotvecs[j] = rotation.as_rotvec()

                sample = rotvecs

                # 保存为npz文件
                new_filename = f"{os.path.splitext(filename)[0]}_{i + 1}_{label}"
                new_filepath = os.path.join("amass", new_filename)
                np.savez(new_filepath, sample=sample)

                # slice_df = pd.DataFrame(sample)

                # slice_df.to_csv(new_filepath, index=False)
                print(f"Saved {new_filepath}")
# 使用方法
directory = "raw"  # 替换为你的目录路径
split_csv_by_last_column(directory)
