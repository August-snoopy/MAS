import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

skeleton = dict()

skeleton['parents'] = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 7, 9, 10, 11, 7, 13, 14, 15]
# skeleton['parents'] = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 10, 11, 8, 13, 14, 15, 8, 17, 18, 19]
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
    "RightUpLeg": np.array([0, -body_parts["Hip width"] / 2, -body_parts["Upper leg"] / 2]),
    "RightLeg": np.array([0, -body_parts["Hip width"] / 2, -body_parts["Upper leg"] - body_parts["Lower leg"] / 2]),
    "RightFoot": np.array([body_parts["Foot length"], -body_parts["Hip width"] / 2,
                           -body_parts["Lower leg"] - body_parts["Lower leg"] - body_parts["Heel height"]]),
    "LeftUpLeg": np.array([0, body_parts["Hip width"] / 2, -body_parts["Upper leg"] / 2]),
    "LeftLeg": np.array([0, -body_parts["Hip width"] / 2, -body_parts["Upper leg"] - body_parts["Lower leg"] / 2]),
    "LeftFoot": np.array([body_parts["Foot length"], -body_parts["Hip width"] / 2,
                          -body_parts["Lower leg"] - body_parts["Lower leg"] - body_parts["Heel height"]]),
    # "Spine": np.array([0, 0, body_parts["Body"] * 1 / 4]),
    # "Spine1": np.array([0, 0, body_parts["Body"] * 2 / 4]),
    "Spine2": np.array([0, 0, body_parts["Body"] * 3 / 4]),
    # "Neck": np.array([0, 0, body_parts["Body"]]),
    # "Neck1": np.array([0, 0, body_parts["Body"] + body_parts["Neck"]]),
    "Head": np.array([0, 0, body_parts["Head"] + body_parts["Neck"] + body_parts["Body"]]),
    "RightShoulder": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"]]),
    "RightArm": np.array([0, -body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] / 2]),
    "RightForeArm": np.array([0, -body_parts["Shoulder width"] / 2,
                              body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"] / 2]),
    "RightHand": np.array(
        [0, -body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"]]),
    "LeftShoulder": np.array([0, body_parts["Shoulder width"] / 2, body_parts["Body"] / 3]),
    "LeftArm": np.array([0, body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] / 2]),
    "LeftForeArm": np.array([0, -body_parts["Shoulder width"] / 2,
                             body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"] / 2]),
    "LeftHand": np.array(
        [0, body_parts["Shoulder width"] / 2, body_parts["Body"] - body_parts["Upper arm"] - body_parts["Forearm"]])
}
joint_num = len(skeleton['parents'])
skeleton['p3d0'] = (np.array(list(initial_positions.values())) / 1000).reshape(-1, joint_num, 3)


# print(np.mean(skeleton['p3d0'], axis=1))
# 将skeleton存为npz文件
# np.savez("mad_skeleton.npz", **skeleton)


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

            # 生成切片并保存为新的文件
            for i in range(len(splits) - 1):
                start, end = splits[i], splits[i + 1]
                slice_df = df.iloc[start:end]
                label = slice_df.iloc[:, -1].values[0]

                # reshape becomes (number of frames, number of joints, 4, 3)
                # (4, 3) is (3, 3) rotation_instance rotation_matrix and (1, 3) acceleration vector
                sample = slice_df.values[:, :-13].reshape(-1, 21, 4, 3)

                # If only 17 original joints are needed, then remove some columns
                sample = np.delete(sample, [7, 8, 10, 11], axis=1)

                # acceleration information is not retained here
                sample = np.array(sample[:, :, :3, :], dtype=np.float32)

                # converts the rotation_instance rotation_matrix to a rotation_instance vector
                # the shape of the sample becomes (number of frames, number of joints, 3)
                rotation_vectors = np.zeros(sample.shape[:3])
                for j, rotation_matrix in enumerate(sample):
                    rotation_instance = R.from_matrix(rotation_matrix)
                    rotation_vectors[j] = rotation_instance.as_rotvec()

                sample = rotation_vectors

                # save as a npz file
                new_filename = f"{os.path.splitext(filename)[0]}_{i + 1}_{label}"
                new_filepath = os.path.join("amass2", new_filename)
                np.savez(new_filepath, sample=sample)

                # slice_df = pd.DataFrame(sample)

                # slice_df.to_csv(new_filepath, index=False)
                print(f"Saved {new_filepath}")


# 使用方法
# 替换为你的目录路径
split_csv_by_last_column("20240507")
