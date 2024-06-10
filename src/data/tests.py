import os
import numpy as np
import pandas as pd
NODES = {
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


root = '.'
_file_names = [os.path.join(root, file_name) for file_name in os.listdir(root) if file_name.endswith('.csv')]
for file in _file_names:
    info = pd.read_csv(file, header=0)
    for i in range(len(info)):
        x = []
        for node in NODES:
            rotation_matrix = info[[f"{node}_position_{i}{j}" for i in range(1, 4) for j in range(1, 4)]].iloc[i].values
            x.append(rotation_matrix)
        x = np.array(x, dtype=np.float32).reshape(21, -1)

print(x[:5])