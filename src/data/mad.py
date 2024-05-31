import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.angle_to_joint import ang2joint


class MADataset(Dataset):
    def __init__(self, root: str, train: bool, shuffle = False, transform=None):
        super(MADataset, self).__init__()
        self.train = train
        self.shuffle = shuffle
        self.root = root
        self.transform = transform if transform else self._preprocess
        self.motion_input_size: int = 50
        self.motion_target_size: int = 25
        self._load_skeleton()
        self._all_motion_poses = self._load_all()

    def __getitem__(self, item):
        motion_poses = self._all_motion_poses[item]
        motion_poses = self._preprocess(motion_poses)
        while motion_poses is None:
            idx = np.random.randint(self._file_length)
            motion_poses = self._all_motion_poses[idx]
            motion_poses = self._preprocess(motion_poses)

        # if self._data_aug:
        #     if np.random.rand() > .5:
        #         idx = [i for i in range(motion_poses.size(0) - 1, -1, -1)]
        #         idx = torch.LongTensor(idx)
        #         motion_poses = motion_poses[idx]

        motion_input = motion_poses[:self.motion_input_size].float()
        motion_target = motion_poses[self.motion_input_size: self.motion_input_size + self.motion_target_size].float()
        return motion_input, motion_target

    def __len__(self):
        return len(self._all_motion_poses)

    def _load_all(self):
        all_motion_poses = []
        # file name 是读取root目录下的所有npz文件
        _file_names = [os.path.join(self.root, file_name) for file_name in os.listdir(self.root) if
                       file_name.endswith('.npz') and "_NA_" not in file_name]

        # if self.shuffle:
        #     np.random.shuffle(_file_names)

        # 创建一个切片，用于分割训练集和验证集
        if self.train:
            _file_names = _file_names[:int(len(_file_names) * 0.8)]
        else:
            _file_names = _file_names[int(len(_file_names) * 0.8):]

        self._file_length = len(_file_names)
        for file_name in _file_names:

            info = np.load(file_name)
            # info = pd.read_csv(file_name, header=0)

            motion_poses = info['sample']
            N = len(motion_poses)
            if N < self.motion_input_size + self.motion_target_size:
                continue

            # down sample
            # frame_rate = 60
            # sample_rate = int(frame_rate // 25)
            # sampled_idx = np.arange(0, N, sample_rate)
            # motion_poses = motion_poses[sampled_idx]

            T = motion_poses.shape[0]
            # motion_poses = Rotation.from_rotvec(motion_poses.reshape(-1, 3)).as_rotvec()
            # TODO: definite what the 52 is.
            # motion_poses = motion_poses.reshape(T, 52, 3)
            # motion_poses[:, 0] = 0

            # TODO: what is the p3d0?
            p3d0_tmp = self.p3d0.repeat([motion_poses.shape[0], 1, 1])
            motion_poses = ang2joint(
                p3d0_tmp,
                torch.tensor(motion_poses).float(),
                self.parent
            ).reshape(T, -1)

            all_motion_poses.append(motion_poses)
        return all_motion_poses

    def _preprocess(self, motion_features):
        if motion_features is None:
            return None
        seq_len: int = motion_features.shape[0]

        # 针对每一个样本，如果当前样本（训练+测试）序列长度大于预设，则随机截取
        if self.motion_input_size + self.motion_target_size < seq_len:
            start_idx = np.random.randint(seq_len - self.motion_input_size - self.motion_target_size + 1)
            end_idx = start_idx + self.motion_input_size
        else:
            return None
        motion_input: torch.Tensor = torch.zeros((self.motion_input_size, motion_features.shape[1]))
        motion_input[:self.motion_input_size] = motion_features[start_idx:end_idx]

        motion_target: torch.Tensor = torch.zeros((self.motion_target_size, motion_features.shape[1]))
        motion_target[:self.motion_target_size] = motion_features[end_idx:end_idx + self.motion_target_size]

        motion: torch.Tensor = torch.cat((motion_input, motion_target), dim=0)

        return motion

    def _load_skeleton(self):
        skeleton_info = np.load(
            os.path.join(parent_parent_parent_dir, 'data', 'mad_skeleton.npz')
        )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i, parent in enumerate(parents):
            self.parent[i] = parent


if __name__ == '__main__':
    # 获取当前目录的父目录的父目录的父目录
    parent_parent_parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))

    # 改变当前的工作目录
    os.chdir(parent_parent_parent_dir)
    test_root = "data/amass"
    test_root = os.path.join(parent_parent_parent_dir, test_root)
    data = MADataset(test_root, train=True, shuffle=True)
    print(len(data))
