import glob
import os

import numpy as np
import torch
import tqdm
from torch_geometric.data import InMemoryDataset


class AMASSDataset(InMemoryDataset):
    def __init__(self, root, split_name, data_aug=False, transform=None, pre_transform=None):
        super(AMASSDataset, self).__init__(root, transform, pre_transform)
        self._split_name = split_name
        self._data_aug = data_aug
        self._root_dir = config.root_dir

        self._amass_anno_dir = config.amass_anno_dir

        self._amass_file_names = self._get_amass_names


        self.amass_motion_input_length = config.motion.amass_input_length
        self.amass_motion_target_length = config.motion.amass_target_length

        self.motion_dim = config.motion.dim
        self._load_skeleton()
        self._all_amass_motion_poses = self._load_all()
        self._file_length = len(self._all_amass_motion_poses)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._all_amass_motion_poses)

    def _load_all(self):
        all_amass_motion_poses = []
        for amass_motion_name in tqdm(self._amass_file_names):
            amass_info = np.load(amass_motion_name)
            amass_motion_poses = amass_info['poses']  # 156 joints(all joints of SMPL)
            N = len(amass_motion_poses)
            if N < self.amass_motion_target_length + self.amass_motion_input_length:
                continue

            frame_rate = amass_info['mocap_framerate']
            sample_rate = int(frame_rate // 25)
            sampled_index = np.arange(0, N, sample_rate)
            amass_motion_poses = amass_motion_poses[sampled_index]

            T = amass_motion_poses.shape[0]
            amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
            amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
            amass_motion_poses[:, 0] = 0

            p3d0_tmp = self.p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
            amass_motion_poses = ang2joint(
                p3d0_tmp,
                torch.tensor(amass_motion_poses).float(),
                self.parent
            ).reshape(-1, 52, 3)[:, 4:22]  #  .reshape(T, -1)

            num_node = amass_motion_poses.shape[1]

            # TODO 将amass_motion_poses转换为时空图数据，使用torch_geometric进行存储

            all_amass_motion_poses.append(amass_motion_poses)
        return all_amass_motion_poses

    @property
    def _get_amass_names(self):
        # create list
        seq_names = []

        if self._split_name == 'train':
            seq_names += np.loadtxt(
                os.path.join(self._amass_anno_dir.replace('amass', ''), "amass_train.txt"), dtype=str
            ).tolist()
        else:
            seq_names += np.loadtxt(
                os.path.join(self._amass_anno_dir.replace('amass', ''), "amass_test.txt"), dtype=str
            ).tolist()

        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        return list(file_list)

    @property
    def processed_file_names(self):
        return ['amass_data.pt']

    def _preprocess(self, amass_motion_feats):
        if amass_motion_feats is None:
            return None
        amass_seq_len = amass_motion_feats.shape[0]

        if self.amass_motion_input_length + self.amass_motion_target_length < amass_seq_len:
            start = np.random.randint(
                amass_seq_len - self.amass_motion_input_length - self.amass_motion_target_length + 1)
            end = start + self.amass_motion_input_length
        else:
            return None
