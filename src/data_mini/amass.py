import glob
import os
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.angle_to_joint import ang2joint


class AMASSDataset(Dataset):
    """
    AMASSDataset is a PyTorch Dataset for loading AMASS motion data.
    """

    def __init__(self, config, split_name: str, data_aug: bool = True):
        """
        Args:
            config: The configuration object containing all the necessary parameters.
            split_name: str, the name of the split, can be 'train' or 'test'.
            data_aug: bool, optional, whether to perform data augmentation, default is True.
        """
        super(AMASSDataset, self).__init__()
        self._split_name = split_name  # 'train' or 'test'
        self._data_aug = data_aug  # whether to perform data augmentation
        self._root_dir = config.root_dir  # the root directory of the dataset
        self._amass_anno_dir = config.amass_anno_dir  # the directory containing the AMASS annotations
        self._amass_file_names = self._get_amass_names()  # the names of the AMASS files
        self.amass_motion_input_length = config.motion.amass_input_length  # the length of the AMASS motion input
        self.amass_motion_target_length = config.motion.amass_target_length  # the length of the AMASS motion target
        self.motion_dim = config.motion.dim  # the dimension of the motion
        self._load_skeleton()
        self._all_amass_motion_poses = self._load_all()
        self._file_length = len(self._all_amass_motion_poses)

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of samples in the dataset.
        """
        if self._file_length is not None:
            return self._file_length
        return len(self._all_amass_motion_poses)

    def _get_amass_names(self) -> List[str]:
        """
        Get the names of the AMASS files based on the split name.

        Returns:
            List[str]: A list of AMASS file names.
        """
        # create list
        seq_names = []
        if self._split_name == 'train':
            seq_names += self._load_names("amass_train.txt")
        else:
            seq_names += self._load_names("amass_test.txt")
        return self._get_file_list(seq_names)

    def _load_names(self, file_name: str) -> List[str]:
        """
        TODO: 这里存放的是Annotation文件的路径，里边应该写入数据文件的路径
        Load the names of the AMASS files from a text file.

        Args:
            file_name: str, the name of the text file.

        Returns:
            List[str]: A list of AMASS file names.
        """
        return np.loadtxt(
            os.path.join(self._amass_anno_dir.replace('amass', ''), file_name), dtype=str
        ).tolist()

    def _get_file_list(self, seq_names: List[str]) -> List[str]:
        """
        Get the list of AMASS file paths based on the sequence names.

        Args:
            seq_names: List[str], a list of sequence names.

        Returns:
            List[str]: A list of AMASS file paths.
        """
        file_list = []
        for dataset in seq_names:
            subjects = glob.glob(self._amass_anno_dir + '/' + dataset + '/*')
            for subject in subjects:
                if os.path.isdir(subject):
                    files = glob.glob(subject + '/*poses.npz')
                    file_list.extend(files)
        return file_list

    def _preprocess(self, amass_motion_feats):
        """
        Preprocess the AMASS motion features.

        Args:
            amass_motion_feats: The AMASS motion features to be preprocessed.

        Returns:
            The preprocessed AMASS motion features.
        """
        if amass_motion_feats is None:
            return None
        amass_seq_len = amass_motion_feats.shape[0]
        if self.amass_motion_input_length + self.amass_motion_target_length < amass_seq_len:
            start = np.random.randint(
                amass_seq_len - self.amass_motion_input_length - self.amass_motion_target_length + 1)
            end = start + self.amass_motion_input_length
        else:
            return None
        amass_motion_input = self._get_motion_input(amass_motion_feats, start, end)
        amass_motion_target = self._get_motion_target(amass_motion_feats, end)
        return torch.cat([amass_motion_input, amass_motion_target], axis=0)

    def _get_motion_input(self, amass_motion_feats, start: int, end: int):
        """
        Get the motion input from the AMASS motion features.

        Args:
            amass_motion_feats: The AMASS motion features.
            start: int, the start index of the motion input.
            end: int, the end index of the motion input.

        Returns:
            The motion input.
        """
        amass_motion_input = torch.zeros((self.amass_motion_input_length, amass_motion_feats.shape[1]))
        amass_motion_input[:end - start] = amass_motion_feats[start:end]
        return amass_motion_input

    def _get_motion_target(self, amass_motion_feats, end: int):
        """
        Get the motion target from the AMASS motion features.

        Args:
            amass_motion_feats: The AMASS motion features.
            end: int, the start index of the motion target.

        Returns:
            The motion target.
        """
        amass_motion_target = torch.zeros((self.amass_motion_target_length, amass_motion_feats.shape[1]))
        amass_motion_target[:self.amass_motion_target_length] = amass_motion_feats[
                                                                end:end + self.amass_motion_target_length]
        return amass_motion_target

    def _load_skeleton(self):
        """
        Load the skeleton information from the SMPL skeleton file.
        """
        skeleton_info = np.load(
            os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz')
        )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _load_all(self):
        """
        Load all the AMASS motion poses.

        Returns:
            A list of all the AMASS motion poses.
        """
        all_amass_motion_poses = []
        for amass_motion_name in tqdm(self._amass_file_names):
            amass_info = np.load(amass_motion_name)
            amass_motion_poses = amass_info['poses']  # 156 joints(all joints of SMPL)
            N = len(amass_motion_poses)
            if N < self.amass_motion_target_length + self.amass_motion_input_length:
                continue
            amass_motion_poses = self._sample_motion(amass_info, amass_motion_poses, N)
            all_amass_motion_poses.append(amass_motion_poses)
        return all_amass_motion_poses

    def _sample_motion(self, amass_info, amass_motion_poses, N):
        """
        Sample the AMASS motion poses based on the frame rate.

        Args:
            amass_info: The AMASS information.
            amass_motion_poses: The AMASS motion poses.
            N: The number of AMASS motion poses.

        Returns:
            The sampled AMASS motion poses.
        """
        frame_rate = amass_info['mocap_framerate']
        sample_rate = int(frame_rate // 25)
        sampled_index = np.arange(0, N, sample_rate)
        amass_motion_poses = amass_motion_poses[sampled_index]
        T = amass_motion_poses.shape[0]
        amass_motion_poses = R.from_rotvec(amass_motion_poses.reshape(-1, 3)).as_rotvec()
        amass_motion_poses = amass_motion_poses.reshape(T, 52, 3)
        amass_motion_poses[:, 0] = 0
        p3d0_tmp = self.p3d0.repeat([amass_motion_poses.shape[0], 1, 1])
        amass_motion_poses = ang2joint(p3d0_tmp,
                                       torch.tensor(amass_motion_poses).float(),
                                       self.parent).reshape(-1, 52, 3)[:, 4:22].reshape(T, -1)
        return amass_motion_poses

    def __getitem__(self, index):
        """
        Get the AMASS motion input and target at the given index.

        Args:
            index: The index of the AMASS motion input and target.

        Returns:
            A tuple of AMASS motion input and target.
        """
        amass_motion_poses = self._all_amass_motion_poses[index]
        amass_motion = self._preprocess(amass_motion_poses)
        if amass_motion is None:
            while amass_motion is None:
                index = np.random.randint(self._file_length)
                amass_motion_poses = self._all_amass_motion_poses[index]
                amass_motion = self._preprocess(amass_motion_poses)
        if self._data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(amass_motion.size(0) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                amass_motion = amass_motion[idx]
        amass_motion_input = amass_motion[:self.amass_motion_input_length].float()
        amass_motion_target = amass_motion[-self.amass_motion_target_length:].float()
        return amass_motion_input, amass_motion_target
