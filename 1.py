import glob
import os
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data, InMemoryDataset

from config import Config
from src.utils.angle_to_joint import ang2joint

config = Config()


class AMASSDataset(InMemoryDataset):
    def __init__(self, root, split_name, transform=None, pre_transform=None):
        self._split_name = split_name
        self._root_dir = config.root_dir
        self._amass_anno_dir = config.amass_anno_dir  # 获取要使用的amass数据集的路径
        self._amass_file_names = self._get_file_names()  # 获取所有训练数据路径

        self._load_skeleton()  # 获取边权重
        self._action_to_index = self.create_action_to_index_mapping()  # 获取标签

        self.amass_motion_input_length = config.motion.amass_input_length
        self.amass_motion_target_length = config.motion.amass_target_length
        self.motion_dim = config.motion.dim

        super(AMASSDataset, self).__init__(root, transform, pre_transform)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._all_amass_motion_poses)

    def _get_file_names(self):
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
    def raw_file_names(self):
        return self._amass_file_names

    @property
    def processed_file_names(self):
        return ['amass_data.pt']

    def create_action_to_index_mapping(self) -> Dict[str, int]:
        action_types = set(filename.split('/')[-1].split('-')[0] for filename in self._amass_file_names)
        return {action: index for index, action in enumerate(sorted(action_types))}

    def filename_to_onehot(self, filename: str) -> torch.Tensor:
        action_type = filename.split('/')[-1].split('-')[0]
        index = self._action_to_index[action_type]
        onehot = torch.zeros(len(self._action_to_index))
        onehot[index] = 1
        return onehot

    def _load_skeleton(self):
        # load skeleton info
        skeleton_info = np.load(
            os.path.join(self._root_dir, 'body_models', 'smpl_skeleton.npz')
        )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            self.parent[i] = parents[i]

    def _create_edge_index(self, num_node):
        # Create spatial edges based on skeleton structure
        spatial_edges = []
        for i in range(num_node):
            parent = self.parent.get(i, -1)
            if parent != -1:
                spatial_edges.append((i, parent))
                spatial_edges.append((parent, i))  # Add reverse edge for undirected graph

        edges = spatial_edges
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return edge_index

    def process(self):
        all_amass_motion_poses = []
        _amass_file_names = self._get_file_names()
        for amass_motion_name in _amass_file_names:
            # Check the category according to the file name,
            # take the characters before the "-" character as its label, and remove the spaces on both sides
            label = self.filename_to_onehot(amass_motion_name)

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
            ).reshape(-1, 52, 3)[:, 4:22]  # .reshape(T, -1)

            num_node = amass_motion_poses.shape[1]

            # Convert amass_motion_poses to spatio-temporal graph data
            edge_index = self._create_edge_index(num_node)

            for amass_motion_one_step in amass_motion_poses:
                x = amass_motion_one_step.clone().detach().requires_grad_(True)
                y = label
                data = Data(x=x, edge_index=edge_index, y=y)
                all_amass_motion_poses.append(data)

        datas, slices = self.collate(all_amass_motion_poses)
        torch.save((datas, slices), self.processed_dir[0])

        return all_amass_motion_poses

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

        return amass_motion_feats[start:end]

    def get(self, idx):
        return self._all_amass_motion_poses[idx]

    def test(self):
        """
        A simple test method to verify the dataset functionality.
        """
        print(f"Dataset length: {len(self)}")

        if len(self) > 0:
            sample = self[0]
            print(f"Sample type: {type(sample)}")
            print(f"Sample attributes: {sample.keys}")
            print(f"x shape: {sample.x.shape}")
            print(f"y shape: {sample.y.shape}")
            print(f"edge_index shape: {sample.edge_index.shape}")
            print(f"Number of nodes: {sample.num_nodes}")
            print(f"Number of edges: {sample.num_edges}")
        else:
            print("Dataset is empty.")

        return "Test completed."


# Usage example:
dataset = AMASSDataset(root='./', split_name='train')
test_result = dataset.test()
print(test_result)
