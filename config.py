from __future__ import absolute_import, division, print_function

import json
import os.path as osp
import sys
import time
from dataclasses import dataclass, field, asdict


def add_path(path: str) -> None:
    """Add a directory to the system path if it's not already included."""
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass
class Motion:
    """Configuration for motion parameters."""
    amass_input_length: int = 50
    amass_input_length_dct: int = 50
    amass_target_length: int = 25
    amass_target_length_train: int = 25
    amass_target_length_eval: int = 25
    dim: int = 54
    pw3d_input_length: int = 50
    pw3d_target_length_train: int = 25
    pw3d_target_length_eval: int = 25

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass
class MotionMLP:
    """Configuration for the motion MLP network."""
    hidden_dim: int = 54
    seq_len: int = 50
    num_layers: int = 48
    with_normalization: bool = True
    spatial_fc_only: bool = False
    norm_axis: str = 'spatial'

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass
class MotionFCIn:
    """Configuration for the input fully connected layer of the motion network."""
    in_features: int = 54
    out_features: int = 54
    with_norm: bool = False
    activation: str = 'relu'
    init_w_trunc_normal: bool = False
    temporal_fc: bool = False

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass
class MotionFCOut:
    """Configuration for the output fully connected layer of the motion network."""
    in_features: int = 54
    out_features: int = 54
    with_norm: bool = False
    activation: str = 'relu'
    init_w_trunc_normal: bool = True
    temporal_fc: bool = False

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    """Main configuration class."""
    seed: int = 888
    abs_dir: str = field(init=False)
    this_dir: str = field(init=False)
    repo_name: str = 'MAS'
    root_dir: str = field(init=False)
    log_dir: str = field(init=False)
    snapshot_dir: str = field(init=False)
    log_file: str = field(init=False)
    link_log_file: str = field(init=False)
    val_log_file: str = field(init=False)
    link_val_log_file: str = field(init=False)
    amass_anno_dir: str = field(init=False)
    pw3d_anno_dir: str = field(init=False)
    motion: Motion = field(default_factory=Motion)
    data_aug: bool = True
    deriv_input: bool = True
    deriv_output: bool = True
    use_relative_loss: bool = True
    pre_dct: bool = False
    post_dct: bool = False
    motion_mlp: MotionMLP = field(default_factory=MotionMLP)
    motion_fc_in: MotionFCIn = field(default_factory=MotionFCIn)
    motion_fc_out: MotionFCOut = field(default_factory=MotionFCOut)
    batch_size: int = 32
    num_workers: int = 4
    cos_lr_max: float = 3e-4
    cos_lr_min: float = 5e-8
    cos_lr_total_iters: int = 100
    weight_decay: float = 1e-4
    model_pth: str = None
    shift_step: int = 5
    print_every: int = 10
    save_every: int = 5000

    def __post_init__(self):
        """Initialize paths and log files after the main initialization."""
        self.abs_dir = osp.dirname(osp.realpath(__file__))  # Absolute path of the current file
        self.this_dir = self.abs_dir.split(osp.sep)[-1]  # Name of the current directory
        self.root_dir = self.abs_dir[:self.abs_dir.index(self.repo_name) + len(self.repo_name)]  # Root directory
        self.log_dir = osp.abspath(osp.join(self.abs_dir, 'log'))
        self.snapshot_dir = osp.abspath(osp.join(self.log_dir, "snapshot"))
        exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self.log_file = f'{self.log_dir}/log_{exp_time}.log'
        self.link_log_file = f'{self.log_dir}/log_last.log'
        self.val_log_file = f'{self.log_dir}/val_{exp_time}.log'
        self.link_val_log_file = f'{self.log_dir}/val_last.log'
        self.amass_anno_dir = osp.join(self.root_dir, 'data/amass/')
        self.pw3d_anno_dir = osp.join(self.root_dir, 'data/3dpw/sequenceFiles/')

    def __iter__(self):
        return iter(self.__dict__.keys())

    def __getitem__(self, key):
        return self.__dict__[key]

    def __str__(self):
        return str(asdict(self))

    def to_dict(self):
        return asdict(self)


config = Config()
add_path(osp.join(config.root_dir, 'lib'))

if __name__ == '__main__':
    # print(config.motion_mlp)
    motion_mlp = MotionMLP()
    if 'seq_len' in motion_mlp:
        print(motion_mlp['seq_len'])