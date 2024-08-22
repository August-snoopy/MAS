import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from src.aceMLP import MLPBlock


class SiMLPe(nn.Module):
    def __init__(self, fc_in=50, dim=51, fc_out=25):
        # (batch, num, dimension) e.g. (32, 50, 51)
        # the "num" is temporal dimension, and the "dimension" is spatial dimension
        super(SiMLPe, self).__init__()
        self.dct_m, _ = self.get_dct_matrix(fc_in)
        _, self.dct_im = self.get_dct_matrix(fc_out)

        self.arrange0 = Rearrange('b n d -> b d n')
        self.arrange1 = Rearrange('b d n -> b n d')

        self.fc_spatial0 = MLPBlock(dim)
        self.fc_dct_spatial0 = MLPBlock(dim)
        self.fc_spatial1 = MLPBlock(dim)

        self.temporal_mlp0 = nn.Sequential(*[MLPBlock(fc_in) for _ in range(31)])
        self.temporal_dct_mlp0 = nn.Sequential(*[MLPBlock(fc_in) for _ in range(31)])
        self.temporal_mlp2 = nn.Linear(fc_in * 2, fc_out)
        self.norm = nn.LayerNorm(fc_out)

        # self.attention = ScaledDotProductAttention(dim)
        self.attention = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.reset_parameters()

    def forward(self, motion_input):
        # [batch, num, joint, position] e.g. (32, 50, 17, 3)
        # first, we need to get the spatial features, notice that keep the dim same
        dct_features = torch.bmm(self.dct_m, motion_input)

        motion_features = self.fc_spatial0(motion_input)
        motion_features = F.relu(motion_features)
        dct_features = self.fc_dct_spatial0(dct_features)
        dct_features = F.relu(dct_features)

        # then we apply mlp on temporal dimension, notice that mlp out is fc_out
        motion_features = self.arrange0(motion_features)
        motion_features = self.temporal_mlp0(motion_features)
        motion_features = F.relu(motion_features)

        dct_features = self.arrange0(dct_features)
        dct_features = self.temporal_dct_mlp0(dct_features)
        dct_features = F.relu(dct_features)
        dct_features = torch.bmm(self.dct_im, dct_features)

        # join two features
        motion_features = torch.cat([motion_features, dct_features], dim=-1)
        motion_features = self.temporal_mlp2(motion_features)
        motion_features = self.norm(motion_features)
        F.relu(motion_features)

        motion_features = self.arrange1(motion_features)

        motion_features, _ = self.attention(motion_features, motion_features, motion_features)
        F.relu(motion_features)

        # finally, we just get the spatial features again
        motion_features = self.fc_spatial1(motion_features)

        return motion_features

    def get_dct_matrix(self, T):
        """
        Get the matrix of DCT transformation and inverse transformation, which is only related to the sequence length.
        Assuming that the original input is T*C and C is the feature dimension, the DCT matrix is T*T
        """
        dct_matrix = np.eye(T)
        # 预计算常数
        sqrt_2_T = np.sqrt(2 / T)
        sqrt_1_T = np.sqrt(1 / T)
        pi_over_T = np.pi / T

        # 生成i的值，只计算一次
        i_values = np.arange(T) + 0.5

        # 生成cos矩阵
        cos_matrix = np.cos(pi_over_T * np.outer(i_values, np.arange(T)))

        # 填充dct_matrix
        dct_matrix[0, :] = sqrt_1_T * cos_matrix[:, 0]
        dct_matrix[1:, :] = sqrt_2_T * cos_matrix[:, 1:].T
        idct_matrix = np.linalg.inv(dct_matrix)
        dct_matrix = torch.tensor(dct_matrix).float().unsqueeze(0).expand(T, -1, -1)
        idct_matrix = torch.tensor(idct_matrix).float().unsqueeze(0).expand(T, -1, -1)
        return dct_matrix, idct_matrix

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
