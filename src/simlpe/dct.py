import numpy as np
import torch


def get_dct_matrix(T):
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
    return dct_matrix, idct_matrix


dct_m, idct_m = get_dct_matrix(50)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)
print(dct_m.shape)
