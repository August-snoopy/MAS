import torch
from torch import nn
from einops.layers.torch import Rearrange


class LayerNormal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.epsilon = 1e-6
        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LayerNormalV2(nn.Module):
    """
    这是另一种层归一化，但是这个归一化是在最后一个维度上进行的
    """
    def __init__(self, dim):
        super().__init__()
        self.epsilon = 1e-6
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class SpatialFC(nn.Module):
    def __init__(self, dim=54):
        """
        Args:
            dim: 隐藏层的维度,手动设置, 54
        """
        super(SpatialFC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange('b n d -> b d n')  # [batch, num, dim] -> [batch, dim, num]
        self.arr1 = Rearrange('b d n -> b n d')  # [batch, dim, num] -> [batch, num, dim]

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x


class TemporalFC(nn.Module):
    def __init__(self, seq):
        super(TemporalFC, self).__init__()
        self.fc = nn.Linear(seq, seq)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, dim, seq, use_spatial_fc=False, layer_norm_axis='spatial'):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = TemporalFC(seq)
        else:
            self.fc0 = SpatialFC(dim)

        if layer_norm_axis == 'spatial':
                self.norm0 = LayerNormal(dim)
        elif layer_norm_axis == 'temporal':
                self.norm0 = LayerNormalV2(seq)
        elif layer_norm_axis == 'all':
                self.norm0 = nn.LayerNorm([dim, seq])
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x


class TransMLP(nn.Module):
    def __init__(self, dim=54, seq=50, use_spatial_fc=False, num_layers=48, layer_norm_axis='spatial'):
        super().__init__()
        self.mlp = nn.Sequential(*[MLPBlock(dim, seq, use_spatial_fc, layer_norm_axis) for _ in range(num_layers)])

    def forward(self, x):
        x = self.mlp(x)
        return x
