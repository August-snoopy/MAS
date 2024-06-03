import torch
from einops.layers.torch import Rearrange
from torch import nn

from src.simlpe import MLPBlock


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        attn_scores = torch.matmul(x, self.attention_weights)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = x * attn_weights.unsqueeze(-1)
        return attn_output



class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = dim ** -0.5

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        Q = self.query(x)  # (batch, seq_len, dim)
        K = self.key(x)  # (batch, seq_len, dim)
        V = self.value(x)  # (batch, seq_len, dim)

        # Compute attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (batch, seq_len, seq_len)
        attn_weights = self.softmax(attn_scores)  # (batch, seq_len, seq_len)

        # Compute attention output
        attn_output = torch.bmm(attn_weights, V)  # (batch, seq_len, dim)
        return attn_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.scaling_factor = torch.rsqrt(torch.tensor(dim, dtype=torch.float32))

    def forward(self, x):
        attn_scores = torch.matmul(x, x.transpose(-2, -1)) * self.scaling_factor
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, x)
        return attn_output


class SiMLPe(nn.Module):
    def __init__(self, fc_in=50, dim=51, fc_out=25):
        # (batch, num, dimension) e.g. (32, 50, 51)
        # the "num" is temporal dimension, and the "dimension" is spatial dimension
        super(SiMLPe, self).__init__()
        self.arrange0 = Rearrange('b n d -> b d n')
        self.arrange1 = Rearrange('b d n -> b n d')

        self.fc_spatial0 = nn.Linear(dim, dim)
        self.fc_spatial1 = nn.Linear(dim, dim)
        self.temporal_mlp0 = nn.Sequential(*[MLPBlock(fc_in) for _ in range(30)])
        self.temporal_mlp1 = nn.Linear(fc_in, fc_out)

        # self.attention = ScaledDotProductAttention(dim)
        self.attention = nn.MultiheadAttention(dim, 3, batch_first=True)
        self.reset_parameters()

    def forward(self, motion_input):
        # [batch, num, joint, position] e.g. (32, 50, 17, 3)
        # first, we need to get the spatial features, notice that keep the dim same
        motion_features = self.fc_spatial0(motion_input)

        # then we apply mlp on temporal dimension, notice that mlp out is fc_out
        motion_features = self.arrange0(motion_features)
        motion_features = self.temporal_mlp0(motion_features)
        motion_features = self.temporal_mlp1(motion_features)
        motion_features = self.arrange1(motion_features)
        # motion_features, _ = self.attention(motion_features, motion_features, motion_features)

        # finally, we just get the spatial features again
        motion_features = self.fc_spatial1(motion_features)

        return motion_features

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
