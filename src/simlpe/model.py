from einops.layers.torch import Rearrange
from torch import nn

from src.simlpe import MLPBlock


class SiMLPe(nn.Module):
    def __init__(self, fc_in=50, dim=51, fc_out=25):
        # (batch, num, dimension) e.g. (32, 50, 51)
        # the "num" is temporal dimension, and the "dimension" is spatial dimension
        super(SiMLPe, self).__init__()
        self.arrange0 = Rearrange('b n d -> b d n')
        self.arrange1 = Rearrange('b d n -> b n d')

        self.fc_spatial0 = nn.Linear(dim, dim)
        self.fc_spatial1 = nn.Linear(dim, dim)
        self.temporal_mlp0 = nn.Sequential(*[MLPBlock(fc_in) for _ in range(31)])
        self.temporal_mlp1 = nn.Linear(fc_in, fc_out)
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

        # finally, we just get the spatial features again
        motion_features = self.fc_spatial1(motion_features)

        return motion_features

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
