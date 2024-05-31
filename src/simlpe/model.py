from einops.layers.torch import Rearrange
from torch import nn

from src.simlpe import TransMLP


class SiMLPe(nn.Module):
    def __init__(self, fc_in=50, hidden=54, fc_out=25):
        super(SiMLPe, self).__init__()
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = TransMLP()

        self.temporal_fc_in = False
        self.temporal_fc_out = False
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(fc_in, fc_in)
        else:
            self.motion_fc_in = nn.Linear(hidden, hidden)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(fc_out, fc_out)
        else:
            self.motion_fc_out = nn.Linear(hidden, hidden)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):
        # [batch, num, joint, position]

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats
