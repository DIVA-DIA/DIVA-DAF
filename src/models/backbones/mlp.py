from typing import List

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int]):

        super().__init__()
        in_ch = hidden_channels[0]
        self.mlp = nn.Sequential(nn.Linear(in_features=in_channels, out_features=in_ch), nn.ReLU())

        for out_chn in hidden_channels[1:]:
            self.mlp.append(nn.Linear(in_features=in_ch, out_features=out_chn))
            self.mlp.append(nn.ReLU())
            in_ch = out_chn

    def forward(self, x):
        return self.mlp(x)
