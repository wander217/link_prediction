from structure.mics import PaCModule
import torch.nn as nn
from torch import Tensor


class DenseLayer(PaCModule):
    def __init__(self,
                 in_channel: int,
                 out_channel: int):
        super().__init__()
        self.fc: nn.Sequential = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channel, out_features=out_channel))

    def forward(self, x: Tensor):
        output: Tensor = self.fc(x)
        return output
