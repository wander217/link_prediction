from structure.mics import PaCModule
import torch.nn as nn
from torch import Tensor


class PositionEncoder(PaCModule):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.fc: nn.Linear = nn.Linear(in_features=in_channel, out_features=out_channel)
        self.norm: nn.LayerNorm = nn.LayerNorm(normalized_shape=out_channel)
        self.act: nn.ReLU = nn.ReLU()

    def forward(self, position: Tensor):
        """
        :param position: position feature (N, in_channel)
        :return: encoded bounding box position (N, out_channel)
        """
        output: Tensor = self.act(self.norm(self.fc(position)))  # (N, out_channel)
        return output
