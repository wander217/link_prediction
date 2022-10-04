import torch
from structure.mics import PaCModule
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor


class FPN(PaCModule):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // 2, bias=False),
            nn.LayerNorm(normalized_shape=in_channel // 2),
            nn.Linear(in_features=in_channel // 2, out_features=in_channel // 4, bias=False),
            nn.LayerNorm(normalized_shape=in_channel // 4),
            nn.Linear(in_features=in_channel // 4, out_features=out_channel, bias=False),
            nn.LayerNorm(normalized_shape=out_channel),
            nn.LogSoftmax(dim=1))

    def forward(self,
                graph: DGLGraph,
                node_feature: Tensor):
        src, tgt = graph.edges()
        src_feature = node_feature[src]  # (N, D)
        tgt_feature = node_feature[tgt]  # (N, D)
        feature: Tensor = torch.cat([src_feature, tgt_feature], dim=1)  # (N, 2 * D)
        predict: Tensor = self.fc(feature)  # (N, 2)
        return predict
