import torch
from structure.mics import PaCModule
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor


class FPN(PaCModule):
    def __init__(self, in_channel: int, out_channel: int, k: int):
        super().__init__()
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // 2, bias=False),
            nn.LayerNorm(normalized_shape=in_channel // 2),
            nn.Linear(in_features=in_channel // 2, out_features=out_channel, bias=False),
            nn.LayerNorm(normalized_shape=out_channel),
            nn.LogSoftmax(dim=1))
        self.k: int = k

    def forward(self,
                graph: DGLGraph,
                node_feature: Tensor):
        src, tgt = graph.edges()
        src_feature: Tensor = node_feature[src]  # (N, D)
        tgt_feature: Tensor = node_feature[tgt]  # (N, D)
        feature: Tensor = torch.reciprocal(1. + torch.exp(self.k * torch.abs(tgt_feature - src_feature)))  # (N, D)
        predict: Tensor = self.fc(feature)  # (N, 2)
        return predict
