import torch
from structure.mics import PaCModule
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor


class FPN(PaCModule):
    def __init__(self, in_channel: int, out_channel: int, k: int):
        super().__init__()
        # self.fc: nn.Sequential = nn.Sequential(
        #     nn.Linear(in_features=in_channel, out_features=in_channel // 2, bias=False),
        #     nn.LayerNorm(normalized_shape=in_channel // 2),
        #     nn.Linear(in_features=in_channel // 2, out_features=out_channel, bias=False),
        #     nn.LayerNorm(normalized_shape=out_channel),
        #     nn.LogSoftmax(dim=1))
        self.fc1: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // 2, bias=False),
            nn.LayerNorm(normalized_shape=in_channel // 2),
            nn.Linear(in_features=in_channel // 2, out_features=out_channel, bias=False),
            nn.LayerNorm(normalized_shape=out_channel),
            nn.ReLU(inplace=True)
        )
        self.fc2: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // 2, bias=False),
            nn.LayerNorm(normalized_shape=in_channel // 2),
            nn.Linear(in_features=in_channel // 2, out_features=out_channel, bias=False),
            nn.LayerNorm(normalized_shape=out_channel),
            nn.ReLU(inplace=True)
        )
        self.k: int = k

    def forward(self,
                graph: DGLGraph,
                node_feature: Tensor):
        src, tgt = graph.edges()
        src_feature: Tensor = self.fc1(node_feature[src])  # (N, 2)
        tgt_feature: Tensor = self.fc2(node_feature[tgt])  # (N, 2)
        predict: Tensor = torch.reciprocal(1. + torch.exp(torch.abs(tgt_feature - src_feature) * self.k))  # (N, 2)
        return predict.squeeze()
