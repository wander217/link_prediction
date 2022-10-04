import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class GraphNorm(nn.Module):
    def __init__(self,
                 feature_num: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps: float = eps
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(feature_num))
        self.beta: nn.Parameter = nn.Parameter(torch.ones(feature_num))

    def _norm(self, x: Tensor):
        mean: Tensor = x.mean(dim=0, keepdim=True)
        std: Tensor = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + self.eps)

    def forward(self, x: Tensor, size: List):
        graphs: Tensor = torch.split(x, size)
        normed_graph: List = [self._norm(graph) for graph in graphs]
        output: Tensor = torch.cat(normed_graph, dim=0)
        output = self.gamma * output + self.beta
        return output
