import torch
from structure.mics import PaCModule
from structure.graph.norm import GraphNorm
from torch import Tensor
from dgl import DGLGraph
from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F


def message(edge) -> Dict:
    bv_j: Tensor = edge.src['Bv']
    e_ij: Tensor = edge.src['Cv'] + edge.dst['Dv']
    score: Tensor = torch.sigmoid(e_ij)
    return {
        "Bv_j": bv_j,
        "score": score
    }


def reduce(node) -> Dict:
    av_i: Tensor = node.data['Av']
    bv_j: Tensor = node.mailbox['Bv_j']
    score: Tensor = node.mailbox['score']
    new_v = av_i + torch.sum(score * bv_j, dim=1) / (torch.sum(score, dim=1) + 1e-6)
    return {'v': new_v}


class GatedGCN(PaCModule):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 dropout: float):
        super().__init__()
        self.fc: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=in_channel, out_features=out_channel),
                nn.LayerNorm(normalized_shape=out_channel),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        self.norm: GraphNorm = GraphNorm(feature_num=out_channel)
        self._residual: bool = in_channel == out_channel
        self._dropout: float = dropout

    def forward(self,
                graph: DGLGraph,
                node_feature: Tensor,
                node_factor: Tensor,
                node_num: List):
        """
        :param graph: graph structure (node_feature + edge)
        :param node_feature: encoded feature (N, D)
        :param node_factor: node factor (N, 1)
        :param node_num: number of graph node (B, 1)
        :return: graph:  new graph structure (new_node_feature + edge)
                 new_node_feature: new_node_feature (N, D)
        """
        graph.ndata['v'] = node_feature
        for i, item in enumerate(['Av', 'Bv', 'Cv', 'Dv']):
            graph.ndata[item] = self.fc[i](node_feature)
        graph.update_all(message_func=message, reduce_func=reduce)
        new_node_feature = graph.ndata['v']
        new_node_feature = new_node_feature * node_factor
        new_node_feature = F.relu(self.norm(new_node_feature, node_num))
        if self._residual:
            new_node_feature = new_node_feature + node_feature
        new_node_feature = F.dropout(new_node_feature,
                                     p=self._dropout,
                                     training=self.training)
        return graph, new_node_feature
