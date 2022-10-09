from structure.mics import PaCModule
from structure.graph.gated_gcn import GatedGCN
from structure.graph.dense_layer import DenseLayer
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor
from typing import List
import torch


class GraphLayer(PaCModule):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 num_layer: int,
                 dropout: float):
        super().__init__()
        self.gcn: nn.ModuleList = nn.ModuleList([
            GatedGCN(in_channel=in_channel,
                     out_channel=out_channel,
                     dropout=dropout)
            for _ in range(num_layer)
        ])
        self.dense: nn.ModuleList = nn.ModuleList([
            DenseLayer(in_channel=out_channel + i * out_channel,
                       out_channel=out_channel)
            for i in range(1, num_layer + 1)
        ])

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
        all_node_feature: List = [node_feature]
        new_node_feature = node_feature
        for i, conv in enumerate(self.gcn):
            graph, new_node_feature = conv(graph=graph,
                                           node_feature=new_node_feature,
                                           node_factor=node_factor,
                                           node_num=node_num)
            all_node_feature.append(new_node_feature)
            concat_node = torch.cat(all_node_feature, dim=-1)
            new_node_feature = self.dense[i](concat_node)
        return graph, new_node_feature
