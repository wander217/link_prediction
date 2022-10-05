from structure.encoder import Encoder
from structure.graph import GraphLayer
from structure.decoder import FPN
from structure.mics import PaCModule
from dataset import DocAlphabet
from torch import Tensor
from dgl import batch
from typing import Dict, List
from utils import weight_init


class DocLinkPrediction(PaCModule):
    def __init__(self,
                 encoder: Dict,
                 graph: Dict,
                 decoder: Dict,
                 alphabet: DocAlphabet):
        super().__init__()
        encoder['text_encoder']['vocab_len'] = alphabet.size()
        encoder['text_encoder']['max_len'] = alphabet.max_len
        self.encoder: Encoder = Encoder(**encoder)
        self.graph: GraphLayer = GraphLayer(**graph)
        self.decoder: FPN = FPN(**decoder)
        self.apply(weight_init)

    def forward(self,
                txt: Tensor,
                mask: Tensor,
                position: Tensor,
                graph: batch,
                node_factor: Tensor,
                node_num: List):
        node_feature: Tensor = self.encoder(txt=txt,
                                            mask=mask,
                                            position=position)
        graph, node_feature = self.graph(graph=graph,
                                         node_feature=node_feature,
                                         node_factor=node_factor,
                                         node_num=node_num)
        predict: Tensor = self.decoder(graph=graph,
                                       node_feature=node_feature)
        return predict
