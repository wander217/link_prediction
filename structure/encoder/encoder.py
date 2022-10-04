from structure.mics import PaCModule
from structure.encoder.position_encoder import PositionEncoder
from structure.encoder.transformer_encoder import TransformerEncoder
from typing import Dict
from torch import Tensor


class Encoder(PaCModule):
    def __init__(self,
                 hidden_channel: int,
                 text_encoder: Dict,
                 position_encoder: Dict):
        super().__init__()
        text_encoder['char_embedding_dim'] = hidden_channel
        position_encoder['out_channel'] = hidden_channel
        self.text_encoder: TransformerEncoder = TransformerEncoder(**text_encoder)
        self.position_encoder: PositionEncoder = PositionEncoder(**position_encoder)

    def forward(self, txt: Tensor, mask: Tensor, position: Tensor):
        """
        :param txt: text of document (N, T)
        :param mask: padding mask (N, T)
        :param position: position of bounding box (N, 4)
        :return: node feature (N, hidden_channel)
        """
        txt_feature = self.text_encoder(txt, mask)
        pos_feature = self.position_encoder(position)
        node_feature = txt_feature + pos_feature
        return node_feature
