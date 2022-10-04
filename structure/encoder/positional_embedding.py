import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_len: int,
                 char_embedding_dim: int):
        """
        :param max_len: max length of sequence
        :param char_embedding_dim: dimension of channel after go through word embedding layer
        """
        super().__init__()
        positional_embedding: Tensor = torch.zeros(max_len, char_embedding_dim)
        position: Tensor = torch.arange(0, max_len).unsqueeze(1).float()
        hat: Tensor = torch.arange(0, char_embedding_dim, 2).float()
        hat = hat * torch.Tensor([-math.log(10000.0) / char_embedding_dim])
        div_term: Tensor = torch.exp(hat)
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        positional_embedding = positional_embedding.unsqueeze(0).unsqueeze(0)
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, x: Tensor):
        """
        :param x: embed text (n,t,d)
        :return: positional embed text (n,t,d)
        """
        output: Tensor = x + self.positional_embedding[:, :, :x.size(2), :]
        return output
