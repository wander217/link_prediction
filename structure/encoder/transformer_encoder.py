import torch
import torch.nn as nn
from structure.mics import PaCModule
from structure.encoder.positional_embedding import PositionalEmbedding
from torch import Tensor


class TransformerEncoder(PaCModule):
    def __init__(self,
                 vocab_len: int,
                 max_len: int,
                 char_embedding_dim: int,
                 n_head: int,
                 dim_feedforward: int,
                 dropout: float,
                 n_layer: int):
        """
        :param vocab_len: length of character dictionary
        :param max_len: max length of sequence
        :param char_embedding_dim: dimension of sequence after go through embedding layer
        :param n_head: number head of transformer encode layer
        :param dim_feedforward: dim_feedforward of transformer encode layer
        :param dropout: dropout rate of transformer encode layer
        :param n_layer: number layer of transformer encode layer
        """
        super().__init__()
        # embedding word
        self.word_embedding: nn.Module = nn.Embedding(vocab_len, char_embedding_dim)

        # encoding word
        transformer_encoder_layer: nn.Module = nn.TransformerEncoderLayer(d_model=char_embedding_dim,
                                                                          nhead=n_head,
                                                                          dim_feedforward=dim_feedforward,
                                                                          dropout=dropout)
        self.transformer_encoder: nn.Module = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer,
                                                                    num_layers=n_layer)
        self.positional_embedding: nn.Module = PositionalEmbedding(max_len=max_len,
                                                                   char_embedding_dim=char_embedding_dim)
        self.pe_dropout: nn.Module = nn.Dropout(p=dropout)
        self.out_dropout: nn.Module = nn.Dropout(p=dropout)
        self.norm: nn.Module = nn.LayerNorm(normalized_shape=char_embedding_dim)

    def forward(self, txt: Tensor, mask: Tensor):
        """
        :param txt: text in document batch (n,t)
        :param mask: mask for text in document batch (n,t)
        :return: txt_feature: text feature (n,d)
        """
        # embedding word
        embed_txt: Tensor = self.word_embedding(txt)  # (n,t,d)

        # generate key padding mask, indentify padding position and padding node
        # src_key_padding_mask, graph_node_mask = self.compute_mask(mask)
        src_key_padding_mask = torch.logical_not(mask)
        b_n, t, d = embed_txt.shape

        # embedding text
        embed_txt: Tensor = self.positional_embedding(embed_txt)  # (n,t,d)
        embed_txt = self.pe_dropout(embed_txt)  # (n,t,d)
        embed_txt = embed_txt.reshape(b_n, t, d)  # (n,t,d)

        # ensure batch_num is after
        embed_txt = embed_txt.transpose(0, 1).contiguous()  # (t, n,d)
        encode_txt = self.transformer_encoder(src=embed_txt, src_key_padding_mask=src_key_padding_mask)  # (t, n,d)

        # ensure batch_num is first
        encode_txt = encode_txt.transpose(0, 1).contiguous()  # (n,t,d)
        encode_txt = self.norm(encode_txt)  # (n,t,d)
        encode_txt = self.out_dropout(encode_txt)  # (n,t,d)

        # if true, this position is not padding
        # txt_mask: Tensor = torch.logical_not(src_key_padding_mask).byte()  # (b*n,t,d)
        txt_feature: Tensor = self.avg_pooling(encode_txt, mask.byte())  # (n ,d)

        # ensure padding position is always zeros
        # graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)  # (b*n ,1)
        # txt_feature = txt_feature * graph_node_mask.byte()  # (b*n ,d)
        # graph_node_mask = graph_node_mask.view(b, n, -1)
        return txt_feature

    # @staticmethod
    # def compute_mask(mask: Tensor):
    #     """
    #     :param mask: if false, this position is padded (B*N,T)
    #     :return: mask for padding position :
    #             src_key_padding_mask: (B*N,T)
    #             graph_node_mask: (B*N, T)
    #     """
    #     b_n, t = mask.shape
    #     mask_sum: Tensor = mask.sum(dim=-1)  # (b*n,)
    #
    #     # if true, this node is valid node
    #     graph_node_mask: Tensor = torch.logical_not(mask_sum.eq(0)).bool()  # (b*n,)
    #     graph_mode_mask = graph_node_mask.unsqueeze(-1).expand(b_n, t)  # (b * n, t)
    #
    #     # ensure weight of attention weight will not be nan after softmax
    #     # don't mask padding for invalid node
    #     # mask it after encode sequence
    #     src_key_padding_mask: Tensor = torch.logical_not(mask.bool()) & graph_mode_mask
    #     return src_key_padding_mask, graph_mode_mask

    @staticmethod
    def avg_pooling(txt: Tensor, txt_mask: Tensor):
        """
        :param txt: encoded text (n,t,d)
        :param txt_mask: if true, this position is not padding (n,t)
        :return: encoded text (n,d)
        """
        # ensure padding position is always zero
        txt = txt * txt_mask.detach().unsqueeze(2).float()  # (b*n,t,d)
        sum_out: Tensor = torch.sum(txt, dim=1)  # (b*n,d)
        txt_len: Tensor = txt_mask.float().sum(dim=1)  # (b*n, )
        txt_len = txt_len.unsqueeze(1).expand_as(sum_out)  # (b*n,d)

        # remove 0 value and replace with 1. to avoid dividing to 0
        txt_len = txt_len + txt_len.eq(0).float()  # (b*n,d)
        mean_out: Tensor = sum_out.div(txt_len)  # (b*n,d)
        return mean_out
