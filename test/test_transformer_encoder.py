from structure.encoder import TransformerEncoder
import torch

encoder = TransformerEncoder(vocab_len=230,
                             max_len=100,
                             char_embedding_dim=512,
                             n_head=4,
                             dim_feedforward=1024,
                             dropout=0.1,
                             n_layer=3)
print(encoder.params_counter())
txt_feature = encoder(torch.randint(1, 100, (6, 100)), torch.ones((6, 100)))
print("txt_feature:", txt_feature.shape)
