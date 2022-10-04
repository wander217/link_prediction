import torch

from structure.encoder import Encoder

model = Encoder(
    hidden_channel=512,
    text_encoder={
        "vocab_len": 230,
        "max_len": 100,
        "char_embedding_dim": 512,
        "n_head": 4,
        "dim_feedforward": 1024,
        "dropout": 0.3,
        "n_layer": 3
    },
    position_encoder={
        "in_channel": 4,
        "out_channel": 512
    }
)

out = model(torch.randint(1, 100, (6, 100)),
            torch.ones(6, 100),
            torch.randn(6, 4))
print(model.params_counter())
print(out.shape)
