import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0., 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1., 0.02)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.normal_(1., 0.02)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Linear):
        m.weight.data.fill_(1.)
        if m.bias is not None:
            m.bias.data.fill_(0.)
