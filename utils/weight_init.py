import torch.nn as nn


def init_weight(module):
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0., 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1., 0.02)
        module.bias.data.fill_(0.)
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.normal_(1., 0.02)
        module.bias.data.fill_(0.)
    elif isinstance(module, nn.Linear):
        module.weight.data.fill_(1.)
        module.bias.data.fill_(0.)
