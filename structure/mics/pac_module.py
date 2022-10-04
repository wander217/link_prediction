import torch.nn as nn
import numpy as np


class PaCModule(nn.Module):
    def params_counter(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
