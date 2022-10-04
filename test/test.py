import torch

a = torch.FloatTensor([[1, 2], [2, 3]])
print(a.sum(dim=-1))