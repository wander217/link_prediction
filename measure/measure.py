import numpy as np
from torch import Tensor


class Measure:

    def __call__(self, pred: Tensor, target: Tensor):
        pred_mat: np.ndarray = (pred > 0.5).int().detach().cpu().numpy()
        target_mat: np.ndarray = target.detach().cpu().numpy()
        n_correct: int = (pred_mat == target_mat).sum()
        return n_correct
