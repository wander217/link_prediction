import torch
import os
import torch.nn as nn
import torch.optim as optim
from typing import Any
from collections import OrderedDict


class Checkpoint:
    def __init__(self, workspace: str, resume: str):
        self._workspace: str = workspace
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self._resume: str = resume.strip()

    def save_last(self,
                  epoch: int,
                  model: nn.Module,
                  optimizer: optim.Optimizer):
        last_path: str = os.path.join(self._workspace, "last.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, last_path)

    def save_model(self, model: nn.Module, epoch: int) -> Any:
        path: str = os.path.join(self._workspace, "checkpoint_{}.pth".format(epoch))
        torch.save({"model": model.state_dict()}, path)

    def load(self, device=torch.device('cpu')):
        if isinstance(self._resume, str) and bool(self._resume):
            data: OrderedDict = torch.load(self._resume, map_location=device)
            model: OrderedDict = data.get('model')
            optimizer: OrderedDict = data.get('optimizer')
            epoch: int = data.get('epoch')
            return model, optimizer, epoch

    @staticmethod
    def load_path(path: str, device=torch.device('cpu')) -> OrderedDict:
        data: OrderedDict = torch.load(path, map_location=device)
        assert 'model' in data
        model: OrderedDict = data.get('model')
        return model
