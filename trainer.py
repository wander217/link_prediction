import torch
import yaml
import argparse
from typing import Dict
from structure import DocLinkPrediction
from dataset import DocAlphabet, DocLoader
from utils import Logger, Checkpoint, Averager
from measure import Measure
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import warnings
import os


class Trainer:
    def __init__(self,
                 alphabet: Dict,
                 structure: Dict,
                 optimizer: Dict,
                 train_loader: Dict,
                 valid_loader: Dict,
                 checkpoint: Dict,
                 logger: Dict,
                 start_epoch: int,
                 total_epoch: int):
        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        self._alphabet: DocAlphabet = DocAlphabet(**alphabet)
        # model constructor
        self._model: DocLinkPrediction = DocLinkPrediction(**structure, alphabet=self._alphabet)
        self._model = self._model.to(self._device)
        # criterion constructor
        self._criterion: nn.BCELoss = nn.BCELoss()
        self._criterion = self._criterion.to(self._device)
        # optimizer constructor
        optim_cls = getattr(optim, optimizer['name'])
        self._optim = optim_cls(self._model.parameters(), **optimizer['params'])
        # checkpoint constructor
        self._checkpoint: Checkpoint = Checkpoint(**checkpoint)
        # logger constructor
        self._logger: Logger = Logger(**logger)
        # measure constructor
        self._measure: Measure = Measure()
        # dataset constructor
        self._train_loader: DataLoader = DocLoader(**train_loader, alphabet=self._alphabet).build()
        self._valid_loader: DataLoader = DocLoader(**valid_loader, alphabet=self._alphabet).build()
        self._start_epoch: int = start_epoch
        self._total_epoch: int = total_epoch

    def train(self):
        self.load()
        self._logger.report_delimiter()
        self._logger.report_time("Starting:")
        self._logger.report_delimiter()
        for epoch in range(self._start_epoch, self._total_epoch + 1):
            self._logger.report_delimiter()
            self._logger.report_time("Epoch {}:".format(epoch))
            self._logger.report_delimiter()
            train_rs = self.train_step()
            valid_rs = self.valid_step()
            self.save(train_rs, valid_rs, epoch)
        self._logger.report_delimiter()
        self._logger.report_time("Finish:")
        self._logger.report_delimiter()

    def train_step(self):
        self._model.train()
        train_loss: Averager = Averager()
        accurate: Averager = Averager()
        for batch, (graphs, labels,
                    texts, bboxes,
                    masks, node_factors,
                    node_sizes) in enumerate(self._train_loader):
            self._optim.zero_grad()
            graphs = graphs.to(self._device)
            labels = labels.to(self._device)
            texts = texts.to(self._device)
            masks = masks.to(self._device)
            bboxes = bboxes.to(self._device)
            predict = self._model(graph=graphs,
                                  mask=masks,
                                  position=bboxes,
                                  txt=texts)
            loss = self._criterion(predict, labels.float())
            loss.backward()
            self._optim.step()
            train_loss.update(loss.item() * labels.size(0), labels.size(0))
            bath_acc: float = self._measure(predict, labels)
            accurate.update(bath_acc, labels.size(0))
        return {
            "loss": train_loss.calc(),
            "acc": accurate.calc(),
        }

    def valid_step(self):
        self._model.eval()
        valid_loss: Averager = Averager()
        accurate: Averager = Averager()
        with torch.no_grad():
            for batch, (graphs, labels,
                        texts, bboxes,
                        masks, node_factors,
                        node_sizes) in enumerate(self._valid_loader):
                graphs = graphs.to(self._device)
                labels = labels.to(self._device)
                texts = texts.to(self._device)
                masks = masks.to(self._device)
                bboxes = bboxes.to(self._device)
                predict = self._model(graph=graphs,
                                      mask=masks,
                                      position=bboxes,
                                      txt=texts)
                loss = self._criterion(predict, labels.float())
                bath_acc: float = self._measure(predict, labels)
                valid_loss.update(loss.item() * labels.size(0), labels.size(0))
                accurate.update(bath_acc, labels.size(0))
        return {
            "loss": valid_loss.calc(),
            "acc": accurate.calc(),
        }

    def load(self):
        state_dict: Tuple = self._checkpoint.load()
        if state_dict is not None:
            self._model.load_state_dict(state_dict[0])
            self._optim.load_state_dict(state_dict[1])
            self._start_epoch = state_dict[2] + 1

    def save(self, train_rs: Dict, valid_rs: Dict, epoch: int):
        self._logger.report_metric("training", train_rs)
        self._logger.report_metric("validation", valid_rs)
        self._logger.write({
            'training': train_rs,
            'validation': valid_rs
        })
        self._checkpoint.save_last(epoch, self._model, self._optim)
        self._checkpoint.save_model(self._model, epoch)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser("Training config")
    parser.add_argument("-c", "--config_path", type=str, default='', help="config path")
    parser.add_argument("-p", "--root_path", type=str, default='', help="dataset root path")
    parser.add_argument("-a", "--alphabet_path", type=str, default='', help="alphabet path")
    parser.add_argument("-r", "--resume", type=str, default='', help="checkpoint path")
    args = parser.parse_args()
    if args.config_path.strip():
        with open(args.config_path.strip()) as f:
            data: dict = yaml.safe_load(f)
    if args.alphabet_path.strip():
        data['alphabet']['path'] = args.alphabet_path.strip()
    if args.resume.strip():
        data['checkpoint']['resume'] = args.resume.strip()
    if args.root_path.strip():
        tmp: str = args.root_path.strip()
        for item in ['train', 'valid']:
            data["{}_loader".format(item)]['dataset']['path'] = os.path.join(tmp, "{}.json".format(item))
    trainer = Trainer(**data)
    trainer.train()
