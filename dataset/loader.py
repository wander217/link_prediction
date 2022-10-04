import torch
from typing import Tuple, List, Dict
import numpy as np
from torch import Tensor
from dataset.alphabet import DocAlphabet
from dataset.dataset import DocDataset
from torch.utils.data import DataLoader
import dgl


def get_factor(sizes: List) -> Tensor:
    tab_snorm: List = [torch.ones((size, 1)).float() / float(size) for size in sizes]
    factor: Tensor = torch.cat(tab_snorm).sqrt()
    return factor


def graph_collate(batch: Tuple, pad_encode: int = 0):
    graphs, labels, texts, lengths, bboxes, masks = map(list, zip(*batch))
    # label of edge
    labels = np.concatenate(labels).flatten()
    # find max length to padding
    max_len: int = np.max(np.concatenate(lengths))
    new_text: List = [np.expand_dims(np.pad(text,
                                            (0, max_len - text.shape[0]),
                                            'constant',
                                            constant_values=pad_encode), axis=0)
                      for text in texts]
    texts = np.concatenate(new_text)
    new_mask: List = [np.expand_dims(np.pad(mask,
                                            (0, max_len - mask.shape[0]),
                                            'constant',
                                            constant_values=pad_encode), axis=0)
                      for mask in masks]
    marks = np.concatenate(new_mask)
    node_sizes = [graph.number_of_nodes() for graph in graphs]
    node_factor: Tensor = get_factor(node_sizes)
    batched_graph = dgl.batch(graphs)

    return (batched_graph,
            torch.from_numpy(labels),
            torch.from_numpy(texts),
            torch.from_numpy(bboxes),
            torch.from_numpy(marks),
            node_factor,
            node_sizes)


class DocLoader:
    def __init__(self,
                 num_workers: int,
                 batch_size: int,
                 drop_last: bool,
                 shuffle: bool,
                 pin_memory: bool,
                 dataset: Dict,
                 alphabet: DocAlphabet):
        self.dataset: DocDataset = DocDataset(**dataset, alphabet=alphabet)
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=graph_collate
        )
