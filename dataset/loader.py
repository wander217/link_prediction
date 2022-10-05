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
    new_texts = []
    for text in texts:
        for item in text:
            new_texts.append(np.pad(item,
                                    (0, max_len - item.shape[0]),
                                    'constant',
                                    constant_values=pad_encode))
    new_texts = np.array(new_texts)
    new_masks = []
    for mask in masks:
        for item in mask:
            new_masks.append(np.pad(item,
                                    (0, max_len - item.shape[0]),
                                    'constant',
                                    constant_values=pad_encode))
    new_masks = np.array(new_masks)
    print(new_texts[0], new_masks[0])
    new_bboxes = []
    for bbox in bboxes:
        for item in bbox:
            new_bboxes.append(item)
    new_bboxes = np.array(new_bboxes)
    node_sizes = [graph.number_of_nodes() for graph in graphs]
    node_factor: Tensor = get_factor(node_sizes)
    batched_graph = dgl.batch(graphs)

    return (batched_graph,
            torch.from_numpy(labels).byte(),
            torch.from_numpy(new_texts).long(),
            torch.from_numpy(new_bboxes).float(),
            torch.from_numpy(new_masks).bool(),
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
