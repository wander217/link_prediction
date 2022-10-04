from torch.utils.data import Dataset
from dataset.alphabet import DocAlphabet
from typing import List, Dict
import dgl
import json
import numpy as np


def process(sample: Dict, alphabet: DocAlphabet):
    """
    Processing a document data
    :param sample: a dict containing keys:
        - img: image path
        - target: A list containing bbox information.
                  Each bbox is containing:
                  +, label:  label of this bbox
                  +, text: text inside this bbox
                  +, bbox: a polygon having type (8, 2)
    :param alphabet: use to encode text
    :return:
        - bboxes: contain 4 point coordinate including width and height
        - labels: contain label of bounding box in batch
        - texts: contain text of batch
        - lengths: contain length of each text in batch
    """
    TARGET_KEY = "target"
    TEXT_KEY = "text"
    BBOX_KEY = "box"
    LINKED_KEY = "linked"

    texts: List = []
    bboxes: List = []
    masks: List = []
    lengths: List = []
    for target in sample[TARGET_KEY]:
        text = alphabet.encode(target[TEXT_KEY])
        mask = np.ones(text.shape[0])
        lengths.append(text.shape[0])
        if text.shape[0] == 0:
            continue
        texts.append(text)
        bboxes.append(np.array(target[BBOX_KEY]).astype(np.int32))
        masks.append(mask)
    linked: List = sample[LINKED_KEY]
    return (np.array(bboxes),
            np.array(linked),
            np.array(texts),
            np.array(lengths),
            np.array(masks))


class DocDataset(Dataset):
    def __init__(self, path: str, alphabet: DocAlphabet, knn_num: int):
        self._alphabet: DocAlphabet = alphabet
        self._samples: List = []
        self._knn_num: int = knn_num
        self._load(path)

    def convert_data(self, sample):
        bboxes, linked, texts, lengths, masks = process(sample, self._alphabet)
        node_size = linked.shape[0]
        src: List = []
        dst: List = []
        labels: List = []
        for i in range(node_size):
            dist: List = []
            x_i, y_i, w_i, h_i = bboxes[i]
            for j in range(node_size):
                if i == j:
                    continue
                x_j, y_j, w_j, h_j = bboxes[j]
                dist.append([np.linalg.norm(np.array([x_i - x_j, y_i - y_j])), j])
            sorted(dist, key=lambda x: x[0])
            # select knn_num neighbor node
            for j in range(self._knn_num):
                labels.append(linked[i] == j)
                src.append(i)
                dst.append(j)
        graph = dgl.DGLGraph()
        graph.add_nodes(node_size)
        graph.add_edges(src, dst)
        return graph, labels, texts, lengths, bboxes, masks

    def _load(self, target_path: str):
        with open(target_path, 'r', encoding='utf-8') as f:
            samples: List = json.loads(f.read())
        self._samples.extend(samples)

    def __getitem__(self, index: int):
        result = self.convert_data(self._samples[index])
        return result

    def __len__(self):
        return len(self._samples)
