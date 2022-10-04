from typing import List, Dict
import numpy as np
import unicodedata
from pathlib import Path
import os


def remove_space(txt: str):
    return txt.strip().strip("\r\t").strip("\n")


class DocAlphabet:
    def __init__(self, path: str, max_len: int) -> None:
        self.encoded_pad: int = 0
        self.decoded_pad: str = '<pad>'
        path = os.path.join(Path(__file__).parent.parent, path)
        with open(path, 'r', encoding='utf-8') as f:
            alphabet = remove_space(f.readline())
            alphabet = ' ' + alphabet
            alphabet = unicodedata.normalize("NFC", alphabet)
        self._character: Dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self._character[self.decoded_pad] = self.encoded_pad
        self._number: Dict = {i + 1: c for i, c in enumerate(alphabet)}
        self._number[self.encoded_pad] = self.decoded_pad
        self.max_len: int = max_len

    def encode(self, txt: str) -> np.ndarray:
        # convert character to number
        txt = unicodedata.normalize("NFC", txt)
        encoded_txt: List = [self._character.get(char, self.encoded_pad)
                             for char in txt.lower()]
        return np.array(encoded_txt, dtype=np.int32)

    def decode(self, encoded_txt: np.ndarray) -> str:
        # convert number to character
        decoded_txt: List = [self._number.get(num, self.decoded_pad) for num in encoded_txt]
        # remove padding character
        decoded_txt = [char for char in decoded_txt if char != self.decoded_pad]
        return ''.join(decoded_txt)

    def size(self):
        return len(self._number)
