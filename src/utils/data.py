import argparse
import os
from typing import Callable

import numpy as np
import torch
import torch.utils.data as Data


class MyDataset(Data.Dataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray,
                 fn_extract: Callable[[str | os.PathLike, int], np.ndarray]):
        self.args = args
        self.flist = flist
        self.labels = labels
        self.fn_extract = fn_extract

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        sample_path = self.flist[idx]
        x = self.fn_extract(sample_path, self.args.seq_length)
        y = self.labels[idx]
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.flist)
