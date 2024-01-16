import argparse
import os
from typing import Union

import numpy as np
from torch.utils.data import DataLoader

from attacks import DFAttack
from attacks.modules import DFNet
from utils.data import MyDataset
from utils.general import parse_trace, feature_transform


class TamAttack(DFAttack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def _build_model(self):
        model = DFNet(length=self.args.seq_length, num_classes=self.nc, in_channels=2)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        TikTok feature extraction for a single trace
        """
        trace = parse_trace(data_path)
        feat = feature_transform(trace, feature_type='tam', seq_length=seq_length)
        return feat

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, is_train=True) -> (MyDataset, DataLoader):
        dataset = MyDataset(self.args, flist, labels, TamAttack.extract)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_train,
                            num_workers=self.args.workers)
        return dataset, loader
