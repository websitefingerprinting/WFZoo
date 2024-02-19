import argparse
import os
from typing import Union

import numpy as np

from attacks import DFAttack
from utils.general import parse_trace, feature_transform


class TiktokAttack(DFAttack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        TikTok feature extraction for a single trace
        """
        trace = parse_trace(data_path)
        feat = feature_transform(trace, feature_type='tiktok', seq_length=seq_length)
        return feat
