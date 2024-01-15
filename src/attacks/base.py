import argparse
import os

import torch
from utils.logger import init_logger


class Attack(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = init_logger(str(self.__class__.__name__))

        self.device = self._acquire_device()

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            self.logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            self.logger.info('Use CPU')
        return device

    def _get_data(self, **kwargs):
        pass

    def vali(self, **kwargs):
        pass

    def run(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def test(self):
        pass
