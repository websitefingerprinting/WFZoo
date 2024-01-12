from typing import Union

import torch
from ignite.metrics import Metric


class WFMetric(Metric):
    def __init__(self, nmc: int, device: Union[str, torch.device] = torch.device("cpu")):
        self.nmc = nmc
        self._p = 0
        self._n = 0
        self._tp = 0
        self._fp = 0
        super(WFMetric, self).__init__(device=device)

    def reset(self):
        self._p = 0
        self._n = 0
        self._tp = 0
        self._fp = 0
        super(WFMetric, self).reset()

    def update(self, output: tuple):
        y_pred, y = output[0].detach(), output[1].detach()

        indices = torch.argmax(y_pred, dim=1)
        idx_p = y < self.nmc
        idx_n = y == self.nmc

        self._p += torch.sum(idx_p).item()
        self._n += torch.sum(idx_n).item()

        self._tp += torch.sum(indices[idx_p] == y[idx_p]).item()
        self._fp += torch.sum(indices[idx_n] != y[idx_n]).item()

    def compute(self):
        """
        :return: tp, fp, p, n
        """
        return self._tp, self._fp, self._p, self._n
