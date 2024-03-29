import argparse
import os
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from utils.data import MyDataset
from utils.general import get_flist_label
from utils.general import timeit
from utils.logger import init_logger
from utils.metric import WFMetric


class Attack(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = init_logger(str(self.__class__.__name__))

        self.device = self._acquire_device()

        self.nmc = args.mon_classes  # number of monitored classes
        self.nc = args.mon_classes + 1 if args.open_world else args.mon_classes  # number of total classes
        self.unmon_inst = args.unmon_inst if args.open_world else 0
        self.flist, self.labels = get_flist_label(self.args.data_path, mon_cls=self.nmc, mon_inst=self.args.mon_inst,
                                                  unmon_inst=self.unmon_inst, suffix=self.args.suffix)
        self.logger.info("Number of data: {}".format(len(self.flist)))

    def _build_model(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        Input format from the raw traces
        """
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

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, feat_extract_func: Callable, is_train=True) -> (
            MyDataset, DataLoader):
        dataset = MyDataset(self.args, flist, labels, feat_extract_func)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_train,
                            num_workers=self.args.workers)
        return dataset, loader

    @timeit
    def run(self, one_fold_only: bool = False):
        res = np.zeros(4)  # tp, fp, p, n
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)
        for fold, (train_index, test_index) in enumerate(sss.split(self.flist, self.labels)):
            if one_fold_only and fold > 0:
                break
            train_list, train_labels = self.flist[train_index], self.labels[train_index]
            test_list, test_labels = self.flist[test_index], self.labels[test_index]
            res_one_fold = self.train(fold + 1, train_list, train_labels, test_list, test_labels)
            res += res_one_fold
            self.logger.info("-" * 10)
        self.logger.info("Total: tp: {:.0f}, fp: {:.0f}, p: {:.0f}, n: {:.0f}".format(res[0], res[1], res[2], res[3]))

    def train(self, fold: int, train_list: np.ndarray, train_labels: np.ndarray, val_list: np.ndarray,
              val_labels: np.ndarray) -> np.ndarray:
        _, train_loader = self._get_data(train_list, train_labels, self.extract, is_train=True)
        _, val_loader = self._get_data(val_list, val_labels, self.extract, is_train=False)

        model = self._build_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(model.parameters(), lr=self.args.lr0)

        amp_mode = 'amp' if self.args.amp and self.device != torch.device("cpu") else None

        trainer = create_supervised_trainer(model, optimizer, criterion, self.device, amp_mode=amp_mode)
        val_metrics = {
            "accuracy": WFMetric(self.nmc),
            "loss": Loss(criterion)
        }
        val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=amp_mode)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_loss(engine: Engine):
            self.logger.info(f"Fold[{fold}] | Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] |"
                             f" Loss: {engine.state.output:.2f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine: Engine):
            val_evaluator.run(val_loader)
            _metrics = val_evaluator.state.metrics
            self.logger.info(
                f"Validation Results - Fold[{fold}] Epoch[{engine.state.epoch}] | "
                f"Avg loss: {_metrics['loss']:.2f} | "
                f"tp: {_metrics['accuracy'][0]:4.0f} fp: {_metrics['accuracy'][1]:4.0f} "
                f"p: {_metrics['accuracy'][2]:4.0f} n: {_metrics['accuracy'][3]:4.0f}"
            )

        trainer.run(train_loader, max_epochs=self.args.epochs)
        metrics = val_evaluator.state.metrics

        torch.cuda.empty_cache()
        return np.array(metrics['accuracy'])

    def test(self):
        pass

    def vali(self, **kwargs):
        pass
