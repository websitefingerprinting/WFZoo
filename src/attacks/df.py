from typing import Union
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from utils.data import MyDataset
from utils.general import parse_trace, feature_transform, get_flist_label
from attacks.base import Attack
from attacks.modules import DFNet


class DFAttack(Attack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.nmc = args.mon_classes  # number of monitored classes
        self.nc = args.mon_classes + 1 if args.open_world else args.mon_classes  # number of total classes
        self.unmon_inst = args.unmon_inst if args.open_world else 0
        self.flist, self.labels = get_flist_label(self.args.data_path, mon_cls=self.nmc, mon_inst=self.args.mon_inst,
                                                  unmon_inst=self.unmon_inst, suffix=self.args.suffix)
        print("Number of data: {}".format(len(self.flist)))

    def _build_model(self):
        model = DFNet(length=self.args.seq_length, num_classes=self.nc)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        DF feature extraction for a single trace
        """
        trace = parse_trace(data_path)
        feat = feature_transform(trace, feature_type='df', seq_length=seq_length)
        return feat

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, is_train=True) -> (MyDataset, DataLoader):
        dataset = MyDataset(self.args, flist, labels, DFAttack.extract)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=is_train,
                            num_workers=self.args.workers)
        return dataset, loader

    def run(self, one_fold_only: bool = False):
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        for fold, (train_index, test_index) in enumerate(sss.split(self.flist, self.labels)):
            if one_fold_only and fold > 0:
                break
            train_list, train_labels = self.flist[train_index], self.labels[train_index]
            test_list, test_labels = self.flist[test_index], self.labels[test_index]
            self.train(train_list, train_labels, test_list, test_labels)

    def train(self, train_list: np.ndarray, train_labels: np.ndarray, val_list: np.ndarray, val_labels: np.ndarray):
        _, train_loader = self._get_data(train_list, train_labels, is_train=True)
        _, val_loader = self._get_data(val_list, val_labels, is_train=False)

        model = self._build_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adamax(model.parameters(), lr=self.args.lr0)

        trainer = create_supervised_trainer(model, optimizer, criterion, self.device)
        val_metrics = {
            "accuracy": Accuracy(),
            "loss": Loss(criterion)
        }
        val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device)

        @trainer.on(Events.ITERATION_COMPLETED(every=100))
        def log_training_loss(engine: Engine):
            print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine: Engine):
            val_evaluator.run(val_loader)
            metrics = val_evaluator.state.metrics
            print(
                f"Validation Results - Epoch[{engine.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} "
                f"Avg loss: {metrics['loss']:.2f}")

        trainer.run(train_loader, max_epochs=self.args.epochs)
