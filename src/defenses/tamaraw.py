import argparse
import os
from typing import Union

import numpy as np

from defenses.base import Defense
from defenses.config import TamarawConfig
from utils.general import parse_trace, set_random_seed


class TamarawDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = TamarawConfig(args)
        self.config.load_config()

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:

        trace1 = parse_trace(data_path)
        trace2 = self.reorder(trace1)
        trace3 = self.pad(trace2)

        return trace3

    def pad(self, trace: Union[np.array, list]) -> np.ndarray:
        # input is the reordered real packet list
        # output the padded list both in the middle and the end
        # always assume the first packet is an outgoing one
        trace = np.array(trace)
        last_t = trace[-1, 0]
        # find the last_t after padding the tail
        expect_last_outgoing_t = (((np.ceil(last_t / self.config.rho_client)) // self.config.lpad + 1) *
                                  self.config.lpad * self.config.rho_client)
        expect_last_incoming_t = (((np.ceil(last_t / self.config.rho_server)) // self.config.lpad + 1) *
                                  self.config.lpad * self.config.rho_server)
        expect_last_t = max(expect_last_outgoing_t, expect_last_incoming_t)

        outgoing_real = trace[trace[:, 1] > 0]
        incoming_real = trace[trace[:, 1] < 0]
        outgoing_pad_time = np.arange(0, expect_last_t, self.config.rho_client)
        outgoing_pad = np.column_stack((outgoing_pad_time, [self.DUMMY] * len(outgoing_pad_time)))
        incoming_pad_time = np.arange(0, expect_last_t, self.config.rho_server)
        incoming_pad = np.column_stack((incoming_pad_time, [-self.DUMMY] * len(incoming_pad_time)))
        trace_pad = np.concatenate((outgoing_pad, incoming_pad), axis=0)
        trace_pad = trace_pad[trace_pad[:, 0].argsort()]

        # start to insert dummy packets before the end
        cur_out_ind, cur_in_ind = 0, 0
        for ind, pkt in enumerate(trace_pad):
            curtime, direction = pkt[0], pkt[1]
            if direction > 0:
                if cur_out_ind < len(outgoing_real) and np.isclose(curtime, outgoing_real[cur_out_ind, 0]):
                    trace_pad[ind, 1] = 1
                    cur_out_ind += 1
            else:
                if cur_in_ind < len(incoming_real) and np.isclose(curtime, incoming_real[cur_in_ind, 0]):
                    trace_pad[ind, 1] = -1
                    cur_in_ind += 1
        assert cur_out_ind == len(outgoing_real)
        assert cur_in_ind == len(incoming_real)
        assert len(np.where(trace_pad[:, 1] == 1)[0]) == len(outgoing_real)
        assert len(np.where(trace_pad[:, 1] == -1)[0]) == len(incoming_real)
        return trace_pad

    def reorder(self, list1: np.ndarray) -> np.ndarray:
        T = [self.config.rho_client, self.config.rho_server]
        strategy = self.config.strategy

        list2 = [list1[0]]
        last = [0, 0]
        for cur, direction in list1[1:]:
            direction = int(direction)
            cursign = int((-direction + 1) / 2)  # [-1, 1] -> [1, 0]
            if strategy == 'pessimistic':
                timestamp = np.max([
                    self.find(cur, T[cursign], True),
                    self.find(last[0], T[cursign], False),
                    self.find(last[1], T[cursign], False),
                ])
            elif strategy == 'optimistic':
                timestamp = np.max([self.find(cur, T[cursign], True),
                                    self.find(last[cursign], T[cursign], False)])
            else:
                raise ValueError("Wrong strategy code: {}".format(strategy))
            timestamp = round(timestamp, 4)
            list2.append([timestamp, direction])
            last[cursign] = timestamp
        list2 = sorted(list2, key=lambda l: l[0])
        assert len(list2) == len(list1)
        return np.array(list2)

    @staticmethod
    def find(t, rho, is_edge):
        """
        find next nearest nT for t;
        when is_edge is True, t <= nT else t < nT
        """
        n = np.ceil(t / rho)
        if np.isclose(n * rho, t) and not is_edge:
            n = n + 1
        return n * rho
