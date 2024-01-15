import argparse
import os
from datetime import datetime
from typing import Union

import numpy as np

from defenses.base import Defense
from defenses.config import FrontConfig
from utils.general import parse_trace


class FrontDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = FrontConfig(args)
        self.config.load_config()

    @staticmethod
    def sample_from_rayleigh(wnd: float, n: int) -> np.ndarray:
        ts = sorted(np.random.rayleigh(wnd, n))
        return np.reshape(ts, (-1, 1))

    def simulate(self, data_path: Union[str, os.PathLike], dump: bool = True) -> np.ndarray:
        # pay attention that numpy may have the same random seed for a batch of multiprocessing processes
        # https://github.com/numpy/numpy/issues/9650
        # https://stackoverflow.com/questions/67691168/how-to-generate-different-random-values-at-each-subprocess-during-a-multiprocess
        np.random.seed(datetime.now().microsecond)

        trace = parse_trace(data_path)

        wnd_client = np.random.uniform(self.config.w_min, self.config.w_max)
        wnd_server = np.random.uniform(self.config.w_min, self.config.w_max)

        client_dummy_num = np.random.randint(1, self.config.n_client)
        server_dummy_num = np.random.randint(1, self.config.n_server)

        first_incoming_pkt_time = trace[np.where(trace[:, 1] < 0)][0, 0]
        last_pkt_time = trace[-1, 0]

        client_timetable = self.sample_from_rayleigh(wnd_client, client_dummy_num)
        client_timetable = client_timetable[np.where(self.config.start_t + client_timetable[:, 0] <= last_pkt_time)]

        server_timetable = self.sample_from_rayleigh(wnd_server, server_dummy_num)
        server_timetable[:, 0] += first_incoming_pkt_time
        server_timetable = server_timetable[np.where(self.config.start_t + server_timetable[:, 0] <= last_pkt_time)]

        client_pkts = np.concatenate((client_timetable, self.DUMMY * np.ones((len(client_timetable), 1))), axis=1)
        server_pkts = np.concatenate((server_timetable, -self.DUMMY * np.ones((len(server_timetable), 1))), axis=1)

        defended_trace = np.concatenate((trace, client_pkts, server_pkts), axis=0)
        defended_trace = defended_trace[defended_trace[:, 0].argsort(kind='mergesort')]

        # print("Client dummy number: {}, Server dummy number: {}".format(client_dummy_num, server_dummy_num))

        if dump:
            fname = data_path.split('/')[-1]
            dump_dir = os.path.join(self.output_dir, fname)
            np.savetxt(dump_dir, defended_trace, fmt='%.6f\t%d')
        return defended_trace
