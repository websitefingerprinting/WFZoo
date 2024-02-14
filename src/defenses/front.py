import argparse
import os
from typing import Union

import numpy as np

from defenses.base import Defense
from defenses.config import FrontConfig
from utils.general import parse_trace, set_random_seed


class FrontDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = FrontConfig(args)
        self.config.load_config()

    @staticmethod
    def sample_from_rayleigh(wnd: float, n: int) -> np.ndarray:
        ts = sorted(np.random.rayleigh(wnd, n))
        return np.reshape(ts, (-1, 1))

    @set_random_seed
    def simulate(self, data_path: Union[str, os.PathLike], dump: bool = True) -> np.ndarray:
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
            self.dump_trace(data_path, defended_trace)
        return defended_trace
