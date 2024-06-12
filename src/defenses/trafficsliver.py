import argparse
import os
from typing import Union, List

import numpy as np

from defenses.base import Defense
from defenses.config import TrafficSliverConfig
from utils.general import parse_trace, set_random_seed


class TrafficSliverDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = TrafficSliverConfig(args)
        self.config.load_config()
        self.latencies, self.client_ids = self.get_circuit_latencies(self.config.latency_file_path)

    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike]) -> np.ndarray:

        n_circuits = self.config.n_circuits
        batch_size_min = self.config.batch_size_min
        batch_size_max = self.config.batch_size_max
        latencies = self.latencies
        client_ids = self.client_ids

        trace = parse_trace(data_path)
        trace[:, 1] = np.sign(trace[:, 1])  # in case we are simulating defended traces, 888 -> 1

        w_out, w_in = self.get_weights(n_circuits)
        last_client_route, last_server_route = self.get_path(n_circuits, w_out), self.get_path(n_circuits, w_in)

        # record the path id for each cell
        route_info = []

        # sample a batch size and send such volume on this path
        batch_size_out = np.random.randint(batch_size_min, batch_size_max + 1)
        batch_size_in = np.random.randint(batch_size_min, batch_size_max + 1)
        cnt_out, cnt_in = 0, 0

        latency_info = self.sample_latencies(latencies, client_ids, n_circuits)

        debug_route_used = [0] * n_circuits

        for _, direction in trace:
            if direction == 1:
                route_info.append(last_client_route)
                debug_route_used[last_client_route] = 1
                cnt_out += 1
                if cnt_out >= batch_size_out:
                    # resample a batch_size and a new path
                    batch_size_out = np.random.randint(batch_size_min, batch_size_max + 1)
                    last_client_route = self.get_path(n_circuits, w_out)
                    cnt_out = 0
            else:
                # incoming direction
                route_info.append(last_server_route)
                debug_route_used[last_server_route] = 1
                cnt_in += 1
                if cnt_in >= batch_size_in:
                    # resample a batch_size and a new path
                    batch_size_in = np.random.randint(batch_size_min, batch_size_max + 1)
                    last_server_route = self.get_path(n_circuits, w_in)
                    cnt_in = 0

        trace_with_path = self.simulate_bwr_time(trace, latency_info, route_info)  # L x 3: time, direction, circuit_id
        return trace_with_path

    def dump_trace(self, original_path: Union[str, os.PathLike], defended_trace: np.ndarray) -> None:
        fname = original_path.split('/')[-1]
        dump_dir = os.path.join(self.output_dir, fname)
        with open(dump_dir, 'w') as f:
            for i in range(self.config.n_circuits):
                sub_arr = defended_trace[defended_trace[:, 2] == i][:, :2]
                if len(sub_arr) == 0:
                    continue
                sub_arr[:, 0] -= sub_arr[0, 0]
                np.savetxt(f, sub_arr, fmt='%.6f\t%d')
                if i < self.config.n_circuits - 1:
                    f.write('\n')

    @staticmethod
    def simulate_bwr_time(trace, latency_info, route_info):
        delta = 0.0001  # Delta time to introduce as the time between two cells are sent from the end side
        last_direction = 1
        last_time = 0
        delay = 0
        time_last_incoming = 0
        new_trace = []

        for i in range(0, len(trace)):
            time, direction, route_path = trace[i][0], trace[i][1], route_info[i]
            # Get the latency for this route
            chosen_latency = np.random.choice(latency_info[(route_path % len(latency_info))])
            if direction != last_direction:
                # Calculate the RTT/2 (latency) request/response, from the time the out cell is sent till the
                # correspongin incell arrives
                delay = (time - last_time) / 2

            # original_time - delay = time when the in-cell (measured at client) is on exit original_time - delay +
            # chosen_latency = time when the in-cell is at client after travelling across one of the m circuits.
            # time_last_incomming = time of the las in-cell before the out-cell was on client, it is used + delta to set
            # the time of the outgoing cell
            new_packet = None
            if direction == -1:
                new_packet = [time - delay + chosen_latency, direction, route_path]
            elif direction == 1 and last_direction == -1:
                # If is the first out in the burst, it referes to the last icomming time
                new_packet = [time_last_incoming + delta, direction, route_path]
            elif direction == 1 and last_direction == 1:
                # If we are in an out burst, refers to the last out
                new_packet = [last_time + delta, direction, route_path]
            assert new_packet
            new_trace.append(new_packet)
            time_last_incoming = time - delay + chosen_latency
            last_time = time
            last_direction = direction
        new_trace = np.array(new_trace)
        trace_with_path = new_trace[new_trace[:, 0].argsort()]
        trace_with_path[:, 0] -= trace_with_path[0, 0]
        return trace_with_path

    @staticmethod
    def get_circuit_latencies(fdir):
        with open(fdir, 'r') as f:
            lines = f.readlines()
        # Get the multiple circuits of the selected client:
        multipath_latencies = []
        client_ids = []
        for laten in lines:
            laten = laten.split('\n')[0]
            clientid = int(laten.split(' ')[0])
            client_ids.append(clientid)
            multipath_latencies.append(laten.split(' ')[2].split(','))

        res = []
        # str -> float
        for row in multipath_latencies:
            res.append(list(map(eval, row)))
        assert len(res) == len(client_ids)
        return res, client_ids

    @staticmethod
    def get_weights(n_circuits: int, alpha: float = 1) -> (np.ndarray, np.ndarray):
        # sample a weight vector from dirichlet for client and server
        # by default the hyper param for dirichlet is [1]*n_circuits
        return np.random.dirichlet([alpha] * n_circuits), np.random.dirichlet([alpha] * n_circuits)

    @staticmethod
    def get_path(n_circuits: int, weight: np.ndarray) -> int:
        # get a path to send for client and server
        return np.random.choice(np.arange(n_circuits), p=weight)

    @staticmethod
    def sample_latencies(latencies: List, client_ids: List, n_circuits: int) -> List:
        # there is a bug in the original code which cause the random_client will never get the maximum
        random_client_id = np.random.randint(min(client_ids), max(client_ids) + 1)
        random_client_id = 13
        res = []
        for latency, client_id in zip(latencies, client_ids):
            if client_id == random_client_id:
                res.append(latency)
        # # I only need n circuits, it works when n <  number of circuits in latency file (I had max 6)
        return res[:n_circuits]
