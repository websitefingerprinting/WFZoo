import argparse
import os
from typing import Union, List

import numpy as np

from defenses.base import Defense
from defenses.config import RegulatorConfig
from utils.general import parse_trace, set_random_seed


class RegulatorDefense(Defense):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = RegulatorConfig(args)
        self.config.load_config()

    @set_random_seed
    def simulate(self, data_path: Union[str, os.PathLike], dump: bool = True) -> np.ndarray:
        trace = parse_trace(data_path)
        # get download and upload separately
        download_packets = trace[trace[:, 1] < 0][:, 0]
        upload_packets = trace[trace[:, 1] > 0][:, 0]

        # get defended traces
        padded_download_time, padded_download_dir, padding_budget, padding_packets = self.regulator_download(
            download_packets)
        padded_upload_time, padded_upload_dir, delayed_pkts_num = self.regulator_upload_full(padded_download_time,
                                                                                             upload_packets)

        padded_download = np.stack((padded_download_time, padded_download_dir), axis=-1)
        padded_upload = np.stack((padded_upload_time, padded_upload_dir), axis=-1)
        defended_trace = np.concatenate((padded_download, padded_upload), axis=0)
        # sort and make sure outgoing packets are in the first
        inds = np.lexsort((-defended_trace[:, 1], defended_trace[:, 0]))
        defended_trace = defended_trace[inds]

        if dump:
            self.dump_trace(data_path, defended_trace)
        return defended_trace

    def regulator_download(self, target_trace: Union[np.ndarray, List]) -> (List, List, int, int):
        initial_rate = self.config.initial_rate
        decay_rate = self.config.decay_rate
        max_padding_budget = self.config.max_padding_budget
        padding_budget = np.random.randint(0, max_padding_budget)
        burst_threshold = self.config.surge_threshold

        output_trace = []  # record timestamps
        output_trace_dir = []  # record dummy flags

        position = 10

        # send packets at a constant rate initially (to construct circuit)
        download_start = target_trace[position]
        added_packets = int(download_start * 10)
        for i in range(added_packets):
            pkt_time = i * .1
            output_trace.append(pkt_time)
            output_trace_dir.append(-self.DUMMY)

        output_trace.append(target_trace[position])
        output_trace_dir.append(-1)

        current_time = download_start
        burst_time = target_trace[position]

        padding_packets = 0
        position = 1

        while True:
            # calculate target rate
            target_rate = initial_rate * (decay_rate ** (current_time - burst_time))

            if target_rate < 1:
                target_rate = 1

            # if the original trace has been completely sent
            if position == len(target_trace):
                break

            # find number of real packets waiting to be sent
            queue_length = 0
            for c in range(position, len(target_trace)):
                if target_trace[c] < current_time:
                    queue_length += 1
                else:
                    break

            # if waiting packets exceeds treshold, then begin a new burst
            if queue_length > (burst_threshold * target_rate):
                burst_time = current_time

            # calculate gap
            gap = 1 / float(target_rate)
            current_time += gap

            if queue_length == 0 and padding_packets >= padding_budget:
                # no packets waiting and padding budget reached
                continue
            elif queue_length == 0 and padding_packets < padding_budget:
                # no packets waiting, but padding budget not reached
                output_trace.append(current_time)
                output_trace_dir.append(-self.DUMMY)
                padding_packets += 1
            else:
                # real packet to send
                output_trace.append(current_time)
                output_trace_dir.append(-1)
                position += 1

        assert len(output_trace) >= len(target_trace)
        assert len(output_trace) == len(output_trace_dir)
        assert len(np.where(np.array(output_trace_dir) == -1)[0]) == len(target_trace)

        return output_trace, output_trace_dir, padding_budget, padding_packets

    def regulator_upload_full(self, download_trace: Union[np.ndarray, List], upload_trace: Union[np.ndarray, List]) \
            -> (List, List, int):
        upload_ratio = self.config.upload_ratio
        delay_cap = self.config.delay_cap

        delayed_pkts_num = 0

        # send one upload packet for every $upload_ratio download packets
        upload_size = int(len(download_trace) / upload_ratio)
        output_trace = list(np.random.choice(download_trace, upload_size))

        # send at constant rate at first
        position = 10
        download_start = download_trace[position]
        added_packets = int(download_start * 5)
        for i in range(added_packets):
            pkt_time = i * .2
            output_trace.append(pkt_time)

        # first assume all the pkts are dummy and later mark the real ones in the matching procedure below
        output_trace_dir = [self.DUMMY] * len(output_trace)

        # assign each packet to the next scheduled sending time in the output trace
        output_trace = sorted(output_trace)
        delay_packets = []  # different from the original design, change to 2d array (time, flag)
        packet_position = 0
        for t in upload_trace:
            found_packet = False
            for p in range(packet_position + 1, len(output_trace)):
                if output_trace[p] >= t and (output_trace[p] - t) < delay_cap:
                    packet_position = p
                    output_trace_dir[packet_position] = 1
                    found_packet = True
                    break

            # cap delay at delay_cap seconds
            if not found_packet:
                delayed_pkts_num += 1
                delay_packets.append([t + delay_cap, 1])

        output_trace = np.stack((output_trace, output_trace_dir), axis=-1)
        if len(delay_packets) > 0:
            output_trace = np.concatenate((output_trace, delay_packets), axis=0)
        output_trace = output_trace[output_trace[:, 0].argsort()]

        assert len(output_trace) >= len(upload_trace)
        assert len(output_trace[output_trace[:, 1] == 1]) == len(upload_trace)
        return list(output_trace[:, 0]), list(output_trace[:, 1]), delayed_pkts_num
