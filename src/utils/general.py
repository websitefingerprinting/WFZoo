import os
import random
from datetime import datetime
from functools import wraps
from time import strftime
from time import time
from typing import Tuple, Union, Callable, Optional, Any

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_seed(func):
    def wrapper(*args, **kwargs):
        np.random.seed(datetime.now().microsecond)
        return func(*args, **kwargs)

    return wrapper


def timeit(f: Callable):
    @wraps(f)
    def wrap(*args: Optional[Any], **kw: Optional[Any]) -> float:
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def parse_trace(fdir: str, sanity_check: bool = True) -> np.ndarray:
    """
    Parse a trace file based on our predefined format
    """
    trace = pd.read_csv(fdir, delimiter="\t", header=None)
    trace = np.array(trace)

    if sanity_check:
        # it is possible the trace has a long tail
        # if there is a time gap between two bursts larger than CUT_OFF_THRESHOULD
        # We cut off the trace here sicne it could be a long timeout or
        # maybe the loading is already finished
        # Set a very conservative value
        CUT_OFF_THRESHOLD = 15
        start, end = 0, len(trace)
        ipt_burst = np.diff(trace[:, 0])
        ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]

        if len(ipt_outlier_inds) > 0:
            outlier_ind_first = ipt_outlier_inds[0]
            if outlier_ind_first < 50:
                start = outlier_ind_first + 1
            outlier_ind_last = ipt_outlier_inds[-1]
            if outlier_ind_last > 50:
                end = outlier_ind_last + 1
        trace = trace[start:end].copy()

        # remove the first few lines that are incoming packets
        start = -1
        for time, size in trace:
            start += 1
            if size > 0:
                break

        trace = trace[start:].copy()
        trace[:, 0] -= trace[0, 0]
        assert trace[0, 0] == 0
    return trace


def feature_transform(sample: np.ndarray, feature_type: str, seq_length: int) -> np.ndarray:
    """
    Transform a raw sample to the specific feature space.
    :return a numpy array of shape (1 or 2, seq_length)
    """
    if feature_type == 'df':
        feat = np.sign(sample[:, 1])

    elif feature_type == 'tiktok':
        feat = sample[:, 0] * np.sign(sample[:, 1])

    elif feature_type == 'tam':
        max_load_time = 80  # s
        time_window = 0.044  # s

        sample = sample[sample[:, 0] < max_load_time]
        num_bins = int(sample[-1, 0] / time_window) + 1

        outgoing = sample[np.sign(sample[:, 1]) > 0]
        incoming = sample[np.sign(sample[:, 1]) < 0]

        cnt_outgoing, _ = np.histogram(outgoing[:, 0], bins=num_bins)
        cnt_incoming, _ = np.histogram(incoming[:, 0], bins=num_bins)
        # merge to 2d feature
        feat = np.stack((cnt_outgoing, cnt_incoming), axis=1)
        assert feat.flatten().sum() == len(sample), \
            "Sum of feature ({}) is not equal to the length of the trace ({}). BUG?".format(
                feat.flatten().sum(), len(sample))

    elif feature_type == 'burst':
        sample = sample[:, 1]
        # Create a mask for consecutive elements that are the same
        mask = np.where(np.sign(sample[:-1]) != np.sign(sample[1:]))[0] + 1
        mask = np.concatenate((mask, [len(sample)]))  # add the last index
        # Count the number of elements between sign changes
        feat = np.diff(mask, prepend=0)
        assert sum(feat) == len(sample), \
            "Sum of burst lengths ({}) is not equal to the length of the trace ({}). BUG?".format(sum(feat),
                                                                                                  len(sample))
    else:
        raise NotImplementedError("Feature type {} is not implemented.".format(feature_type))

    # make sure 2d
    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    # pad to seq_length
    if len(feat) < seq_length:
        pad = np.zeros((seq_length - len(feat), feat.shape[1]))
        feat = np.concatenate((feat, pad))
    feat = feat[:seq_length, :]
    return np.transpose(feat, (1, 0))


def get_flist_label(data_path: Union[str, os.PathLike], mon_cls: int, mon_inst: int, unmon_inst: int,
                    suffix: str = '.cell') \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a list of file paths and corresponding labels.
    :param data_path: the path to the data directory
    :param mon_cls: number of monitored classes
    :param mon_inst: number of monitored instances per class
    :param unmon_inst: number of unmonitored instances
    :param suffix: file suffix
    :return: a list of file paths and a list of corresponding labels
    """
    flist = []
    labels = []
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            pth = os.path.join(data_path, '{}-{}{}'.format(cls, inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(cls)
    for inst in range(unmon_inst):
        pth = os.path.join(data_path, '{}{}'.format(inst, suffix))
        if os.path.exists(pth):
            flist.append(pth)
            labels.append(mon_cls)

    assert len(flist) > 0, "No files found in {}!".format(data_path)
    return np.array(flist), np.array(labels)


def init_directories(output_parent_dir: Union[str, os.PathLike], defense_name: str) -> str:
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)

    # Define output directory
    timestamp = strftime('%m%d_%H%M%S')
    output_dir = os.path.join(output_parent_dir, defense_name + '_' + timestamp)
    os.makedirs(output_dir)
    return output_dir
