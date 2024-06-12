import argparse
import os
from pathlib import Path
from typing import Union, Optional, Any

import numpy as np

from utils.general import init_directories, set_random_seed
from utils.logger import init_logger


class Defense(object):
    def __init__(self, args: argparse.Namespace):
        self.DUMMY = 888
        self.args = args
        self.logger = init_logger(str(self.__class__.__name__))
        self.output_dir = init_directories(args.output_dir,
                                           str(Path(args.data_path).stem) + '_' +
                                           str(self.__class__.__name__) + '_' +
                                           str(args.config_section)
                                           )
        self.logger.info("Output directory: {}".format(self.output_dir))

    def simulate(self, data_path: Union[str, os.PathLike], dump: bool = True, **kwargs: Optional[Any]) -> np.ndarray:
        try:
            defended_trace = self._simulate(data_path, **kwargs)
            if len(defended_trace) == 0:
                raise ValueError("Empty defended trace")
            if dump:
                self.dump_trace(data_path, defended_trace)
            return defended_trace
        except Exception as e:
            self.logger.error(f"Error in {data_path}: {e}")
            return np.array([])

    # pay attention that numpy may have the same random seed for a batch of multiprocessing processes
    # https://github.com/numpy/numpy/issues/9650
    # https://stackoverflow.com/questions/67691168/how-to-generate-different-random-values-at-each-subprocess-during-a-multiprocess
    @set_random_seed
    def _simulate(self, data_path: Union[str, os.PathLike], **kwargs):
        raise NotImplementedError

    def dump_trace(self, original_path: Union[str, os.PathLike], defended_trace: np.ndarray) -> None:
        fname = original_path.split('/')[-1]
        dump_dir = os.path.join(self.output_dir, fname)
        np.savetxt(dump_dir, defended_trace, fmt='%.6f\t%d')
