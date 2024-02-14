import argparse

from utils.general import init_directories, set_random_seed
from utils.logger import init_logger


class Defense(object):
    def __init__(self, args: argparse.Namespace):
        self.DUMMY = 888
        self.args = args
        self.logger = init_logger(str(self.__class__.__name__))
        self.output_dir = init_directories(args.output_dir, str(self.__class__.__name__))
        self.logger.info("Output directory: {}".format(self.output_dir))

    # pay attention that numpy may have the same random seed for a batch of multiprocessing processes
    # https://github.com/numpy/numpy/issues/9650
    # https://stackoverflow.com/questions/67691168/how-to-generate-different-random-values-at-each-subprocess-during-a-multiprocess
    @set_random_seed
    def simulate(self, **kwargs):
        raise NotImplementedError
