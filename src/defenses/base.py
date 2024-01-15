import argparse

from utils.general import init_directories
from utils.logger import init_logger


class Defense(object):
    def __init__(self, args: argparse.Namespace):
        self.DUMMY = 888
        self.args = args
        self.logger = init_logger(str(self.__class__.__name__))
        self.output_dir = init_directories(args.output_dir, str(self.__class__.__name__))
        self.logger.info("Output directory: {}".format(self.output_dir))

    def simulate(self, **kwargs):
        raise NotImplementedError
