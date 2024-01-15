import argparse
import configparser


class DefenseConfig(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_parser = configparser.ConfigParser()
        self.config_section = self.args.config_section

    def load_config(self, **kwargs):
        raise NotImplementedError


class FrontConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.n_client = None
        self.n_server = None
        self.w_min = None
        self.w_max = None
        self.start_t = None

    def load_config(self):
        self.config_parser.read(self.args.config_path)
        self.n_client = self.config_parser.getint(self.config_section, 'n_client')
        self.n_server = self.config_parser.getint(self.config_section, 'n_server')
        self.w_min = self.config_parser.getfloat(self.config_section, 'w_min')
        self.w_max = self.config_parser.getfloat(self.config_section, 'w_max')
        self.start_t = self.config_parser.getfloat(self.config_section, 'start_t')

        assert self.w_min < self.w_max, "w_min should be smaller than w_max"
        assert self.n_client > 1, "client_n should be larger than 1"
        assert self.n_server > 1, "server_n should be larger than 1"
        assert self.start_t >= 0, "start_t should be positive"