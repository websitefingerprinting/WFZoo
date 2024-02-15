import argparse
import configparser


class DefenseConfig(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config_parser = configparser.ConfigParser()
        self.config_section = self.args.config_section
        self.converters = None

    def load_config(self):
        converters = self.converters
        if converters is None:
            converters = {}

        # Read the configuration file
        self.config_parser.read(self.args.config_path)

        # Check if the specified section exists in the configuration file
        if not self.config_parser.has_section(self.config_section):
            raise ValueError(f"Section '{self.config_section}' not found in the configuration file.")

        # Get all options and their values in the specified section
        options = self.config_parser.options(self.config_section)

        # check whether you are loading the correct config file
        if set(converters.keys()).difference(options):
            raise ValueError(f"Options in the config file do not match the options in the converters.")

        # Populate self.raw_config with the options and their values
        for option in options:
            raw_value = self.config_parser.get(self.config_section, option)
            # Use the specified converter or the default str() conversion
            converter = converters.get(option, str)
            setattr(self, option, converter(raw_value))


class FrontConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'n_client': int,
            'n_server': int,
            'w_min': float,
            'w_max': float,
            'start_t': float
        }


class TamarawConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'rho_client': float,
            'rho_server': float,
            'lpad': int,
            'strategy': str
        }


class RegulatorConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {
            'initial_rate': int,
            'decay_rate': float,
            'surge_threshold': float,
            'max_padding_budget': int,
            'upload_ratio': float,
            'delay_cap': float
        }


class WtfpadConfig(DefenseConfig):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.converters = {}
        self.interpolate = True
        self.remove_tokens = True
        self.stop_on_real = True
        self.percentile = 0


if __name__ == '__main__':
    # Example usage:
    args = argparse.Namespace(config_section='heavy',
                              config_path='/Users/jgongac/WFZoo/src/defenses/config/regulator.ini')
    defense_config = RegulatorConfig(args)
    defense_config.load_config()
    print(defense_config)
