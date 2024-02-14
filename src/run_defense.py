import argparse
from multiprocessing import Pool
from typing import List, Union

import numpy as np

from defenses import FrontDefense, TamarawDefense, RegulatorDefense
from utils.general import get_flist_label, timeit


def parse_arguments():
    parser = argparse.ArgumentParser(description='WF transfer project')
    parser.add_argument('--defense', choices=['front', 'tamaraw', 'regulator'], help='choose the defense')

    # paths and file config
    parser.add_argument('--data-path', type=str, help="data path")
    # config-path
    parser.add_argument('--config-path', type=str, default=None, help="config path")
    parser.add_argument('--config-section', '-c', type=str, default='default', help="config section")
    parser.add_argument('--output-dir', type=str, default='../defense_results/',
                        help='location of model checkpoints')
    parser.add_argument('--suffix', type=str, default='.cell', help='suffix of the output file')
    parser.add_argument('--mon-classes', default=100, type=int, help='Number of monitored classes')
    parser.add_argument('--mon-inst', default=100, type=int,
                        help='Number of monitored instances per class')
    parser.add_argument('--unmon-inst', default=100, type=int,
                        help='Number of unmonitored instances per class')
    parser.add_argument('--open-world', default=False, action="store_true", help='Open world or not')

    # nworkers
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 20)')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')

    _args = parser.parse_args()
    return _args


@timeit
def parallel_simulate(_flist: Union[List[str], np.ndarray]):
    with Pool(args.workers) as p:
        p.map(defense.simulate, _flist)


if __name__ == '__main__':
    args = parse_arguments()
    if args.defense == 'front':
        defense = FrontDefense(args)
    elif args.defense == 'tamaraw':
        defense = TamarawDefense(args)
    elif args.defense == 'regulator':
        defense = RegulatorDefense(args)
    else:
        raise NotImplementedError("Attack not implemented")

    if not args.open_world:
        args.unmon_inst = 0
    flist, _ = get_flist_label(args.data_path, args.mon_classes, args.mon_inst, args.unmon_inst, args.suffix)
    defense.logger.info("Simulating {} files using {}...".format(len(flist), args.defense))
    parallel_simulate(flist)
