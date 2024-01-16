import argparse

import torch

from attacks import DFAttack, TiktokAttack, TamAttack
from utils.general import seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser(description='WF transfer project')
    parser.add_argument('--attack', choices=['df', 'tiktok', 'tam'], default='df', help='choose the attack')

    # paths and file config
    parser.add_argument('--data-path', type=str, help="data path")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--suffix', type=str, default='.cell', help='suffix of the output file')
    parser.add_argument('--one-fold', default=False, action="store_true", help='Run one fold or ten folds')
    parser.add_argument('--mon-classes', default=100, type=int, help='Number of monitored classes')
    parser.add_argument('--mon-inst', default=100, type=int,
                        help='Number of monitored instances per class')
    parser.add_argument('--unmon-inst', default=100, type=int,
                        help='Number of unmonitored instances per class')
    parser.add_argument('--open-world', default=False, action="store_true", help='Open world or not')
    parser.add_argument('--seq-length', default=5000, type=int, help='The input trace length')

    # optimization
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr0', type=float, default=0.002, help='initial optimizer learning rate')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam',
                        help='optimizer')

    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multiple gpus')
    parser.add_argument('--amp', action='store_true', default=False, help='use mixed precision training')

    # LOG
    parser.add_argument('--verbose', action='store_true', default=False, help='print detailed performance')
    parser.add_argument('--log_itr_interval', type=int, default=100, help='log iteration interval')

    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = parse_arguments()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    seed_everything(2024)
    attack = None
    if args.attack == 'df':
        attack = DFAttack(args)
    elif args.attack == 'tiktok':
        attack = TiktokAttack(args)
    elif args.attack == 'tam':
        attack = TamAttack(args)
    else:
        raise NotImplementedError("Attack not implemented")

    attack.run(args.one_fold)
