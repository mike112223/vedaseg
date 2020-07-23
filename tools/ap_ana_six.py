import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                '../../vedaseg'))

from vedaseg.assembler import assemble


def parse_args():
    parser = argparse.ArgumentParser(description='Search for threshold')
    parser.add_argument('--config', help='config file path',
                        default='/media/yuhaoye/DATA7/temp_for_upload/vedaseg'
                                '/configs/ap_ana.py')
    parser.add_argument('--checkpoint', help='model checkpoint file path',
                        default='/home/yuhaoye/tmp/epoch_135.pth')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--conf', type=float, default=-1.)
    parser.add_argument('--size', type=float, default=-1.)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    checkpoint = args.checkpoint

    if args.conf > 0:
        conf = [args.conf]
    else:
        conf = np.arange(0.1, 1.0, 0.1)

    if args.size > 0:
        size = [args.size]
    else:
        size = np.arange(100, 100, 1000)

    runner = assemble(cfg_fp, checkpoint, True, args.show, args.save)
    runner(ap_ana=True,
           conf_thresholds=conf,
           size_thresholds=size)


if __name__ == '__main__':
    main()
