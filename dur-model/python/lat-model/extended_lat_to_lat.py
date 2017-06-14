#! /usr/bin/env python
from __future__ import print_function
from os.path import basename

import argparse
import logging
import numpy as np
import sys

# Logging settings
log = logging.getLogger(basename(__file__))
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--acoustic', action='store_true', dest='acoustic',
    help='Modify acoustic score')
parser.add_argument(
    '--normalize', action='store_true', dest='normalize',
    help='Normalize extra score by dividing by the number of phones')
parser.add_argument(
    'scale1', metavar='scale', help='Extra score scale', type=float)
parser.add_argument(
    'scale2', metavar='phone_penalty', help='Phone penalty', type=float)

args = parser.parse_args()

if args.acoustic:
    log.info('Modifying acoustic scores')

scale1 = args.scale1
scale2 = args.scale2

log.info('%s %s', scale1, scale2)

log_scale2 = np.log(scale2)

for l in sys.stdin:
    ss = l.split()

    if len(ss) == 4:
        x = ss[3].split(',')
        if args.normalize:
            normalizer = 1.0/float(x[3])
        else:
            normalizer = 1.0

        if args.acoustic:
            new_am_score = float(x[1]) - scale1 * (
                normalizer * -float(x[2]) - log_scale2 * float(x[3]))
            print('{} {}  {}  {},{:f},{}'.format(
                ss[0], ss[1], ss[2], x[0], new_am_score,  x[-1]))
        else:
            new_graph_score = float(x[0]) - scale1 * (
                normalizer * -float(x[2]) - log_scale2 * float(x[3]))
            print('{} {}  {}  {:f},{},{}'.format(
                ss[0], ss[1], ss[2], new_graph_score, x[1], x[-1]))
    else:
        print(l, end='')
