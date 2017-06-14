#!/usr/bin/env python
from __future__ import print_function

import argparse
import codecs
import logging
import numpy as np
import sklearn.utils
import sys

from collections import OrderedDict
from os.path import basename
from pylearn2.datasets import vector_spaces_dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.utils import serial

from durmodel_utils import get_features, write_features

# Logging settings
log = logging.getLogger(basename(__file__))
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# Script argument parser settings
parser = argparse.ArgumentParser()
parser.add_argument(
    '--read-features', action="store", dest="read_features_filename",
    help="Read features from file")
parser.add_argument(
    '--write-features', action="store", dest="write_features_filename",
    help="Read features from file")
parser.add_argument(
    '--devpercent', action="store", dest="devpercent", default=5, type=float,
    help="Use the bottom N% from each source file as dev data")

parser.add_argument(
    '--save', action="store", dest="save_filename",
    help="Save data to file")
parser.add_argument(
    '--savedev', action="store", dest="savedev_filename",
    help="Save development data to file")
parser.add_argument(
    '--shuffle', action="store_true",
    help="Shuffle data before train/dev split")
parser.add_argument(
    'data', metavar='data-file features-file', nargs='+',
    help='Pairs of data and features files')


def get_source_features(path, encoding='utf-8'):
    log.debug('-> loading source features from: %s', path)
    with codecs.open(path, encoding=encoding) as file_features:
        for i, feature in enumerate(file_features):
            yield feature.strip(), i


if __name__ == '__main__':
    args = parser.parse_args()

    if divmod(len(args.data), 2)[1] != 0:
        log.error('Odd number of files given?')
        sys.exit(1)

    sources = [
        (args.data[i*2], args.data[i*2+1]) for i in range(len(args.data) / 2)]

    if args.read_features_filename:
        log.info('.. Reading features from %s', args.read_features_filename)
        feature_dict = OrderedDict(get_features(args.read_features_filename))
    else:
        log.info('.. Accumulating features from %s to %s',
                 sources[0][1], sources[-1][1])
        feature_dict = OrderedDict()
        for f in [s[1] for s in sources]:
            with codecs.open(f, encoding='utf-8') as file_feature:
                for i, feature in enumerate(file_feature):
                    feature_dict.setdefault(feature.strip(), len(feature_dict))

    if args.write_features_filename:
        log.info('.. Writing features to %s', args.write_features_filename)
        write_features(args.write_features_filename, feature_dict)

    # Initialize all matrices
    X = np.zeros((0, len(feature_dict)), dtype=np.float16)
    speakers = np.zeros((0, 1), dtype=np.int)
    y = np.zeros((0, 1), dtype=np.float)

    for source in sources:
        log.info('.. Reading from data file %s with features from %s',
                 source[0], source[1])

        source_feature_dict = OrderedDict(get_source_features(source[1]))

        dataset = serial.load(source[0])
        num_examples = dataset.get_num_examples()

        features_data = dataset.get_data()[0]
        speakers_data = dataset.get_data()[1]
        dur_data = dataset.get_data()[2]

        start = X.shape[0]
        X.resize((start + num_examples, len(feature_dict)))

        for fname, fvalue in source_feature_dict.iteritems():
            X[start:, feature_dict[fname]] = features_data[:, fvalue]

        speakers.resize((start + num_examples, 1))
        speakers[start:, :] = speakers_data[:, :]

        y.resize((start + num_examples, 1))
        y[start:, :] = dur_data[:, :]

    num_speakers = dataset.get_data_specs()[0].components[1].max_labels

    if args.shuffle:
        log.info('.. Shuffling data')
        X, speakers, y = sklearn.utils.shuffle(X, speakers, y)
        log.info('.. Done shuffling')

    num_dev = int(X.shape[0] * 0.01 * args.devpercent)
    log.info('.. Using %f percent of data (%d examples) as development data',
             args.devpercent, num_dev)

    if num_dev > 0:
        X_dev = X[-num_dev:]
        X = X[:-num_dev]
        speakers_dev = speakers[-num_dev:]
        speakers = speakers[:-num_dev]
        y_dev = y[-num_dev:]
        y = y[:-num_dev]

    space = CompositeSpace([VectorSpace(dim=len(feature_dict)),
                            IndexSpace(dim=1, max_labels=num_speakers),
                            VectorSpace(dim=1)])
    source = ('features', 'speakers', 'targets')
    final_dataset = vector_spaces_dataset.VectorSpacesDataset(
        data=(X, speakers, y),
        data_specs=(space, source))

    if args.save_filename:
        log.info('.. Writing data to %s', args.save_filename)
        serial.save(args.save_filename, final_dataset)

    if args.savedev_filename:
        log.info('.. Writing dev data to %s', args.savedev_filename)
        final_dataset_dev = vector_spaces_dataset.VectorSpacesDataset(
            data=(X_dev, speakers_dev, y_dev),
            data_specs=(space, source))

        serial.save(args.savedev_filename, final_dataset_dev)
