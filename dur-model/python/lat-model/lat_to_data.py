#!/usr/bin/env python
from __future__ import print_function

import argparse
import codecs
import gzip
import logging
import numpy as np
import sys

from collections import OrderedDict
from os.path import basename

from pylearn2.utils import serial
from pylearn2.datasets import vector_spaces_dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.sandbox.rnn.space import SequenceDataSpace

from durmodel_utils import read_transitions

import lattice
import durmodel_utils

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
    '--encoding', action='store', dest='encoding',
    help='encoding of the loaded files', default='utf-8')
parser.add_argument(
    '--read-features', action="store", dest="read_features_filename",
    help="Read features from file")
parser.add_argument(
    '--write-features', action="store", dest="write_features_filename",
    help="Read features from file")
parser.add_argument(
    '--save', action="store", dest="save_filename", help="Save data to file")
parser.add_argument(
    '--language', action='store', dest='language', help="Language of the data",
    default="ESTONIAN")
parser.add_argument(
    '--stress', action='store', dest='stress_dict_filename',
    help="Stress dictionary")
parser.add_argument(
    '--left-context', action='store', dest='left_context',
    help="Left context length", default=2, type=int)
parser.add_argument(
    '--right-context', action='store', dest='right_context',
    help="Left context length", default=2, type=int)
parser.add_argument(
    '--no-duration-feature', action='store_true', dest='no_use_duration',
    help="Don't Use duration features")
parser.add_argument(
    '--utt2spk', action='store', dest='utt2spk',
    help="Use the mapping in the given file to add speaker ID to each sample")
parser.add_argument(
    '--sequences', action='store_true',
    help="Create a dataset of sequences, with a delay equal to --right-context argument")

parser.add_argument(
    'transitions', metavar='transitions.txt',
    help='Transition model, produced with show-transitions')
parser.add_argument(
    'nonsilence', metavar='nonsilence.txt', help='Nonsilence phonemes')
parser.add_argument(
    'words', metavar='words.txt', help='words.txt file')
parser.add_argument(
    'train_lattice', metavar='ali-lat.txt', help='Aligned phone lattice')


def create_matrices(feature_dict, full_features_and_durs):
    num_items = len(full_features_and_durs)

    context_matrix = np.zeros((num_items, len(feature_dict)), dtype=np.float16)
    log.info('.. Created context matrix of shape %s and size %d',
             context_matrix.shape, context_matrix.size)

    speaker_vector = np.zeros((num_items, 1), dtype=np.int)
    log.info('.. Created speaker matrix of shape %s and size %d',
             speaker_vector.shape, speaker_vector.size)

    y = np.zeros((num_items, 1), dtype=np.int)
    log.info('.. Created outcome matrix of shape %s and size %d',
             y.shape, y.size)

    for i, (_features, _speaker_id, dur) in enumerate(full_features_and_durs):
        for feature_name, value in _features:
            feature_id = feature_dict.get(feature_name, -1)
            if feature_id >= 0:
                context_matrix[i, feature_id] = value
        speaker_vector[i, 0] = _speaker_id
        y[i] = dur

    return context_matrix, speaker_vector, y


def get_X_raw_utt(utt_features_and_durs, feature_dict):
    X_raw_utt = np.zeros(
        (len(utt_features_and_durs), len(feature_dict)),  dtype=np.float16)
    prev_duration = 5

    for i, (_features, _dur) in enumerate(utt_features_and_durs):
        X_raw_utt[i, 0] = durmodel_utils.dur_function(prev_duration)

        for feature_name, value in _features:
            feature_id = feature_dict.get(feature_name, -1)

            if feature_id >= 0:
                X_raw_utt[i, feature_id] = value

        prev_duration = _dur

    return X_raw_utt


def get_y_raw_utt(utt_features_and_durs, nonsilence_phonemes):
    y_raw_utt = np.zeros((len(utt_features_and_durs), 2), dtype=np.int16)
    y_raw_utt[:, 0] = np.array([d for _, d in utt_features_and_durs])

    for i, (_features, _dur) in enumerate(utt_features_and_durs):
        features_set = set([f[0] for f in _features])
        nonsilence_set = set(nonsilence_phonemes)

        if len(nonsilence_set.intersection(features_set)):
            y_raw_utt[i, 1] = 1

    return y_raw_utt


def create_raw_matrices(feature_dict, features_and_dur, nonsilence_phonemes):
    X_raw = []
    y_raw = []

    for utt_features_and_durs in features_and_dur:
        X_raw.append(get_X_raw_utt(utt_features_and_durs, feature_dict))
        y_raw.append(get_y_raw_utt(utt_features_and_durs, nonsilence_phonemes))

    return np.asarray(X_raw, y_raw)


def get_features_and_durs(train_lattice, sequences):
    num_sentences_read = 0
    sentence_lines = []

    log.debug('Opening train lattice: %s', train_lattice)

    for l in gzip.open(train_lattice):
        if len(l.strip()) > 0:
            sentence_lines.append(l)
        elif len(sentence_lines) > 0:
            try:
                features_and_durs = []
                lat = lattice.parse_aligned_lattice(sentence_lines)

                log.info('Processing lattice %s', lat.name)

                for arc in lat.arcs:
                    features_and_dur_seq = durmodel_utils.make_local(
                        arc.start_frame,
                        arc.word_id,
                        arc.phone_ids,
                        transitions,
                        word_list,
                        nonsilence_phonemes,
                        language=args.language,
                        stress_dict=stress_dict)

                    if sequences:
                        features_and_durs.extend(features_and_dur_seq)
                    else:
                        features_and_durs.append(features_and_dur_seq)

                if sequences:
                    yield features_and_durs
                else:
                    utt_full_features_and_durs = durmodel_utils.make_linear(
                        features_and_durs,
                        nonsilence_phonemes,
                        utt2spkid[lat.name] if utt2spkid else 0)

                    yield utt_full_features_and_durs
            except IOError as e:
                log.error('I/O error({0}): {1} -- {2} when processing lattice {3}'.format(
                    e.errno, e.strerror, e.message,  lat.name))
            except ValueError as e:
                log.error('ValueError({0}): {1} -- {2} when processing lattice {3}'.format(
                    0, '', e.message,  lat.name))
            except Exception as e:
                log.error('Exception({0}): {1} -- {2}'.format(
                    e.errno, e.strerror, e.message))

            num_sentences_read += 1
            sentence_lines = []

    log.info('Read alignments for %d utterances', num_sentences_read)


def get_nonsilence_phonemes(path, encoding='utf-8'):
    log.debug('-> loading from: %s', path)
    with codecs.open(path, encoding=encoding) as file_nonsilence:
        for line in file_nonsilence:
            yield line.strip().partition('_')[0]


def get_utt2spkid(path, encoding='utf-8'):
    if path:
        return None, {}

    utt2spkid = {}
    speaker_ids = {}

    log.debug('-> loading from: %s', path)

    with codecs.open(path, encoding=encoding) as file_utt2spk:
        for line in file_utt2spk:
            ss = line.split()
            utt2spkid[ss[0]] = speaker_ids.setdefault(ss[1], len(speaker_ids))

    return utt2spkid, speaker_ids


def get_words(path, encoding='utf-8'):
    log.debug('-> loading from: %s', path)
    with codecs.open(path, encoding=encoding) as file_words:
        for line in file_words:
            yield line.split()[0]


if __name__ == '__main__':
    args = parser.parse_args()
    durmodel_utils.LEFT_CONTEXT = args.left_context
    durmodel_utils.RIGHT_CONTEXT = args.right_context

    transitions = read_transitions(args.transitions)

    log.debug('transitions[%d] = %s', len(transitions) - 2, transitions[-2])
    log.debug('transitions[%d] = %s', len(transitions) - 1, transitions[-1])
    log.debug('read features filename: %s', args.read_features_filename)
    log.debug('sequences: %s', args.sequences)
    log.debug('write features filename: %s', args.write_features_filename)

    log.info('Reading non-silence phonemes')
    nonsilence_phonemes = set(get_nonsilence_phonemes(args.nonsilence))
    log.info(' -> nonsilence_phonemes: %d', len(nonsilence_phonemes))

    log.info('Reading words.txt')
    word_list = list(get_words(args.words, encoding=args.encoding))
    log.info(' -> # of words: %s', len(word_list))

    stress_dict = None
    if args.stress_dict_filename:
        log.info('Reading stress dictionary')
        stress_dict = durmodel_utils.load_stress_dict(
            args.stress_dict_filename)
        log.info('stress dict: %s', len(stress_dict))

    utt2spkid, speaker_ids = get_utt2spkid(args.utt2spk)

    log.info('Processing alignments...')

    gen_fearures = get_features_and_durs(args.train_lattice, args.sequences)

    if args.sequences:
        all_features_and_durs = list(gen_fearures)

        if args.read_features_filename:
            log.info('.. Reading features from %s',
                     args.read_features_filename)
            feature_dict = OrderedDict(
                durmodel_utils.get_features(args.read_features_filename))
        else:
            log.info('.. Reading features from all_features_and_durs')
            feature_dict = OrderedDict()
            feature_dict['prev_dur'] = 0
            for _features, _speaker_id, d in all_features_and_durs:
                for f in _features:
                    feature_name = f[0]
                    feature_dict.setdefault(feature_name, len(feature_dict))

        if args.write_features_filename:
            log.info('.. Writing features to %s', args.write_features_filename)
            durmodel_utils.write_features(
                args.write_features_filename, feature_dict)

        X, y = create_raw_matrices(
            feature_dict, all_features_and_durs, nonsilence_phonemes)

        source = ('features', 'targets')
        space = CompositeSpace([
            SequenceDataSpace(VectorSpace(dim=len(feature_dict))),
            SequenceDataSpace(VectorSpace(dim=2))
        ])

        from durmodel_elements import DurationSequencesDataset

        dataset = DurationSequencesDataset(
            data=(X, y),
            data_specs=(space, source))
    else:
        full_features_and_durs = reduce(
            lambda prev, curr: prev.extend(curr) or prev, gen_fearures)

        if args.read_features_filename:
            log.info('.. Reading features from %s',
                     args.read_features_filename)
            feature_dict = OrderedDict(
                durmodel_utils.get_features(args.read_features_filename))
        else:
            log.info('.. Reading features from full_features_and_durs')
            feature_dict = OrderedDict()
            for (_features, _speaker_id, d) in full_features_and_durs:
                for f in _features:
                    feature_name = f[0]
                    feature_dict.setdefault(feature_name, len(feature_dict))

        if args.write_features_filename:
            log.info('.. Writing features to %s', args.write_features_filename)
            durmodel_utils.write_features(
                args.write_features_filename, feature_dict)

        matrices = create_matrices(feature_dict, full_features_and_durs)

        space = CompositeSpace([VectorSpace(dim=len(feature_dict)),
                                IndexSpace(dim=1, max_labels=len(speaker_ids)),
                                VectorSpace(dim=1)])
        source = ('features', 'speakers', 'targets')
        dataset = vector_spaces_dataset.VectorSpacesDataset(
            data=matrices,
            data_specs=(space, source)
        )

    if args.save_filename:
        log.info('.. Writing data to %s', args.save_filename)
        serial.save(args.save_filename, dataset)
