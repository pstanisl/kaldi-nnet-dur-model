#!/usr/bin/env python
'''
Created on Sep 9, 2013

@author: tanel
'''
from __future__ import print_function

__author__ = 'tanel'

import sys


class Arc:
    def __init__(self, id, start, end, word_id, score1, score2, phone_ids):
        self.id = id
        self.start = start
        self.end = end
        self.word_id = word_id
        self.score1 = score1
        self.score2 = score2
        self.additional_score1 = 0.0
        self.additional_score2 = 0.0
        self.phone_ids = phone_ids
        self.start_frame = -1
        self.end_frame = -1


class Lattice:

    def __init__(self, name, arcs, final_states):
        self.name = name
        self.arcs = arcs
        self.final_states = final_states
        self.prev_arcs = {}
        self.next_arcs = {}
        for arc in arcs:
            self.next_arcs.setdefault(arc.start, []).append(arc)
            self.prev_arcs.setdefault(arc.end, []).append(arc)

    def get_previous_arcs(self, arc):
        return self.prev_arcs.get(arc.start, [])

    def get_next_arcs(self, arc):
        return self.next_arcs.get(arc.end, [])

    def to_lat(self, output):
        print(self.name, file=output)
        for arc in self.arcs:
            print('{:d} {:d}  {:d}  {:f},{:f},{}'.format(
                arc.start, arc.end, arc.word_id, arc.score1, arc.score2,
                '_'.join([str(i) for i in arc.phone_ids])), file=output)
        for final_state in self.final_states:
            print(final_state, file=output)

        print(file=output)

    def to_extended_lat(self, output):
        print(self.name, file=output)
        for arc in self.arcs:
            print('{:d} {:d}  {:d}  {:f},{:f},{:f},{:f},{}'.format(
                arc.start, arc.end, arc.word_id, arc.score1, arc.score2,
                arc.additional_score1, arc.additional_score2,
                '_'.join([str(i) for i in arc.phone_ids])), file=output)
        for final_state in self.final_states:
            print(final_state, file=output)

        print(file=output)


def set_next_arc_start_frames(arc, arcs_by_start, start_frame):
    arc.start_frame = start_frame
    arc.end_frame = start_frame + len(arc.phone_ids)
    for next_arc in arcs_by_start.get(arc.end, []):
        set_next_arc_start_frames(next_arc, arcs_by_start, arc.end_frame)


def parse_aligned_lattice(lines):
    line = ''
    name = ''
    try:
        name = lines[0].strip()
        arcs = []
        final_states = []
        i = 0
        arcs_by_start = {}
        for line in lines[1:]:
            ss = line.split()
            if len(ss) == 4:
                start = int(ss[0])
                end = int(ss[1])
                word_id = int(ss[2])
                weight_parts = ss[3].split(',')
                score1 = float(weight_parts[0])
                score2 = float(weight_parts[1])
                if len(weight_parts[2]) > 0:
                    frames = [int(f) for f in weight_parts[2].split('_')]
                else:
                    frames = []
                arc = Arc(i, start, end, word_id, score1, score2, frames)
                if start == 0:
                    arc.start_frame = 0
                arcs.append(arc)
                arcs_by_start.setdefault(arc.start, []).append(arc)

                i += 1
            else:
                final_states.append(int(ss[0]))

        # for arc in arcs_by_start[0]:
        #     set_next_arc_start_frames(arc, arcs_by_start, 0)

        lat = Lattice(name, arcs, final_states)
        return lat
    except:
        e = sys.exc_info()[0]
        raise Exception(
            'Failed to process lattice {}: error at line {} ({})'.format(
                name, line, e))
