# coding=utf-8
from __future__ import division, print_function, unicode_literals
from psy import Grm, data


def test_grm():
    scores = data['lsat.dat']
    grm = Grm(scores=scores)
    grm.em()
