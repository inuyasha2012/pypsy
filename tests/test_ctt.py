# coding=utf-8
from __future__ import print_function, division, unicode_literals
from psy.ctt import BivariateCtt
from psy import data


def test_ctt():
    score = data['lsat.dat']
    ctt = BivariateCtt(score)
    ctt.get_alpha_reliability()
    ctt.get_composite_reliability()
