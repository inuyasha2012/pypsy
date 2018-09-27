# coding=utf-8
from __future__ import print_function, division, unicode_literals
from psy import Mirt, data


def test_mirt():
    score = data['lsat.dat']
    Mirt(scores=score, dim_size=2).em()
