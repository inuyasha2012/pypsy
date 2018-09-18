# coding=utf-8
from __future__ import print_function, division, unicode_literals
from psy.ctt import BivariateCtt
import numpy as np


def test_ctt():
    score = np.loadtxt('data/lsat.csv', delimiter=",")
    ctt = BivariateCtt(score)
    ctt.get_alpha_reliability()
    ctt.get_composite_reliability()