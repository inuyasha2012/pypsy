# coding=utf-8
from __future__ import division, print_function, unicode_literals
from psy import Grm
import numpy as np


def test_grm():
    scores = np.loadtxt('data/lsat.csv', delimiter=',')
    grm = Grm(scores=scores)
    grm.em()
