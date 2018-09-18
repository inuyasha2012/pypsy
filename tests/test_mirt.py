# coding=utf-8
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Mirt2PL


def test_mirt():
    score = np.loadtxt('data/lsat.csv', delimiter=",")
    Mirt2PL(scores=score, dim_size=2).em()
