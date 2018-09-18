# coding=utf-8
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Irt2PL


def test():
    score = np.loadtxt('data/lsat.csv', delimiter=",")
    Irt2PL(scores=score).em()
