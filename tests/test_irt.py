# coding=utf-8
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Irt


class TestIrt(object):

    def test_probit(self):
        score = np.loadtxt('data/lsat.csv', delimiter=",")
        model = Irt(scores=score, link='probit')
        model.fit()

    def test_logit(self):
        score = np.loadtxt('data/lsat.csv', delimiter=",")
        model = Irt(scores=score, link='logit')
        model.fit()
