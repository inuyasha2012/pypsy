# coding=utf-8
from __future__ import print_function, division, unicode_literals
from psy import Irt, data


class TestIrt(object):

    def test_probit(self):
        score = data['lsat.dat']
        model = Irt(scores=score, link='probit')
        model.fit()

    def test_logit(self):
        score = data['lsat.dat']
        model = Irt(scores=score, link='logit')
        model.fit()
