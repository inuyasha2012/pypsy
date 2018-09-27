# coding=utf-8
# 探索性因子分析
from __future__ import print_function, division, unicode_literals
from psy import Factor, data


def test_fa():
    score = data['lsat.dat']
    factor = Factor(score, 5)
    factor.loadings
