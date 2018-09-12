# coding=utf-8
# 经典测量理论
from __future__ import print_function, division, unicode_literals
from psy.ctt import BivariateCtt
import numpy as np

score = np.loadtxt('data/lsat.csv', delimiter=",")
ctt = BivariateCtt(score)
print(ctt.get_alpha_reliability())
print(ctt.get_composite_reliability())
print(ctt.get_discrimination())
print(ctt.get_difficulty())