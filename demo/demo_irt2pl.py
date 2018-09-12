# coding=utf-8
# 单维IRT参数估计
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Irt2PL

score = np.loadtxt('data/lsat.csv', delimiter=",")
res = Irt2PL(scores=score).em()
print(res)