# coding=utf-8
# 多维项目反应理论的参数估计
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Mirt2PL

score = np.loadtxt('data/lsat.csv', delimiter=",")
res = Mirt2PL(scores=score, dim_size=2).em()
print(res[2])
print(res[0])
