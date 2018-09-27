# coding=utf-8
# 探索性因子分析
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Factor, data

score = data['lsat.dat']
factor = Factor(score, 5)
print(factor.loadings)
