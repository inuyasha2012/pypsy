# coding=utf-8
# 正态数据下的验证性因子分析
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import cfa

lam = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])

data = np.loadtxt('data/ex5.6.dat')
lam, phi, var_e = cfa(data, lam)
# 因子载荷
print(lam)
# 误差方差
print(np.diag(var_e))
# 潜变量协方差矩阵
print(phi)