# coding=utf-8
# 属性数据下的验证性因子分析
from __future__ import  print_function, division, unicode_literals
import numpy as np
from psy import delta_i_ccfa

data = np.loadtxt('data/ex5.2.dat')
lam0 = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [0, 1]
])
lam, phi, theta = delta_i_ccfa(data, lam0)
print('===因子载荷===')
print(lam)
print('===因子得分协方差矩阵===')
print(phi)
print('===残差方差===')
print(theta)
