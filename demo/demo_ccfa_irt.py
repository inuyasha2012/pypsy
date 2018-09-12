# coding=utf-8
# 利用属性数据下的验证性因子分析计算项目反应模型的参数
from __future__ import  print_function, division, unicode_literals
import numpy as np
from psy import delta_i_ccfa, get_irt_parameter, get_thresholds

data = np.loadtxt('data/lsat.csv', delimiter=',')
lam0 = np.ones((5, 1))
lam, phi, theta = delta_i_ccfa(data, lam0)
_thresholds = get_thresholds(data)
thresholds = np.array(_thresholds)
a, b = get_irt_parameter(lam, thresholds, theta)

print('probit区分度')
print(a)
print('logistic区分度')
print(a * 1.702)
print('难度')
print(b / a)