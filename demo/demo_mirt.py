# coding=utf-8
# 多维项目反应理论的参数估计
from __future__ import print_function, division, unicode_literals
from psy import Mirt, data

score = data['lsat.dat']
res = Mirt(scores=score, dim_size=2).fit()
print(res[2])
print(res[0])
