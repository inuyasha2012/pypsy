# coding=utf-8
# 单维IRT参数估计
from __future__ import print_function, division, unicode_literals
from psy import Irt, data

score = data['lsat.dat']
model = Irt(scores=score, link='probit')
res = model.fit()
print(res)

model = Irt(scores=score, link='logit')
res = model.fit()
print(res)
