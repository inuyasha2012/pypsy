# coding=utf-8
# 单维IRT参数估计
from __future__ import print_function, division, unicode_literals

import time

from psy import Irt, data
import numpy as np

from psy.irt.irm import MCEMIrt1, MCEMIrt

score = data['lsat.dat']
# model = Irt(scores=score, link='logit', params_type='2PL',
#             constraint={'slop': [{'item': 1, 'value': 1}], 'threshold': [{'item': 5, 'value': 0.5}]})
# res = model.fit()
# print(res)
#
# model = Irt(scores=score, link='probit')
# res = model.fit()
# print(res)

# score = data['lsat.dat']

#
# slop = np.random.uniform(0.7, 2, 20)
# print(slop)
# threshold = np.random.normal(0, 1, size=20)
# # guess = np.random.uniform(0.05, 0.3, 1000)
# theta = np.random.normal(0, 1, size=(20000, 1))
# print(threshold)
# print('==================')
#
# z = slop * theta + threshold
#
# p_logit = 1 / (1 + np.exp(-z))
#
# # p = guess + (1 - guess) * p_logit
#
# score = np.random.binomial(1, p_logit)

model = Irt(scores=score, link='probit')
res = model.fit()
print(res)


s = time.time()
model = MCEMIrt1(scores=score, max_iter=1000, dim=1)
res = model.fit()
e = time.time()
print(e - s)
print(res)

