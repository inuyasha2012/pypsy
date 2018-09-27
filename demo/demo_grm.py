# coding=utf-8
# 项目反应理论中的等级反应模型
from __future__ import division, print_function, unicode_literals
from psy import Grm, data

scores = data['lsat.dat']
grm = Grm(scores=scores)
print(grm.em())
