from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Mirt2PL


f = file('data/lsat.csv')
score = np.loadtxt(f, delimiter=",")
res = Mirt2PL(scores=score, dim_size=2).em()
print(res[2])
print(res[0])
