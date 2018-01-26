from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Irt2PL

f = file('data/lsat.csv')
score = np.loadtxt(f, delimiter=",")
res = Irt2PL(scores=score).em()
print(res)