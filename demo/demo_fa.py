from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import Factor

f = file('data/lsat.csv')
score = np.loadtxt(f, delimiter=",")
factor = Factor(score, 5)
print(factor.loadings)
