import numpy as np

from psy import Factor

f = file('lsat.csv')
score = np.loadtxt(f, delimiter=",")
factor = Factor(score, 5)
print(factor.loadings)
