import numpy as np
from psy import Irt2PL

f = file('lsat.csv')
score = np.loadtxt(f, delimiter=",")
res = Irt2PL(scores=score).em()
print(res)