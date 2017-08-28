import numpy as np
from psy import Mirt2PL


f = file('lsat.csv')
score = np.loadtxt(f, delimiter=",")
res = Mirt2PL(scores=score, dim_size=2).em()
print(res[2])
