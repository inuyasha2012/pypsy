from __future__ import print_function
from psy.ctt import Ctt
import numpy as np

f = file('lsat.csv')
score = np.loadtxt(f, delimiter=",")
ctt = Ctt(score)
print(ctt.get_reliability())
print(ctt.get_cr())
print(ctt.get_discrimination())
print(ctt.get_difficulty())