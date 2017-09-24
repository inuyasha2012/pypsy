from __future__ import division
import numpy as np

from psy.ctt import Ctt

r_list = [0.5, 0.5, 0.5, 0.5, 0.5]
t_list = np.zeros((500, 5))
x_list = np.zeros((500, 5))
for i, r in enumerate(r_list):
    t = np.random.randint(0, 2, 500)
    # t.sort()
    # t = np.round(t, 0)
    # t[t > 1] = 1
    # t[t < 0] = 0
    var_t = np.var(t)
    var_x = var_t / r
    var_e = var_x - var_t
    std_e = var_e ** 0.5
    e = np.random.normal(0, std_e, 500)
    x = np.round(t + e, 0)
    x[x < 0] = 0
    x[x > 1] = 1
    t_list[:, i] = t
    x_list[:, i] = x

var_tt = np.var(np.sum(t_list, axis=1))
var_tx = np.var(np.sum(x_list, axis=1))
np.savetxt('ctt.csv', x_list, delimiter=',')
print var_tt / var_tx
ctt = Ctt(scores=x_list)
print ctt.get_reliability()
print ctt.get_cr()