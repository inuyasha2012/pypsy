# coding=utf-8
from __future__ import  print_function, division, unicode_literals
import numpy as np
from psy import delta_i_ccfa, data


def test_ccfa():
    lam0 = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]
    ])
    delta_i_ccfa(data['ex5.2.dat'], lam0)
