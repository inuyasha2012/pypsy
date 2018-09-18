# coding=utf-8
from __future__ import  print_function, division, unicode_literals
import numpy as np
from psy import delta_i_ccfa, get_irt_parameter, get_thresholds


def test_ccfa_irt():
    data = np.loadtxt('data/lsat.csv', delimiter=',')
    lam0 = np.ones((5, 1))
    lam, phi, theta = delta_i_ccfa(data, lam0)
    _thresholds = get_thresholds(data)
    thresholds = np.array(_thresholds)
    get_irt_parameter(lam, thresholds, theta)