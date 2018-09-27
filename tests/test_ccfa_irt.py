# coding=utf-8
from __future__ import  print_function, division, unicode_literals
import numpy as np
from psy import delta_i_ccfa, get_irt_parameter, get_thresholds, data


def test_ccfa_irt():
    score = data['lsat.dat']
    lam0 = np.ones((5, 1))
    lam, phi, theta = delta_i_ccfa(score, lam0)
    _thresholds = get_thresholds(score)
    thresholds = np.array(_thresholds)
    get_irt_parameter(lam, thresholds, theta)
