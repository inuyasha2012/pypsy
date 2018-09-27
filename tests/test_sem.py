# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from psy import sem, data


def test_sem():
    dat = data['ex5.11.dat']
    beta = np.array([
        [0, 0],
        [1, 0]
    ])
    gamma = np.array([
        [1, 1],
        [0, 0]
    ])
    x = [0, 1, 2, 3, 4, 5]
    lam_x = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ])
    y = [6, 7, 8, 9, 10, 11]
    lam_y = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1],
    ])
    sem(dat, y, x, lam_x, lam_y, beta, gamma)
