# coding=utf-8
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import McmcDina
from psy.utils import r4beta


def test_mcmc_dina():
    attrs = np.random.binomial(1, 0.5, (5, 60))
    skills = np.random.binomial(1, 0.5, (1000, 5))

    g = r4beta(1, 2, 0, 0.6, (1, 60))
    no_s = r4beta(2, 1, 0.4, 1, (1, 60))

    temp = McmcDina(attrs=attrs)
    yita = temp.get_yita(skills)
    p_val = temp.get_p(yita, guess=g, no_slip=no_s)
    score = np.random.binomial(1, p_val)

    em_dina = McmcDina(attrs=attrs, score=score, max_iter=100, burn=50)
    em_dina.mcmc()
