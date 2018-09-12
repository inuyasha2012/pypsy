# coding=utf-8
# dina模型的MCMC参数估计
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import McmcDina
from psy.utils import r4beta

attrs = np.random.binomial(1, 0.5, (5, 60))
skills = np.random.binomial(1, 0.5, (1000, 5))

g = r4beta(1, 2, 0, 0.6, (1, 60))
no_s = r4beta(2, 1, 0.4, 1, (1, 60))

temp = McmcDina(attrs=attrs)
yita = temp.get_yita(skills)
p_val = temp.get_p(yita, guess=g, no_slip=no_s)
score = np.random.binomial(1, p_val)

em_dina = McmcDina(attrs=attrs, score=score, max_iter=10000, burn=7000)
est_skills, est_no_s, est_g = em_dina.mcmc()
