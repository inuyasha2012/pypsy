# coding=utf-8
# 认知诊断ho-dina模型的MCMC参数估计
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import McmcHoDina
from psy.utils import r4beta

attrs = np.random.binomial(1, 0.5, (5, 60))

g = r4beta(1, 2, 0, 0.6, (1, 60))
no_s = r4beta(2, 1, 0.4, 1, (1, 60))

theta = np.random.normal(0, 1, (1000, 1))
lam00 = np.random.normal(0, 1, 5)
lam11 = np.random.uniform(0.5, 3, 5)

ho_dina = McmcHoDina(attrs=attrs)
skills_p = McmcHoDina.get_skills_p(lam0=lam00, lam1=lam11, theta=theta)
skills = np.random.binomial(1, skills_p)

yita = ho_dina.get_yita(skills)
p_val = ho_dina.get_p(yita, guess=g, no_slip=no_s)
score = np.random.binomial(1, p_val)

ho_dina_est = McmcHoDina(attrs=attrs, score=score, max_iter=10000, burn=5000)
est_lam0, est_lam1, est_theta, est_skills, est_no_s, est_g = ho_dina_est.mcmc()

