# coding=utf-8
# 认知诊断DINA模型的EM参数估计
from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import EmDina, MlDina
from psy.utils import r4beta

attrs = np.random.binomial(1, 0.5, (5, 60))
skills = np.random.binomial(1, 0.7, (1000, 5))

g = r4beta(1, 2, 0, 0.6, (1, 60))
no_s = r4beta(2, 1, 0.4, 1, (1, 60))

temp = EmDina(attrs=attrs)
yita = temp.get_yita(skills)
p_val = temp.get_p(yita, guess=g, no_slip=no_s)
score = np.random.binomial(1, p_val)

# 估计项目参数
em_dina = EmDina(attrs=attrs, score=score)
est_no_s, est_g = em_dina.em()

print(np.mean(np.abs(est_no_s - no_s)))
print(np.mean(np.abs(est_g - g)))

# 估计被试掌握技能情况
dina_est = MlDina(guess=est_g, no_slip=est_no_s, attrs=attrs, score=score)
est_skills = dina_est.solve()