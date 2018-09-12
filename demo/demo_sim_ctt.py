# coding=utf-8
# 模拟CTT信度
from __future__ import division, print_function, unicode_literals
import numpy as np
from psy.ctt import BivariateCtt

# 单个试题的模拟信度
R_LIST = [0.5, 0.55, 0.60, 0.65, 0.7]
# 样本量
PERSON_SIZE = 500
true_r = 1
while true_r >= 0.9:
    true_score_list = np.zeros((PERSON_SIZE, 5))
    x_score_list = np.zeros((PERSON_SIZE, 5))
    for i, r in enumerate(R_LIST):
        # 真分数
        t = np.random.randint(0, 2, PERSON_SIZE)
        var_t = np.var(t)
        var_x = var_t / r
        # 随机误差的方差和标准差
        var_e = var_x - var_t
        std_e = var_e ** 0.5
        e = np.random.normal(0, std_e, PERSON_SIZE)
        # 观察分数
        x = np.round(t + e, 0)
        x[x < 0] = 0
        x[x > 1] = 1
        true_score_list[:, i] = t
        x_score_list[:, i] = x

        var_total_t = np.var(np.sum(true_score_list, axis=1))
        var_total_x = np.var(np.sum(x_score_list, axis=1))
        # 真实信度
        true_r = var_total_t / var_total_x

print(true_r)
ctt = BivariateCtt(scores=x_score_list)
# alpha系数
print(ctt.get_alpha_reliability())
# 组合信度
print(ctt.get_composite_reliability())