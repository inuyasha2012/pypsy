# coding=utf-8
# 结构方程模型的参数估计
from __future__ import division, print_function, unicode_literals
import numpy as np
from psy import sem

data = np.loadtxt('data/ex5.11.dat')

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

lam_x, lam_y, phi_x, beta, gamma, var_e, var_e_x, var_e_y = sem(data, y, x, lam_x, lam_y, beta, gamma)

print('==========内源变量因子载荷=========')
print(lam_x)
print('=========外源变量因子载荷==========')
print(lam_y)
print('===========内源潜变量协方差矩阵=========')
print(phi_x)
print('============路径方程外源变量系数=========')
print(beta)
print('============路径方程内源变量系数=======')
print(gamma)
print('=============路径方程误差方差========')
print(np.diag(var_e))
print('============内源变量误差方差======')
print(np.diag(var_e_x))
print('=============外源变量误差方差=========')
print(np.diag(var_e_y))