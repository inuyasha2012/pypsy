# coding=utf-8
import numpy as np
from itertools import product
from psy.settings import GH_POINT_DT


def r4beta(shape1, shape2, a, b, size):
    # 4参数beta分布
    x = np.random.beta(shape1, shape2, size)
    return (b - a) * x + a


def get_log_beta_pd(no_slip, guess):
    # beta分布的对数概率密度函数
    return np.log(0.6 - guess) + np.log(no_slip - 0.4)


def get_log_normal_pd(x):
    # 正态分布的对数概率密度函数
    return x ** 2 * -0.5


def get_log_lognormal_pd(x):
    # 对数正态分布的对数概率密度函数
    return np.log(1.0 / x) + get_log_normal_pd(np.log(x))


def inverse_logistic(y):
    # logistic的反函数
    return -1 * np.log(1.0 / y - 1)


def get_nodes_weights(dim_size):
    # 多维高斯厄米特积分
    x_nodes, x_weights = np.polynomial.hermite.hermgauss(GH_POINT_DT[dim_size])
    x_nodes2_list = [x_nodes for _ in range(dim_size)]
    x_nodes = np.array(list(product(*x_nodes2_list)))
    x_nodes = x_nodes * 2 ** 0.5
    x_weights2_list = [x_weights for _ in range(dim_size)]
    x_weights = np.array(list(product(*x_weights2_list)))
    x_weights = np.prod(x_weights, axis=1)
    x_weights.shape = x_weights.shape[0], 1
    x_weights = x_weights / np.pi ** 0.5
    return x_nodes, x_weights
