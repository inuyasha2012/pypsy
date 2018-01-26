# coding=utf-8
import numpy as np


class GPForth(object):
    # 基于正交梯度投影算法的因子旋转
    def __init__(self, init_loadings, method='varimax'):
        self._init_loadings = init_loadings
        self._method = method

    def solve(self):
        init_loadings = self._init_loadings
        t_mat = np.eye(init_loadings.shape[1])
        method = getattr(self, self._method)
        al = 1
        l = np.dot(init_loadings, t_mat)
        f, gq = method(l)
        g = np.dot(init_loadings.transpose(), gq)
        for i in range(500):
            m = np.dot(t_mat.transpose(), g)
            s = (m + m.transpose()) / 2
            gp = g - np.dot(t_mat, s)
            s = np.sum(np.diag(np.dot(gp, gp.transpose()))) ** 0.5
            if s < 1e-5:
                break
            al = 2 * al
            for j in range(10):
                x = t_mat - al * gp
                u, d, v = np.linalg.svd(x)
                t_mat_temp = np.dot(u, v)
                l = np.dot(init_loadings, t_mat_temp)
                f_temp, gq_temp = method(l)
                if f_temp < (f - 0.5 * s ** 2 * al):
                    break
                al = al / 2.0
            t_mat = t_mat_temp
            f = f_temp
            g = np.dot(init_loadings.transpose(), gq_temp)
        return l

    @staticmethod
    def varimax(l):
        # varimax旋转
        mean = np.mean(l ** 2, axis=0)
        mean.shape = 1, mean.shape[0]
        ql = l ** 2 - mean
        f = -1. * np.sum(np.diag(np.dot(ql.transpose(), ql))) / 4
        gq = -1. * l * ql
        return f, gq
