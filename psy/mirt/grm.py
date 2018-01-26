# coding=utf-8
from __future__ import print_function, division, unicode_literals
import warnings
import numpy as np


class Grm(object):

    def __init__(self, scores=None, init_slop=None, init_threshold=None, max_iter=1000, tol=1e-5, gp_size=11):
        # 试题最大反应计算
        max_score = int(np.max(scores))
        min_score = int(np.min(scores))
        self._rep_len = max_score - min_score + 1
        self.scores = {}
        for i in range(scores.shape[1]):
            temp_scores = np.zeros((scores.shape[0], self._rep_len))
            for j in range(self._rep_len):
                temp_scores[:, j][scores[:, i] == min_score + j] = 1
            self.scores[i] = temp_scores
        # 题量
        self.item_size = scores.shape[1]
        if init_slop is not None:
            self._init_slop = init_slop
        else:
            self._init_slop = np.ones(scores.shape[1])
        if init_threshold is not None:
            self._init_thresholds = init_threshold
        else:
            self._init_thresholds = np.zeros((scores.shape[1], self._rep_len - 1))
            for i in range(scores.shape[1]):
                self._init_thresholds[i] = np.arange(self._rep_len / 2 - 1, -self._rep_len / 2, -1)
        self._max_iter = max_iter
        self._tol = tol
        self.x_nodes, self.x_weights = self.get_gh_point(gp_size)

    @staticmethod
    def get_gh_point(gp_size):
        x_nodes, x_weights = np.polynomial.hermite.hermgauss(gp_size)
        x_nodes = x_nodes * 2 ** 0.5
        x_nodes.shape = x_nodes.shape[0], 1
        x_weights = x_weights / np.pi ** 0.5
        x_weights.shape = x_weights.shape[0], 1
        return x_nodes, x_weights

    @staticmethod
    def p(z):
        # 回答为某一反应的概率函数
        p_val_dt = {}
        for key in z.keys():
            e = np.exp(z[key])
            p = e / (1.0 + e)
            p_val_dt[key] = p
        return p_val_dt

    @staticmethod
    def z(slop, thresholds, theta):
        # z函数
        z_val = {}
        temp = slop * theta
        for i, threshold in enumerate(thresholds):
            z_val[i] = temp[:, i][:, np.newaxis] + threshold
        return z_val

    def _lik(self, p_val_dt):
        loglik_val = 0
        rep_len = self._rep_len
        scores = self.scores
        for i in range(self.item_size):
            for j in range(rep_len):
                p_pre = 1 if j == 0 else p_val_dt[i][:, j - 1]
                p = 0 if j == rep_len - 1 else p_val_dt[i][:, j]
                loglik_val += np.dot(np.log(p_pre - p + 1e-200)[:, np.newaxis], scores[i][:,  j][np.newaxis])
        return np.exp(loglik_val)

    def _e_step(self, p_val_dt, weights):
        # E步计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val_dt) * weights
        # 归一化
        lik_wt_sum = np.sum(lik_wt, axis=0)
        _temp = lik_wt / lik_wt_sum
        # theta的人数分布
        full_dis = np.sum(_temp, axis=1)
        # theta下回答的人数分布
        right_dis_dt = {}
        for i in range(self.item_size):
            right_dis_dt[i] = np.dot(_temp, scores[i])
        # full_dis.shape = full_dis.shape[0], 1
        # 对数似然值
        print(np.sum(np.log(lik_wt_sum)))
        return full_dis, right_dis_dt

    def _pq(self, p_val):
        return p_val * (1 - p_val)

    @staticmethod
    def _item_jac(p_val, pq_val, right_dis, len_threshold, rep_len, theta):
        # 雅克比矩阵
        dloglik_val = np.zeros(len_threshold + 1)
        _theta = theta[:, 0]
        for i in range(rep_len):
            p_pre, pq_pre = (1, 0) if i == 0 else (p_val[:, i - 1], pq_val[:, i - 1])
            p, pq = (0, 0) if i == rep_len - 1 else (p_val[:, i], pq_val[:, i])
            temp1 = _theta * right_dis[:, i] * (1 - p_pre - p)
            dloglik_val[-1] += np.sum(temp1)
            if i < rep_len - 1:
                temp2 = right_dis[:, i] * pq / (p - p_pre + 1e-200)
                dloglik_val[i] += np.sum(temp2)
            if i > 0:
                temp3 = right_dis[:, i] * pq_pre / (p_pre - p + 1e-200)
                dloglik_val[i - 1] += np.sum(temp3)
        return dloglik_val

    @staticmethod
    def _item_hess(p_val, pq_val, full_dis, len_threshold, rep_len, theta):
        # 黑塞矩阵
        ddloglik_val = np.zeros((len_threshold + 1, len_threshold + 1))
        _theta = theta[:, 0]
        for i in range(rep_len):
            p_pre, dp_pre = (1, 0) if i == 0 else (p_val[:, i - 1], pq_val[:, i - 1])
            p, dp = (0, 0) if i == rep_len - 1 else (p_val[:, i], pq_val[:, i])
            if i < rep_len - 1:
                temp1 = full_dis * _theta * dp * (dp_pre - dp) / (p_pre - p + 1e-200)
                ddloglik_val[len_threshold:, i] += np.sum(temp1)
                temp2 = full_dis * dp ** 2 / (p_pre - p + 1e-200)
                ddloglik_val[i, i] += -np.sum(temp2)
            if i > 0:
                temp3 = full_dis * _theta * dp_pre * (dp - dp_pre) / (p_pre - p + 1e-200)
                ddloglik_val[len_threshold:, i - 1] += np.sum(temp3, axis=0)
                temp4 = full_dis * dp_pre ** 2 / (p_pre - p + 1e-200)
                ddloglik_val[i - 1, i - 1] += -np.sum(temp4)
            if 0 < i < rep_len - 1:
                ddloglik_val[i, i - 1] = np.sum(full_dis * dp * dp_pre / (p_pre - p + 1e-200))
            temp5 = full_dis * _theta ** 2 * (dp_pre - dp) ** 2 / (p - p_pre)
            ddloglik_val[-1, -1] += np.sum(temp5, axis=0)
        ddloglik_val += ddloglik_val.transpose() - np.diag(ddloglik_val.diagonal())
        return ddloglik_val

    def _m_step(self, p_val_dt, full_dis, right_dis_dt, slop, thresholds, theta):
        # M步，牛顿迭代
        rep_len = self._rep_len
        len_threshold = thresholds.shape[1]
        delta_list = np.zeros((self.item_size, len_threshold + 1))
        for i in range(self.item_size):
            p_val = p_val_dt[i]
            pq_val = self._pq(p_val)
            right_dis = right_dis_dt[i]
            jac = self._item_jac(p_val, pq_val, right_dis, len_threshold, rep_len, theta)
            hess = self._item_hess(p_val, pq_val, full_dis, len_threshold, rep_len, theta)
            delta = np.linalg.solve(hess, jac)
            slop[i], thresholds[i] = slop[i] - delta[-1], thresholds[i] - delta[:-1]
            delta_list[i] = delta
        return slop, thresholds, delta_list

    def _est_item_parameter(self, slop, threshold, theta, p_val):
        full_dis, right_dis_dt = self._e_step(p_val, self.x_weights)
        return self._m_step(p_val, full_dis, right_dis_dt, slop, threshold, theta)

    def em(self):
        max_iter = self._max_iter
        tol = self._tol
        slop = self._init_slop
        thresholds = self._init_thresholds
        for i in range(max_iter):
            z = self.z(slop, thresholds, self.x_nodes)
            p_val = self.p(z)
            slop, thresholds, delta_list = self._est_item_parameter(slop, thresholds, self.x_nodes, p_val)
            if np.max(np.abs(delta_list)) < tol:
                print(i)
                return slop, thresholds
        warnings.warn("no convergence")
        return slop, thresholds
