# coding=utf-8
from __future__ import division, print_function
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize
import numpy as np
from itertools import combinations, product
import warnings


def _get_polychoric_init(data):
    # 用相关系数作为polychoric的初值
    return np.corrcoef(data, rowvar=False)


def _get_dbinorm(x, y, rho):
    # 计算二元正态分布概率密度
    r = 1 - rho ** 2
    p = 1 / (2 * np.pi * r ** 0.5) * np.exp(-0.5 * (x ** 2 - 2 * rho * x * y + y ** 2) / r)
    return p


def get_thresholds(data):
    # 基于边际信息计算阈值
    max_arr = np.max(data, axis=0)
    min_arr = np.min(data, axis=0)
    props = []
    for i, _max in enumerate(max_arr):
        int_max = int(_max)
        int_min = int(min_arr[i])
        prop = np.zeros(int_max - int_min + 1)
        for j in range(int_min, int_max + 1):
            prop[j - int_min] = len(data[:, i][data[:, i] == j])
        props.append(np.array(prop))
    threholds = []
    for prop in props:
        freq = prop / sum(prop)
        threholds.append(norm.ppf(np.cumsum(freq)[:-1]))
    return threholds


def _get_freq(data_xy):
    # 计算列联表的频率
    max_x = int(max(data_xy[:, 0]))
    min_x = int(min(data_xy[:, 0]))
    max_y = int(max(data_xy[:, 1]))
    min_y = int(min(data_xy[:, 1]))
    freq = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            bool_arr = np.all(data_xy == np.array([i, j]), axis=1)
            freq[i - min_x, j - min_y] = len(bool_arr[bool_arr])
    return freq


def _get_prob(rho, threshold1, threshold2):
    # 计算列联表的二元正态分布累计概率
    len1 = len(threshold1)
    len2 = len(threshold2)
    p1 = norm.cdf(threshold1)
    p2 = norm.cdf(threshold2)
    threshold_pd = np.array(list(product(threshold1, threshold2)))
    bi = multivariate_normal.cdf(threshold_pd, cov=np.array([[1, rho], [rho, 1]]))
    if isinstance(bi, float):
        bi = np.array([[bi]])
    else:
        bi.shape = len1, len2
    bi = np.column_stack((np.zeros(len1), bi, p1))
    temp = np.zeros(bi.shape[1])
    temp[-1] = 1
    temp[1:-1] = p2
    bi = np.row_stack((np.zeros(bi.shape[1]), bi, temp))
    pi = bi[1:, 1:] - bi[1:, :-1] - bi[:-1, 1:] + bi[:-1, :-1]
    pi[pi <= 0] = 1e-200
    return pi


def _get_phi(rho, threshold1, threshold2):
    # 计算列联表的二元正态分布概率密度
    len1 = len(threshold1) + 1
    len2 = len(threshold2) + 1
    _threshold1 = np.concatenate((np.array([-np.inf]), threshold1, np.array([np.inf])))
    _threshold2 = np.concatenate((np.array([-np.inf]), threshold2, np.array([np.inf])))
    phi = np.zeros((len1, len2))
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            p1 = p2 = p3 = p4 = 0
            if i < len1 and j < len2:
                p1 = _get_dbinorm(_threshold1[i], _threshold2[j], rho)
            if i > 1 and j < len2:
                p2 = _get_dbinorm(_threshold1[i - 1], _threshold2[j], rho)
            if i < len1 and j > 1:
                p3 = _get_dbinorm(_threshold1[i], _threshold2[j - 1], rho)
            if i > 1 and j > 1:
                p4 = _get_dbinorm(_threshold1[i - 1], _threshold2[j - 1], rho)
            phi[i - 1, j - 1] = p1 - p2 - p3 + p4
    return phi


def _get_loglik(rho, threshold1, threshold2, freq):
    # 计算极大似然估计下的对数似然函数值
    rho = 1 - 1e-5 if rho >= 1 else rho
    rho = -1 + 1e-5 if rho <= -1 else rho
    pi = _get_prob(rho, threshold1, threshold2)
    lik = np.sum(freq * np.log(pi))
    return -lik


def _get_dloglik(rho, threshold1, threshold2, freq):
    # 计算极大似然估计下的对数似然函数的一阶导数值
    rho = 1 - 1e-5 if rho > (1 - 1e-5) else rho
    rho = 1e-5 - 1 if rho < (1e-5 - 1) else rho
    pi = _get_prob(rho, threshold1, threshold2)
    phi = _get_phi(rho, threshold1, threshold2)
    drho = np.sum(freq / (pi + 1e-20) * phi)
    return np.array([-drho])


def _get_polychoric_cor(data):
    # 计算polychoric相关
    thresholds = get_thresholds(data)
    threshold_idx_list = range(data.shape[1])
    idx_cb_list = combinations(threshold_idx_list, 2)
    init_rho = _get_polychoric_init(data)
    cor = np.zeros((data.shape[1], data.shape[1]))
    for idx_cb in idx_cb_list:
        rho0 = init_rho[idx_cb]
        threshold1 = thresholds[idx_cb[0]]
        threshold2 = thresholds[idx_cb[1]]
        freqs = _get_freq(data[:, idx_cb])
        res = minimize(_get_loglik, rho0, args=(threshold1, threshold2, freqs), method='BFGS', jac=_get_dloglik)
        cor[idx_cb[0], idx_cb[1]] = res.x
    return cor + cor.T


def delta_i_ccfa(data, lam, step=0.01, max_iter=1000000, rdd=3, tol=1e-6):
    # delta默认为I的CFA参数估计
    # polychoric相关矩阵
    s = _get_polychoric_cor(data)
    # 潜在变量协方差矩阵初值，方差固定为1
    phi = np.eye(lam.shape[1])
    for i in range(max_iter):
        # 估计协方差矩阵
        sigma = np.dot(np.dot(lam, phi), lam.transpose())
        omega = sigma - s
        # 固定omega对角线为0
        lo = range(len(omega))
        omega[lo, lo] = 0
        omega_lam = np.dot(omega, lam)
        # lambda的梯度
        dlam = 2 * np.dot(omega_lam, phi)
        dlam[lam == 0] = 0
        # phi的梯度
        dphi = np.dot(lam.transpose(), omega_lam)
        lp = range(lam.shape[1])
        dphi[lp, lp] = 0
        # 梯度下降
        delta_lam = step * dlam
        delta_phi = step * dphi
        lam = lam - delta_lam
        phi = phi - delta_phi
        if max(np.max(np.abs(delta_lam)), np.max(np.abs(delta_phi))) < tol:
            theta = _get_theta(lam, phi)
            return np.round(lam, rdd), np.round(phi, rdd), np.round(theta, rdd)
    warnings.warn('no coverage')
    theta = _get_theta(lam, phi)
    return np.round(lam, rdd), np.round(phi, rdd), np.round(theta, rdd)


def _get_theta(lam, phi):
    I = np.eye(lam.shape[0])
    sigma = np.dot(np.dot(lam, phi), lam.transpose())
    theta = I - sigma
    return np.diag(theta)


def get_irt_parameter(lam, thresholds, theta):
    std = theta ** 0.5
    a = lam[:, 0] / std
    b = thresholds[:, 0] / std
    return a, b