# coding=utf-8
from __future__ import division, print_function
import numpy as np
import warnings
from psy.polychoric import get_polychoric_cor


def delta_i_ccfa(data, lam, step=0.01, max_iter=1000000, rdd=3, tol=1e-6):
    # delta默认为I的CFA参数估计
    # polychoric相关矩阵
    s = get_polychoric_cor(data)
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
