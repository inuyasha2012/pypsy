# coding=utf-8
from __future__ import print_function
import warnings
import numpy as np
from psy.irt.base import BaseEmIrt, ProbitMixin, LogitMixin, Logit3PLMixin
from psy.settings import X_WEIGHTS, X_NODES
from scipy.optimize import minimize

class _Irt(BaseEmIrt):

    def __init__(self, init_slop=None, init_threshold=None, params_type='2PL', *args, **kwargs):
        super(_Irt, self).__init__(*args, **kwargs)
        self._params_dt = {}
        self.params_type = params_type.upper()
        if self.params_type == '2PL':
            if init_slop is not None:
                self._params_dt['slop'] = init_slop
            else:
                self._params_dt['slop'] = np.ones(self.item_size)
        if init_threshold is not None:
            self._params_dt['threshold'] = init_threshold
        else:
            self._params_dt['threshold'] = np.zeros(self.item_size)

    def _m_step(self, full_dis, right_dis, theta, p_val, z_val, **params_dt):
        # 一阶导数, 二阶导数
        dp, ddp = self._get_dp_n_ddp(right_dis, full_dis, p_val, z_val)
        # jac矩阵和hess矩阵
        if self.params_type == '2PL':
            dp_sum, dp_theta_sum, ddp_sum, ddp_theta_sum, ddp_theta_square_sum = self._get_jac_n_hess_val(dp, ddp, theta)
        else:
            dp_sum, ddp_sum = self._get_jac_n_hess_val(dp, ddp, theta)
        delta_list = np.zeros((self.item_size, 2))
        hess_list = []
        # 把求稀疏矩阵的逆转化成求每个题目的小矩阵的逆
        if self.params_type == '2PL':
            for i in range(self.item_size):
                jac = np.array([dp_sum[i], dp_theta_sum[i]])
                hess = np.array(
                    [[ddp_sum[i], ddp_theta_sum[i]],
                     [ddp_theta_sum[i], ddp_theta_square_sum[i]]]
                )
                delta = np.linalg.solve(hess, jac)
                params_dt['slop'][i] -= delta[1]
                params_dt['threshold'][i] -= delta[0]
                delta_list[i] = delta
                hess_list.append(hess)
            return params_dt, delta_list, hess_list
        else:
            for i in range(self.item_size):
                jac = np.array([dp_sum[i]])
                hess = np.array(
                    [[ddp_sum[i]]]
                )
                delta = np.linalg.solve(hess, jac)
                params_dt['threshold'][i] -= delta[0]
                delta_list[i] = delta
                hess_list.append(hess)
            return params_dt, delta_list, hess_list

    def _get_jac_n_hess_val(self, dp, ddp, theta):
        dp_sum = np.sum(dp, axis=0)
        ddp_sum = -1 * np.sum(ddp, axis=0)
        if self.params_type == '2PL':
            dp_theta_sum = np.sum(dp * theta, axis=0)
            ddp_theta_sum = -1 * np.sum(ddp * theta, axis=0)
            ddp_theta_square_sum = -1 * np.sum(ddp * theta ** 2, axis=0)
            return dp_sum, dp_theta_sum, ddp_sum, ddp_theta_sum, ddp_theta_square_sum
        return dp_sum, ddp_sum

    def fit(self):
        # EM算法
        max_iter = self._max_iter
        tol = self._tol
        params_dt = self._params_dt
        for i in range(max_iter):
            z = self.z(theta=X_NODES, **params_dt)
            p_val = self.p(z)
            full_dis, right_dis, loglik = self._e_step(p_val, X_WEIGHTS)
            params_dt, delta_list, hess_list = self._m_step(full_dis=full_dis,
                                                                right_dis=right_dis,
                                                                theta=X_NODES,
                                                                p_val=p_val,
                                                                z_val=z,
                                                                **params_dt)
            if np.max(np.abs(delta_list)) < tol:
                return params_dt
        warnings.warn("no convergence")
        return params_dt


class _ProbitIrt(_Irt, ProbitMixin):

    def _get_dp_n_ddp(self, right_dis, full_dis, p_val, z_val, *args, **kwargs):
        h = self._h(z_val)
        w = self._w(h, p_val)
        dp = w * (right_dis - full_dis * p_val) / h
        ddp = full_dis * w
        return dp, ddp

    def _h(self, z):
        # probit函数的h值，方便计算
        return (1.0 / ((2 * np.pi) ** 0.5)) * np.exp(-1 * z ** 2 / 2.0)

    def _w(self, h, prob):
        # probit函数的w值，可以看成权重，方便计算和呈现
        pq = (1 - prob) * prob
        return h ** 2 / (pq + 1e-10)


class _LogitIrt(_Irt, LogitMixin):
    # EM算法求解
    # E步求期望
    # M步求极大，这里M步只迭代一次

    def _get_dp_n_ddp(self, right_dis, full_dis, p_val, *args, **kwargs):
        return right_dis - full_dis * p_val, full_dis * p_val * (1 - p_val)


class _Logit3PLIrt(BaseEmIrt, Logit3PLMixin):

    def __init__(self, init_slop=None, init_threshold=None, init_guess=None, *args, **kwargs):
        super(_Logit3PLIrt, self).__init__(*args, **kwargs)
        self._params_dt = {}
        if init_slop is not None:
            self._params_dt['slop'] = init_slop
        else:
            self._params_dt['slop'] = np.ones(self.item_size)
        if init_threshold is not None:
            self._params_dt['threshold'] = init_threshold
        else:
            self._params_dt['threshold'] = np.zeros(self.item_size)
        if init_guess is not None:
            self._params_dt['guess'] = init_guess
        else:
            self._params_dt['guess'] = np.zeros(self.item_size)

    def _get_dp_n_ddp(self, right_dis, full_dis, p_val, p_guess_val, *args, **kwargs):
        return right_dis * p_val / (p_guess_val + 1e-6) - full_dis * p_val, full_dis * p_val * (1 - p_val)

    def _m_step(self, full_dis, right_dis, theta, p_val, p_guess_val, **params_dt):
        # 一阶导数, 二阶导数
        dp, ddp = self._get_dp_n_ddp(right_dis, full_dis, p_val, p_guess_val)
        # jac矩阵和hess矩阵
        jac = self._get_jac(dp, theta, params_dt['guess'], p_guess_val)
        delta_list = np.zeros((self.item_size, 2))
        hess_list = []

        # 把求稀疏矩阵的逆转化成求每个题目的小矩阵的逆
        def _item_loglik(params, index):
            # 似然函数
            slop = params[0]
            threshold = params[1]
            guess = params[2]
            p_val = self.p_guess(slop=slop, threshold=threshold, guess=guess, theta=X_NODES)['p_guess_val']
            p_val[p_val <= 0] = 1e-10
            p_val[p_val >= 1] = 1 - 1e-10
            loglik_val = np.log(p_val) * right_dis[:, index][:, np.newaxis] + np.log(1 - p_val) * (full_dis - right_dis[:, index][:, np.newaxis])
            return -np.sum(loglik_val)

        def _jac(params, index):
            slop = params[0]
            threshold = params[1]
            guess = params[2]
            p_dt = self.p_guess(slop=slop, threshold=threshold, guess=guess, theta=X_NODES)
            p_val = p_dt['p_val']
            p_guess_val = p_dt['p_guess_val']
            dp, ddp = self._get_dp_n_ddp(right_dis[:, index][:, np.newaxis], full_dis, p_val, p_guess_val)
            dp_sum = np.sum(dp)
            dp_theta = np.sum(dp * theta)
            dp_guess = np.sum(dp * (1 / (p_guess_val - guess)))
            return np.array([dp_sum, dp_theta, dp_guess])

        slop = params_dt['slop']
        threshold = params_dt['threshold']
        guess = params_dt['guess']

        for i in range(self.item_size):
            res = minimize(
                _item_loglik,
                np.array([slop[i], threshold[i], guess[i]]),
                (i, ),
                'Nelder-Mead',
                _jac,
                options={'disp': True},
                tol=1e-5
            )
            params_dt['slop'][i] = res.x[0]
            params_dt['threshold'][i] = res.x[1]
            params_dt['guess'][i] = res.x[2]
        return params_dt, delta_list, hess_list

    def _get_jac(self, dp, theta, guess, p_guess_val):
        dp_sum = np.sum(dp, axis=0)
        dp_theta = np.sum(dp * theta, axis=0)
        dp_guess = np.sum(dp * (1 / (p_guess_val - guess)), axis=0)
        return dp_sum, dp_theta, dp_guess

    def fit(self):
        # EM算法
        max_iter = self._max_iter
        tol = self._tol
        params_dt = self._params_dt
        for i in range(max_iter):
            p_guess_dt = self.p_guess(
                slop=params_dt['slop'],
                threshold=params_dt['threshold'],
                guess=params_dt['guess'],
                theta=X_NODES
            )
            p_val = p_guess_dt['p_val']
            p_guess_val = p_guess_dt['p_guess_val']
            full_dis, right_dis, loglik = self._e_step(p_guess_val, X_WEIGHTS)
            params_dt, delta_list, hess_list = self._m_step(full_dis=full_dis,
                                                            right_dis=right_dis,
                                                            theta=X_NODES,
                                                            p_val=p_val,
                                                            p_guess_val=p_guess_val,
                                                            **params_dt)
            print(i)
            print('===========================')
            print('===========================')
            # if np.max(np.abs(delta_list)) < tol:
            # return params_dt
        # warnings.warn("no convergence")
        return params_dt

class Irt(object):

    LINK_DT = {
        'logit1PL': _LogitIrt,
        'probit1PL': _ProbitIrt,
        'logit2PL': _LogitIrt,
        'probit2PL': _ProbitIrt,
        'logit3PL': _Logit3PLIrt,
        'probit3PL': _ProbitIrt
    }
    PARAMS_TYPE_TP = ('1PL', '2PL', '3PL')

    def __init__(self, scores, link='logit', params_type='2PL', init_slop=None, init_threshold=None, init_guess=None,
                 max_iter=1000, tol=1e-5,):
        _link = link.lower()
        if _link not in ('probit', 'logit'):
            raise ValueError('link must be probit or logit')
        _params_type = params_type.upper()
        if _params_type not in self.PARAMS_TYPE_TP:
            raise ValueError('params type must be 1PL, 2PL or 3PL')
        _link = self.LINK_DT[_link + _params_type]
        kwargs = {'scores': scores, 'max_iter': max_iter, 'tol': tol, 'init_threshold': init_threshold}
        if _params_type in ('1PL', '2PL'):
            kwargs['params_type'] = _params_type
        if _params_type in ('2PL', '3PL'):
            kwargs['init_slop'] = init_slop
        if _params_type == '3PL':
            kwargs['init_guess'] = init_guess
        self._model = _link(**kwargs)

    def fit(self):
        return self._model.fit()
