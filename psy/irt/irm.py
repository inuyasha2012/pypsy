# coding=utf-8
from __future__ import print_function
import warnings
import numpy as np
from statsmodels.genmod.families.links import probit

from psy.irt.base import BaseEmIrt, ProbitMixin, LogitMixin, Logit3PLMixin, BaseIrt
from psy.irt.trait import BayesLogitModel, MLProbitModel
from psy.settings import X_WEIGHTS, X_NODES
from scipy.optimize import minimize, fmin_bfgs
from scipy.stats import truncnorm
import statsmodels.api as sm


class _Irt(BaseEmIrt):

    def __init__(self, init_slop=None, init_threshold=None, params_type='2PL', constraint=None, *args, **kwargs):
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
        if constraint:
            if 'threshold' in constraint:
                for constraint_threshold in constraint['threshold']:
                    self._params_dt['threshold'][constraint_threshold['item'] - 1] = constraint_threshold['value']
            if self.params_type == '2PL' and 'slop' in constraint:
                for constraint_slop in constraint['slop']:
                    self._params_dt['slop'][constraint_slop['item'] - 1] = constraint_slop['value']

    def _get_jac(self, dp, theta):
        dp_sum = np.sum(dp, axis=0)
        dp_theta = np.sum(dp * theta, axis=0)
        return dp_sum, dp_theta

    def __m_step(self, full_dis, right_dis, theta, p_val, z_val, **params_dt):
        # 一阶导数, 二阶导数
        dp, ddp = self._get_dp_n_ddp(right_dis, full_dis, p_val)
        # jac矩阵和hess矩阵
        # jac = self._get_jac(dp, theta)
        delta_list = np.zeros((self.item_size, 2))
        hess_list = []

        # 把求稀疏矩阵的逆转化成求每个题目的小矩阵的逆
        def _item_loglik(params, instance, index):
            # 似然函数
            slop = params[0]
            threshold = params[1]
            z = instance.z(slop=slop, threshold=threshold, theta=X_NODES)
            p_val = instance.p(z)
            p_val[p_val <= 0] = 1e-10
            p_val[p_val >= 1] = 1 - 1e-10
            loglik_val = np.log(p_val) * right_dis[:, index][:, np.newaxis] + np.log(1 - p_val) * (full_dis - right_dis[:, index][:, np.newaxis])
            return -np.sum(loglik_val)

        def _jac(params, instance, index):
            slop = params[0]
            threshold = params[1]
            z = instance.z(slop=slop, threshold=threshold, theta=X_NODES)
            p_val = instance.p(z)
            dp, ddp = instance._get_dp_n_ddp(right_dis[:, index][:, np.newaxis], full_dis, p_val)
            dp_sum = np.sum(dp)
            dp_theta = np.sum(dp * theta)
            return -np.array([dp_theta, dp_sum])

        slop = params_dt['slop']
        threshold = params_dt['threshold']

        for i in range(self.item_size):
            res = fmin_bfgs(
                _item_loglik,
                np.array([slop[i], threshold[i]]),
                _jac,
                (self, i),
            )
            params_dt['slop'][i] = res[0]
            params_dt['threshold'][i] = res[1]
        return params_dt, delta_list, hess_list

    def _m_step(self, full_dis, right_dis, theta, p_val, z_val, **params_dt):
        # 一阶导数, 二阶导数
        dp, ddp = self._get_dp_n_ddp(right_dis, full_dis, p_val, z_val)
        if self.params_type == '2PL':
            _x = np.concatenate((np.ones((len(theta), 1)), theta), axis=1)
            delta_list = np.zeros((self.item_size, 2))
        else:
            _x = np.ones((len(theta), 1))
            delta_list = np.zeros((self.item_size, 1))
        hess_list = []
        for i in range(self.item_size):
            jac = np.dot(_x.T, dp[:, i, np.newaxis])
            hess = -np.dot(_x.T, _x * ddp[:, i, np.newaxis])
            delta = np.linalg.solve(hess, jac)
            params_dt['threshold'][i] -= delta[0]
            if self.params_type == '2PL':
                params_dt['slop'][i] -= delta[1]
            delta_list[i] = delta[:, 0]
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
            self._params_dt['threshold'] = np.zeros(self.item_size) + 0.1
        if init_guess is not None:
            self._params_dt['guess'] = init_guess
        else:
            self._params_dt['guess'] = np.zeros(self.item_size) + 0.1

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
            dp_guess = np.sum(dp * (1 / (p_guess_val - guess + 1e-10)))
            return -np.array([dp_theta, dp_sum, dp_guess])

        slop = params_dt['slop']
        threshold = params_dt['threshold']
        guess = params_dt['guess']

        v1 = _item_loglik((slop[0], threshold[0], guess[0]), 0)
        v2 = _item_loglik((slop[0], threshold[0], guess[0] + 1e-4), 0)
        d = (v2 - v1) / 1e-4
        jac = _jac((slop[0], threshold[0], guess[0]), 0)

        for i in range(self.item_size):
            res = minimize(
                _item_loglik,
                np.array([slop[i], threshold[i], guess[i]]),
                (i, ),
                'BFGS',
                _jac,
                options={'disp': True},
                tol=1e-5,
            )
            params_dt['slop'][i] = res.x[0]
            params_dt['threshold'][i] = res.x[1]
            params_dt['guess'][i] = res.x[2]
            if params_dt['guess'][i] > 1:
                params_dt['guess'][i] = 1 - 1e-10
            elif params_dt['guess'][i] < 0:
                params_dt['guess'][i] = 1e-10
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


class MCEMIrt(BaseIrt, ProbitMixin):

    def _get_dp_n_ddp(self, right_dis, full_dis, p_val, *args, **kwargs):
        return right_dis - full_dis * p_val, full_dis * p_val * (1 - p_val)

    def __init__(self, dim=1, *args, **kwargs):
        super(MCEMIrt, self).__init__(*args, **kwargs)
        # self.slop, self.threshold = self. _init_value()
        # self.threshold = self.threshold[0]
        self.threshold = np.zeros(self.item_size)
        self.slop = np.ones((dim, self.item_size))
        self.dim = dim

    def _e_step(self, p_val, prior):
        # 计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val) * prior
        # 归一化
        lik_wt_sum = np.sum(lik_wt, axis=0)
        _temp = lik_wt / lik_wt_sum
        # theta的人数分布
        full_dis = np.sum(_temp, axis=1)
        # theta下回答正确的人数分布
        right_dis = np.dot(_temp, scores)
        full_dis.shape = full_dis.shape[0], 1
        # 对数似然值
        loglik_val = np.sum(np.log(lik_wt_sum))
        return full_dis, right_dis, loglik_val

    def _m_step(self, full_dis, right_dis, theta, p_val, z_val, slop, threshold):
        # 一阶导数, 二阶导数
        dp, ddp = self._get_dp_n_ddp(right_dis, full_dis, p_val, z_val)
        _x = np.concatenate((np.ones((len(theta), 1)), theta), axis=1)
        delta_list = np.zeros((self.item_size, 2))
        hess_list = []
        slop_ = slop[:]
        threshold_ = np.zeros_like(threshold)
        for i in range(self.item_size):
            jac = np.dot(_x.T, dp[:, i, np.newaxis])
            hess = -np.dot(_x.T, _x * ddp[:, i, np.newaxis])
            delta = np.linalg.solve(hess, jac)
            threshold_[i] = threshold[i] - delta[0]
            slop_[0][i] = slop[0][i] - delta[1]
            # delta_list[i] = delta[:, 0]
            # hess_list.append(hess)
        return slop_, threshold_

    def fit(self):
        slop = self.slop
        threshold = self.threshold
        for i in range(self._max_iter):
            theta = np.random.normal(size=(50, self.dim))
            prior = 1.0 / 50
            z_val = np.dot(theta, slop) + threshold
            p_val = self.p(z_val)
            full_dis, right_dis, loglik_val = self._e_step(p_val, prior)
            slop_, threshold_ = self._m_step(full_dis, right_dis, theta, p_val, z_val, slop=slop, threshold=threshold)
            c = max(np.mean(np.abs(slop - slop_)), np.mean(np.abs(threshold - threshold_)))
            print(c)
            print(loglik_val)
            if c < self._tol:
                return slop_, threshold_
            slop = slop_
            threshold = threshold_
        return slop, threshold


class LAIrt(BaseIrt, LogitMixin):

    def _get_dp_n_ddp(self, right_dis, full_dis, p_val, *args, **kwargs):
        return right_dis - full_dis * p_val, full_dis * p_val * (1 - p_val)

    def __init__(self, dim=1, *args, **kwargs):
        super(LAIrt, self).__init__(*args, **kwargs)
        # self.slop, self.threshold = self. _init_value()
        # self.threshold = self.threshold[0]
        self.threshold = np.zeros(self.item_size)
        self.slop = np.ones((dim, self.item_size))
        self.dim = dim

    def _e_step(self, slop, threshold):
        p_l = np.zeros((self.person_size, 1))
        theta = np.zeros((self.person_size, 1))
        for i in range(self.person_size):
            mod = BayesLogitModel(slop, threshold, score=self.scores[i])
            theta_ = BayesLogitModel(slop, threshold, score=self.scores[i]).newton
            theta[i] = theta_
            sigma = mod.info(theta_)
            _p_l = (2 * np.pi) ** 0.5 * np.linalg.det(sigma) ** -0.5 * np.exp(mod.loglik(theta_))
            p_l[i] = _p_l

        return full_dis, right_dis, loglik_val

    def _m_step(self, full_dis, right_dis, theta, p_val, z_val, slop, threshold):
        # 一阶导数, 二阶导数
        dp, ddp = self._get_dp_n_ddp(right_dis, full_dis, p_val, z_val)
        _x = np.concatenate((np.ones((len(theta), 1)), theta), axis=1)
        delta_list = np.zeros((self.item_size, 2))
        hess_list = []
        slop_ = slop[:]
        threshold_ = np.zeros_like(threshold)
        for i in range(self.item_size):
            jac = np.dot(_x.T, dp[:, i, np.newaxis])
            hess = -np.dot(_x.T, _x * ddp[:, i, np.newaxis])
            delta = np.linalg.solve(hess, jac)
            threshold_[i] = threshold[i] - delta[0]
            slop_[0][i] = slop[0][i] - delta[1]
            # delta_list[i] = delta[:, 0]
            # hess_list.append(hess)
        return slop_, threshold_

    def fit(self):
        slop = self.slop.T
        threshold = self.threshold
        for i in range(self._max_iter):
            full_dis, right_dis, loglik_val = self._e_step(slop, threshold)
            slop_, threshold_ = self._m_step(full_dis, right_dis, theta, p_val, z_val, slop=slop, threshold=threshold)
            c = max(np.mean(np.abs(slop - slop_)), np.mean(np.abs(threshold - threshold_)))
            print(c)
            print(loglik_val)
            if c < self._tol:
                return slop_, threshold_
            slop = slop_
            threshold = threshold_
        return slop, threshold


class MCEMIrt1(BaseIrt, ProbitMixin):

    def __init__(self, dim=1, *args, **kwargs):
        super(MCEMIrt1, self).__init__(*args, **kwargs)
        self.threshold = np.zeros(self.item_size)
        self.slop = np.ones((dim, self.item_size))
        self.dim = dim
        # self.slop, self.threshold = self. _init_value()
        # self.threshold = self.threshold[0]

    def _e_step(self, slop, threshold, point_size=25, burn=10, thin=5):
        theta = np.random.normal(size=(self.dim, self.person_size))
        v = np.linalg.inv(np.dot(slop, slop.T) + np.eye(self.dim))
        t_z = 0
        t_x = 0
        z_t_z = 0
        z_t_x = 0
        theta_ar = np.zeros((point_size, self.dim, self.person_size))
        x_ar = np.zeros((point_size, self.person_size, self.item_size))
        e = 0
        d = 0
        for i in range(point_size * thin + burn):
            z_val = np.dot(theta.T, slop) + threshold
            # z_val[z_val > 12] = 12
            # z_val[z_val < -12] = -12
            lower = np.zeros_like(self.scores, dtype=np.float)
            lower[self.scores == 0] = -np.inf
            lower[self.scores == 1] = -z_val[self.scores == 1]
            upper = np.zeros_like(self.scores, dtype=np.float)
            upper[self.scores == 1] = np.inf
            upper[self.scores == 0] = -z_val[self.scores == 0]
            x = truncnorm.rvs(lower, upper, loc=z_val)
            loc = np.dot(np.dot(v, slop), (x - threshold).T)
            theta = np.random.normal(loc, v ** 0.5)
            if i >= burn and (i - burn) % thin == 0:
                t_z += np.sum(theta, keepdims=True)
                t_x += np.sum(x, axis=0, keepdims=True)
                z_t_z += theta.dot(theta.T)
                z_t_x += theta.dot(x)
                theta_ar[int((i - burn) / thin)] = theta
                x_ar[int((i - burn) / thin)] = x
                e += x - threshold
                d += ((x - threshold) ** 2).sum(axis=1)
        e /= point_size
        d /= point_size
        my_t_x = np.sum(e + threshold, axis=0)
        my_t_z = (np.sum(e, axis=0).dot(slop.T)).dot(v.T)
        my_z_t_z = v + v.dot(slop).dot(np.sum(d)).dot(slop.T).dot(v.T)
        my_z_t_x = v.dot(slop).dot(np.sum(d + e.dot(threshold.T), axis=0))
        # plt.plot(theta_ar.mean(0)[0])
        # plt.show()
        t_z /= point_size
        t_x /= point_size
        z_t_z /= point_size
        z_t_x /= point_size
        return t_z, t_x, z_t_z, z_t_x, theta_ar

    def ___m_step(self, x):
        _x = x.mean(0)
        _x = sm.add_constant(_x[0])
        slop = np.zeros_like(self.slop)
        threshold = np.zeros_like(self.threshold)
        for i in range(self.item_size):
            _y = self.scores[:, i]
            threshold[i], slop[:, i] = sm.GLM(_y, _x, family=sm.families.Binomial(link=probit)).fit().params
        return slop, threshold

    def __m_step(self, x, pt_size=20):
        slop = np.zeros_like(self.slop)
        threshold = np.zeros_like(self.threshold)
        person_size = self.person_size
        _x = np.zeros((1, person_size * pt_size))
        _y = np.zeros(person_size * pt_size)
        for i in range(self.item_size):
            for j in range(pt_size):
                _x[:, j * person_size: (j + 1) * person_size] = x[j]
                _y[j * person_size: (j + 1) * person_size] = self.scores[:, i]
            _x = sm.add_constant(_x[0])
            threshold[i], slop[:, i] = sm.GLM(_y, _x, family=sm.families.Binomial(link=probit)).fit().params
            _x = np.zeros((1, person_size * pt_size))
            _y = np.zeros(person_size * pt_size)
        return slop, threshold

    def _m_step(self, t_z, t_x, z_t_z, z_t_x):
        temp1 = np.linalg.inv(z_t_z - np.dot(t_z.T, t_z) / self.person_size)
        temp2 = z_t_x - np.dot(t_z.T, t_x) / self.person_size
        slop = np.dot(temp1, temp2)
        threshold = (t_x - np.dot(t_z, slop))/ self.person_size
        return slop, threshold

    def fit(self):
        slop = self.slop
        threshold = self.threshold
        for i in range(self._max_iter):
            t_z, t_x, z_t_z, z_t_x, x_ar = self._e_step(slop, threshold)
            slop, threshold = self.__m_step(x_ar)
            # slop, threshold = self.___m_step(x_ar)
            # slop, threshold = self._m_step(t_z, t_x, z_t_z, z_t_x)
            # threshold = threshold[0]
            slop = slop
            print(slop)
            print(threshold)
        return slop, threshold


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
                 max_iter=1000, tol=1e-5, constraint=None):
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
        if constraint:
            kwargs['constraint'] = constraint
        self._model = _link(**kwargs)

    def fit(self):
        return self._model.fit()
