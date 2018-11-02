# coding=utf-8
from __future__ import unicode_literals, print_function, absolute_import
import numpy as np
from scipy.stats import norm
from psy.exceptions import ConvergenceError
from psy.exceptions.cat import ItemParamError, ScoreError, ThetaError, IterMethodError
from psy.utils import cached_property


class BaseModel(object):

    # 最大牛顿迭代次数
    _newton_max_iter = 1000
    # 牛顿迭代步长
    _newton_step_size = 0.1
    # 最大梯度上升次数
    _gradient_max_iter = 1000
    # 梯度上升步长
    _gradient_step_size = 0.01
    # 参数估计精度
    _tol = 1e-5

    def __init__(self, slop, threshold, init_theta=None, score=None, iter_method='newton', sigma=None):
        """
        不管是probit还是logit，都是用一样的参数估计算法，
        基于牛顿迭代的极大似然算法和贝叶斯最大后验算法
        :param slop: ndarray(float), 多维向量，斜率，区分度
        :param threshold: ndarray(float), 单维向量，阈值，通俗度，难度
        :param init_theta: ndarray(int|float), 特质向量初值
        :param score:  ndarray(0|1), 得分向量
        """
        if not isinstance(slop, np.ndarray):
            raise ItemParamError('item param must be ndarray')
        if not isinstance(threshold, np.ndarray):
            raise ItemParamError('item param must be ndarray')

        if len(slop.shape) == 1:
            slop.shape = 1, slop.shape[0]

        if len(slop) != len(threshold):
            raise ItemParamError('item param must be same length')

        if score is not None:
            if not isinstance(score, np.ndarray):
                raise ScoreError('score must be ndarray')
            if len(score) != len(slop):
                raise ScoreError('score must be same length as item param')

        if init_theta is not None and not isinstance(init_theta, np.ndarray):
            raise ThetaError('init_theta must be ndarray')

        if iter_method not in ('newton', 'gradient_ascent'):
            raise IterMethodError('iter_method must be newton or gradient_ascent')

        self._slop = slop
        self._score = score
        self._threshold = threshold
        self._init_theta = init_theta if init_theta is not None else np.zeros(len(self._slop[0]))
        # 默认bayes先验正态分布标准差
        if sigma is None:
            self._inv_psi = np.identity(len(self._slop[0]))
        else:
            self._inv_psi = np.linalg.inv(sigma)
        self._iter_method = iter_method

    @property
    def score(self):
        return self._score

    def _prob(self, theta):
        raise NotImplementedError

    def prob(self, theta):
        # 回答为1的概率
        if not isinstance(theta, np.ndarray):
            raise ThetaError('theta must be ndarray')
        return self._prob(theta)

    def _z(self, theta):
        """
        probit和logit的z值
        :param theta: ndarray(int|float), 特质向量初值
        :return: ndarray(float), z值向量
        """
        return np.sum(self._slop * theta, 1) - self._threshold

    def z(self, theta):
        if not isinstance(theta, np.ndarray):
            raise ThetaError('theta must be ndarray')
        return self._z(theta)

    def _get_hessian_n_jacobian(self, theta):
        """
        抽象方法，目的是返回海塞矩阵和雅克比一阶导数向量，用于牛顿迭代
        :param theta: ndarray(int|float), 特质向量初值
        """
        raise NotImplementedError

    def _get_jacobian(self, theta):
        """
        抽象方法，目的是返回雅克比向量（矩阵），用于梯度上升
        :param theta:
        """
        raise NotImplementedError

    @property
    def newton(self):
        """
        基于牛顿迭代的参数估计
        :return: ndarray(int|float), 特质向量初值
        """
        theta0 = self._init_theta * 1.0
        for i in range(self._newton_max_iter):
            hes, jac = self._get_hessian_n_jacobian(theta0)
            temp = self._newton_step_size * np.dot(hes, jac)
            theta = theta0 - temp
            if np.max(np.abs(temp)) < self._tol:
                # print i
                return np.round(theta, 3)
            theta0 = theta
        raise ConvergenceError('no convergence')

    @property
    def gradient_ascent(self):
        # 梯度上升参数估计
        theta0 = self._init_theta * 1.0
        for i in range(self._gradient_max_iter):
            jac = self._get_jacobian(theta0)
            theta = theta0 + self._gradient_step_size * jac
            if np.max(np.abs(self._gradient_step_size * jac)) < self._tol:
                return np.round(theta, 3)
            theta0 = theta
        raise ConvergenceError('no convergence')

    @cached_property
    def solve(self):
        return getattr(self, self._iter_method)


class BaseLogitModel(BaseModel):

    # D值
    D = 1

    def _prob(self, theta):
        """
        答1的概率值
        :param theta: ndarray(int|float), 特质向量初值
        :return:  ndarray(float)，作答为1概率的向量值
        """
        e = np.exp(self.D * self._z(theta))
        return e / (1.0 + e)

    def loglik(self, theta):
        # 似然函数
        p_val = self._prob(theta)
        p_val[p_val <= 0] = 1e-10
        p_val[p_val >= 1] = 1 - 1e-10
        loglik_val = np.dot(np.log(p_val), self.score.transpose()) + np.dot(np.log(1 - p_val), (1 - self.score).transpose())
        return loglik_val

    def _dloglik(self, theta, prob_val):
        """
        logistic对数似然函数的一阶导数
        :param theta: ndarray(int|float), 特质向量初值
        :param prob_val: ndarray(float)，作答为1概率的向量值
        :return:
        """
        return self.D * np.dot(self._slop.transpose(), self._score - prob_val)

    def _expect(self, prob_val):
        return self.D ** 2 * np.dot(self._slop.transpose() * prob_val * (1 - prob_val), self._slop)

    def _ddloglik(self, theta, prob_val):
        """
        logistic对数似然函数的二阶导数
        :param theta: ndarray(int|float), 特质向量初值
        :param prob_val: ndarray(float)，作答为1概率的向量值
        :return:
        """
        return -1 * self._expect(prob_val)

    def info(self, theta):
        # 信息矩阵
        if not isinstance(theta, np.ndarray):
            raise ThetaError('theta must be ndarray')
        prob_val = self._prob(theta)
        return self._expect(prob_val)


class MLLogitModel(BaseLogitModel):

    # 极大似然估计

    def _get_hessian_n_jacobian(self, theta):
        prob_val = self._prob(theta)
        hes = np.linalg.inv(self._ddloglik(theta, prob_val))
        jac = self._dloglik(theta, prob_val)
        return hes, jac

    def _get_jacobian(self, theta):
        prob_val = self._prob(theta)
        return self._dloglik(theta, prob_val)


class BayesLogitModel(BaseLogitModel):

    # 贝叶斯modal估计

    def loglik(self, theta):
        return super(BayesLogitModel, self).loglik(theta) - theta ** 2 * 0.5

    def _bayes_dloglik(self, theta, prob_val):
        # 贝叶斯modal的一阶导数
        return self._dloglik(theta, prob_val) - theta

    def _bayes_ddloglik(self, theta, prob_val):
        # 贝叶斯modal的二阶导数
        return self._ddloglik(theta, prob_val) - self._inv_psi

    def _get_hessian_n_jacobian(self, theta):
        prob_val = self._prob(theta)
        hes = np.linalg.inv(self._bayes_ddloglik(theta, prob_val))
        jac = self._bayes_dloglik(theta, prob_val)
        return hes, jac

    def _get_jacobian(self, theta):
        prob_val = self._prob(theta)
        return self._bayes_dloglik(theta, prob_val)

    def info(self, theta):
        # 信息矩阵
        if not isinstance(theta, np.ndarray):
            raise ThetaError('theta must be ndarray')
        _info = super(BayesLogitModel, self).info(theta)
        return _info + self._inv_psi


class BaseProbitModel(BaseModel):

    # probit基础模型

    def _prob(self, theta):
        # probit概率值
        return norm.cdf(self._z(theta))

    def _h(self, z):
        # probit函数的h值，方便计算
        return (1.0 / ((2 * np.pi) ** 0.5)) * np.exp(-1 * z ** 2 / 2.0)

    def _w(self, h, prob):
        # probit函数的w值，可以看成权重，方便计算和呈现

        pq = (1 - prob) * prob
        return h ** 2 / (pq + 1e-10)

    def _get_h_prob_val_w(self, theta):
        z = self._z(theta)
        h = self._h(z) + 1e-10
        prob_val = self._prob(theta)
        w = self._w(h, prob_val)
        return h, prob_val, w

    def _dloglik(self, theta, prob_val, h, w):
        # probit一阶导数
        return np.dot(self._slop.transpose(), w * (self._score - prob_val) / h)

    def _ddloglik(self, theta, w):
        # probit二阶导数
        return -1 * np.dot(self._slop.transpose() * w, self._slop)

    def info(self, theta):
        # 信息矩阵
        if not isinstance(theta, np.ndarray):
            raise ThetaError('theta must be ndarray')
        h, prob_val, w = self._get_h_prob_val_w(theta)
        return np.dot(self._slop.transpose() * w, self._slop)


class MLProbitModel(BaseProbitModel):

    # probit极大似然估计

    def _get_hessian_n_jacobian(self, theta):
        h, prob_val, w = self._get_h_prob_val_w(theta)
        hes = np.linalg.pinv(self._ddloglik(theta, w))
        jac = self._dloglik(theta, prob_val, h, w)
        return hes, jac

    def _get_jacobian(self, theta):
        h, prob_val, w = self._get_h_prob_val_w(theta)
        return self._dloglik(theta, prob_val, h, w)


class BayesProbitModel(BaseProbitModel):

    # 贝叶斯modal估计

    def _bayes_dloglik(self, theta, prob_val, h, w):
        return self._dloglik(theta, prob_val, h, w) - np.dot(self._inv_psi, theta)

    def _bayes_ddloglik(self, theta, w):
        return self._ddloglik(theta, w) - self._inv_psi

    def _get_hessian_n_jacobian(self, theta):
        h, prob_val, w = self._get_h_prob_val_w(theta)
        hes = np.linalg.inv(self._bayes_ddloglik(theta, w))
        jac = self._bayes_dloglik(theta, prob_val, h, w)
        return hes, jac

    def _get_jacobian(self, theta):
        h, prob_val, w = self._get_h_prob_val_w(theta)
        return self._bayes_dloglik(theta, prob_val, h, w)

    def info(self, theta):
        # 信息矩阵
        if not isinstance(theta, np.ndarray):
            raise ThetaError('theta must be ndarray')
        _info = super(BayesProbitModel, self).info(theta)
        return _info + self._inv_psi


def estimate_theta(slop, threshold, score, init_theta=None):
    # TODO 参数估计API
    pass
