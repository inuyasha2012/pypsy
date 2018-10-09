# coding=utf-8
from scipy.stats import norm
import numpy as np


class _ZMixin(object):

    def z(self, theta, threshold, slop=None):
        """
        z function
        :param slop:
        :param threshold:
        :param theta:
        :return:
        """
        if slop is None:
            return theta + threshold
        return slop * theta + threshold


class _MirtZMixin(object):

    def z(self, slop, threshold, theta):
        z_val = np.dot(theta, slop) + threshold
        z_val[z_val > 35] = 35
        z_val[z_val < -35] = -35
        return z_val


class _ProbitMixin(object):
    """
    probit模型
    """

    def p(self, z):
        # 回答正确的概率函数
        return norm.cdf(z)


class _LogitMixin(object):
    """
    logit模型
    """

    def p(self, z):
        # 回答正确的概率函数
        e = np.exp(z)
        p_val = e / (1.0 + e)
        return p_val


class _GuessMixin(object):
    """
    带有猜测参数的模型
    """
    def p_guess(self, guess, *args, **kwargs):
        z_val = self.z(*args, **kwargs)
        p_val = self.p(z_val)
        return {'p_guess_val': guess + (1 - guess) * p_val, 'p_val': p_val}


class LogitMixin(_LogitMixin, _ZMixin):
    """
    1参数和2参数logit模型
    """


class ProbitMixin(_ProbitMixin, _ZMixin):
    """
    1参数和2参数probit模型
    """


class Logit3PLMixin(_LogitMixin, _GuessMixin, _ZMixin):
    """
    3参数logit模型mixin
    """


class Probit3PLMixin(_ProbitMixin, _GuessMixin, _ZMixin):
    """
    3参数probit模型mixin
    """


class MirtLogit2PLMixin(_LogitMixin, _MirtZMixin):
    """
    2参数多维logit模型
    """


class BaseEmIrt(object):

    def __init__(self, scores=None, max_iter=1000, tol=1e-5):
        self.scores = scores
        self._max_iter = max_iter
        self._tol = tol
        self.item_size = scores.shape[1]

    def _e_step(self, p_val, weights):
        # 计算theta的分布人数
        scores = self.scores
        lik_wt = self._lik(p_val) * weights
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

    def _lik(self, p_val):
        # 似然函数
        scores = self.scores
        p_val[p_val <= 0] = 1e-10
        p_val[p_val >= 1] = 1 - 1e-10
        loglik_val = np.dot(np.log(p_val), scores.transpose()) + \
                     np.dot(np.log(1 - p_val), (1 - scores).transpose())
        return np.exp(loglik_val)
