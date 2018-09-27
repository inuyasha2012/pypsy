# coding=utf-8
from scipy.stats import norm
import numpy as np


class ZMixin(object):

    def z(self, slop, threshold, theta):
        """

        :param slop:
        :param threshold:
        :param theta:
        :return:
        """
        return slop * theta + threshold


class RaschZMixin(object):

    def z(self, threshold, theta):
        # z函数
        return theta + threshold


class ProbitMixin(object):

    def p(self, z):
        # 回答正确的概率函数
        return norm.cdf(z)


class LogitMixin(object):

    def p(self, z):
        # 回答正确的概率函数
        e = np.exp(z)
        p_val = e / (1.0 + e)
        return p_val


class GuessLogitMixin(LogitMixin):

    def p(self, guess, *args, **kwargs):
        p_val = super(GuessLogitMixin, self).p(*args, **kwargs)
        return guess + (1 - guess) * p_val


class GuessProbitMixin(ProbitMixin):

    def p(self, guess, *args, **kwargs):
        p_val = super(GuessProbitMixin, self).p(*args, **kwargs)
        return guess + (1 - guess) * p_val

