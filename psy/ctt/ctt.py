# coding=utf-8
from __future__ import division, print_function
import numpy as np

from psy import Factor


class BaseCtt(object):

    def __init__(self, scores):
        self._scores = scores
        self.sum_scores = np.sum(scores, axis=1)
        self.sum_scores.shape = self.sum_scores.shape[0], 1
        self.item_size = scores.shape[1]

    def get_composite_reliability(self):
        # 组合信度
        f = Factor(self._scores, 1)
        loadings = f.loadings
        lambda_sum_square = np.sum(loadings) ** 2
        lambda_square_sum = np.sum(loadings ** 2)
        return lambda_sum_square / (lambda_sum_square - lambda_square_sum + self.item_size)

    def get_alpha_reliability(self):
        scores = self._scores
        item_size = self.item_size
        # 每道试题的方差
        items_var = np.var(scores, axis=0)
        # 所有试题方差的和
        sum_items_var = np.sum(items_var)
        # 计算总分方差
        sum_scores_var = np.var(self.sum_scores)
        return item_size / (item_size - 1) * (1 - sum_items_var / sum_scores_var)


class BivariateCtt(BaseCtt):

    def get_discrimination(self):
        # 区分度
        scores = self._scores
        # 题目分数均值
        scores_mean = np.mean(scores, axis=0)
        # 总分均值
        sum_scores_mean = np.mean(self.sum_scores)
        # 中心化
        center = (scores - scores_mean) * (self.sum_scores - sum_scores_mean)
        # 协方差
        cov = np.mean(center, axis=0)
        # 方差
        std = np.std(scores, axis=0) * np.std(self.sum_scores)
        return cov / std

    def get_difficulty(self):
        # 难度(通俗度)
        return np.mean(self._scores, axis=0)

