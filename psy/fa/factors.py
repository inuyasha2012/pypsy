# coding=utf-8
import numpy as np
from psy.utils import cached_property


class Factor(object):

    # 简单的因子分析，服务于mirt的初值估计

    def __init__(self, scores, factors_num):
        self._scores = scores
        self._factors_num = factors_num

    @cached_property
    def cor(self):
        # 相关矩阵
        scores_cor = np.corrcoef(self._scores, rowvar=False)
        return scores_cor

    @cached_property
    def polycor(self):
        # 伪polycor
        return np.abs(self.cor) ** (1 / 1.15) * np.sign(self.cor)

    @property
    def mirt_loading(self):
        cov = self.polycor
        score_eig = self._get_eigen(cov)
        loadings = score_eig[1][:, :self._factors_num]
        return loadings

    @staticmethod
    def _get_eigen(cov):
        score_eig = np.linalg.eig(cov)
        idx = score_eig[0].argsort()
        eigenvalues = score_eig[0][idx][::-1]
        _eigenvectors = score_eig[1][:, idx][:, ::-1]
        eigenvectors = _eigenvectors * np.sign(np.sum(_eigenvectors, 0))
        return eigenvalues, eigenvectors

    @property
    def loadings(self):
        # 因子载荷
        cov = self.cor
        score_eig = self._get_eigen(cov)
        _loadings = score_eig[0] ** 0.5 * score_eig[1]
        loadings = _loadings[:, :self._factors_num]
        return loadings
