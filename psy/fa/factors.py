# coding=utf-8
import numpy as np
from psy.utils import cached_property


class Factor(object):

    # 简单的因子分析，服务于mirt的初值估计

    def __init__(self, scores, factors_num, cov_mat_type='cor'):
        self._scores = scores
        self._factors_num = factors_num
        self._cov_mat_type = cov_mat_type

    @cached_property
    def cor(self):
        scores_cor = np.corrcoef(self._scores)
        return scores_cor

    @cached_property
    def polycor(self):
        # 伪polycor
        return np.abs(self.cor) ** (1 / 1.15) * np.sign(self.cor)

    @property
    def loadings(self):
        cov = getattr(self, self._cov_mat_type)
        score_eig = np.linalg.eig(cov)
        loadings = -1 * score_eig[1][:, :self._factors_num]
        return loadings
