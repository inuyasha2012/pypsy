# coding=utf-8
from __future__ import print_function
import numpy as np
from scipy.stats import norm
from psy.exceptions import ConvergenceError, ItemParamError, ScoreError, ThetaError, IterMethodError, UnknownModelError
from psy.utils import cached_property, gen_item_bank
import math

# TODO FIX 奇异矩阵


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

    def __init__(self, slop, threshold, init_theta=None, score=None, iter_method='newton'):
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
        # TODO 改为根据样本估计
        self._inv_psi = np.identity(len(self._slop[0]))
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
    D = 1.702

    def _prob(self, theta):
        """
        答1的概率值
        :param theta: ndarray(int|float), 特质向量初值
        :return:  ndarray(float)，作答为1概率的向量值
        """
        e = np.exp(self.D * self._z(theta))
        return e / (1.0 + e)

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
        hes = np.linalg.pinv(self._ddloglik(theta, prob_val))
        jac = self._dloglik(theta, prob_val)
        return hes, jac

    def _get_jacobian(self, theta):
        prob_val = self._prob(theta)
        return self._dloglik(theta, prob_val)


class BayesLogitModel(BaseLogitModel):

    # 贝叶斯modal估计

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
        hes = np.linalg.inv(self._ddloglik(theta, w))
        jac = self._dloglik(theta, prob_val, h, w)
        return hes, jac

    def _get_jacobian(self, theta):
        h, prob_val, w = self._get_h_prob_val_w(theta)
        return self._dloglik(theta, prob_val, h, w)


class BayesProbitModel(BaseProbitModel):

    # 贝叶斯modal估计

    def _bayes_dloglik(self, theta, prob_val, h, w):
        return self._dloglik(theta, prob_val, h, w) - theta

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


class BaseSimTirt(object):

    MODEL = {'bayes_probit': BayesProbitModel, 'ml_probit': MLProbitModel,
             'bayes_logit': BayesLogitModel, 'ml_logit': MLLogitModel}

    def __init__(self, subject_nums, trait_size, model='ml_logit',
                 iter_method='newton', block_size=3, lower=1, upper=4, avg=0, std=1):
        """

        :param subject_nums: int, 模拟被试的人数
        :param trait_size: int, 特质数量
        :param iter_method: str
        :param model: str, 模型
        :param block_size: int, 题块 
        :param lower: int|float
        :param upper: int|float
        :param avg: int|float
        :param std: int|float
        """
        if not isinstance(subject_nums, int):
            raise ValueError('subject_nums must be int')
        if not isinstance(trait_size, int):
            raise ValueError('trait_size must be int')
        if model not in ('bayes_probit', 'bayes_logit', 'ml_probit', 'ml_logit'):
            raise ValueError('mode must be bayes_probit or bayes_logit or ml_probit or ml_logit')
        if block_size not in (2, 3):
            raise ValueError('block_size must be 2 or 3')
        if not isinstance(lower, (int, float)):
            raise ValueError('lower must be int or float')
        if not isinstance(upper, (int, float)):
            raise ValueError('upper must be int or float')
        if not isinstance(avg, (int, float)):
            raise ValueError('avg must be int or float')
        if not isinstance(std, (int, float)):
            raise ValueError('std must be int or float')
        if iter_method not in ('newton', 'gradient_ascent'):
            raise IterMethodError('iter_method must be newton or gradient_ascent')

        self._subject_nums = subject_nums
        self._trait_size = trait_size
        self._block_size = block_size
        self._lower = lower
        self._upper = upper
        self._avg = avg
        self.std = std
        self._iter_method = iter_method
        self._model = self._get_model(model)

    def _get_model(self, model):
        try:
            return self.MODEL[model]
        except KeyError:
            raise UnknownModelError('unknown model, must be "bayes_probit" or '
                                    '"ml_probit" or "bayes_logit" or "ml_logit"')

    @cached_property
    def random_thetas(self):
        """
        生成特质向量
        :return: ndarray
        """
        return np.random.multivariate_normal(np.zeros(self._trait_size),
                                             np.identity(self._trait_size), self._subject_nums)

    def _get_init_theta(self):
        return np.zeros(self._trait_size)

    def _get_mean_error(self, theta_list):
        return np.mean(np.abs(theta_list - self.random_thetas))


class SimAdaptiveTirt(BaseSimTirt):

    def __init__(self, item_size, max_sec_item_size=10, *args, **kwargs):
        super(SimAdaptiveTirt, self).__init__(*args, **kwargs)
        # 题库题量
        self._item_size = item_size
        # 已做答试题编号保存记录
        self._has_answered_item_idx = {}
        # 已做答得分保存记录
        self._score = {}
        # 参数估计保存记录
        self._theta = {}
        # 作答试题斜率保存记录
        self._slop = {}
        # 作答试题阈值保存记录
        self._threshold = {}
        # 第二阶段最大答题次数
        self._max_sec_item_size = max_sec_item_size

    @property
    def scores(self):
        return self._score

    @property
    def thetas(self):
        return self._theta

    def _add_slop(self, theta_idx, slop):
        if theta_idx in self._slop:
            self._slop[theta_idx] = np.concatenate((self._slop[theta_idx], slop))
        else:
            self._slop[theta_idx] = slop

    def _get_slop(self, theta_idx):
        return self._slop[theta_idx]

    def _get_threshold(self, theta_idx):
        return self._threshold[theta_idx]

    def _add_threshold(self, theta_idx, threshold):
        if theta_idx in self._threshold:
            self._threshold[theta_idx] = np.concatenate((self._threshold[theta_idx], threshold))
        else:
            self._threshold[theta_idx] = threshold

    def _add_answered_item_idx(self, theta_idx, used_item_idx_list):
        if theta_idx in self._has_answered_item_idx:
            self._has_answered_item_idx[theta_idx].extend(used_item_idx_list)
        else:
            self._has_answered_item_idx[theta_idx] = used_item_idx_list

    def _get_answered_item_idx_set(self, theta_idx):
        return set(self._has_answered_item_idx[theta_idx])

    def _get_can_use_items(self, theta_idx):
        can_use_idx = self._get_can_use_idx(theta_idx)
        return self.item_bank[list(can_use_idx)]

    def _get_can_use_idx(self, theta_idx):
        can_use_idx = self._item_idx_set - self._get_answered_item_idx_set(theta_idx)
        return can_use_idx

    def _add_score(self, theta_idx, score):
        if theta_idx in self._score:
            self._score[theta_idx] = np.concatenate((self._score[theta_idx], score))
        else:
            self._score[theta_idx] = score

    def _get_score(self, theta_idx):
        return self._score[theta_idx]

    def _add_theta(self, theta_idx, theta):
        if theta_idx in self._theta:
            self._theta[theta_idx].append(theta)
        else:
            self._theta[theta_idx] = [theta]

    def _get_theta(self, theta_idx):
        return self._theta[theta_idx][-1]

    @cached_property
    def item_bank(self):
        return gen_item_bank(self._trait_size, self._item_size, self._block_size)

    @cached_property
    def _item_idx_set(self):
        return set(range(self._item_size))

    def _get_random_choice_items(self, theta_idx):
        rand_choice_size = self._get_random_choice_size()
        while True:
            items = []
            dims = []
            used_idx_list = []
            idx_list = np.random.choice(list(self._item_idx_set), rand_choice_size, False)
            for i in idx_list:
                item = self.item_bank[i]
                items.append(item)
                dims.extend(item['dim'])
                used_idx_list.append(i)
            # if len(set(dims)) == self._trait_size:
            self._add_answered_item_idx(theta_idx, used_idx_list)
            return items

    def _get_random_choice_size(self):
        return int(math.ceil(1.0 * self._trait_size / self._block_size))

    def _get_random_choice_params(self, theta_idx):
        first_rand_items = self._get_random_choice_items(theta_idx)
        slop = []
        threshold = []
        for item in first_rand_items:
            slop.extend(item['params'][0])
            threshold.extend(item['params'][1])
        return np.array(slop), np.array(threshold)

    def _first_random(self, theta, theta_idx):
        # 第一阶段，随机抽题
        slop, threshold = self._get_random_choice_params(theta_idx)
        p_list = self._model(slop, threshold).prob(theta)
        score = np.random.binomial(1, p_list, len(p_list))
        init_theta = self._get_init_theta()
        model = self._model(slop, threshold, init_theta, score, self._iter_method)
        theta = model.solve
        self._add_score(theta_idx, score)
        self._add_theta(theta_idx, theta)
        self._add_slop(theta_idx, slop)
        self._add_threshold(theta_idx, threshold)

    def _second_random(self, theta, theta_idx):
        item = self._get_next_item(theta_idx)

        score = self._get_next_score(item, theta, theta_idx)
        # print score

        est_theta = self._get_estimate_theta(score, theta_idx)
        # print est_theta
        # print np.mean(np.abs(est_theta - theta))
        self._add_theta(theta_idx, est_theta)
        return est_theta

    def _get_estimate_theta(self, score, theta_idx):
        # 参数估计
        now_slop = self._get_slop(theta_idx)
        now_threshold = self._get_threshold(theta_idx)
        init_theta = self._get_init_theta()
        model = self._model(now_slop, now_threshold, init_theta, score, self._iter_method)
        est_theta = model.solve
        return est_theta

    def _get_next_score(self, item, theta, theta_idx):
        # 模拟自适应抽题的下一题得分
        item_slop = item['params'][0]
        self._add_slop(theta_idx, item_slop)
        item_threshold = item['params'][1]
        self._add_threshold(theta_idx, item_threshold)
        p_list = self._model(item_slop, item_threshold).prob(theta)
        item_score = np.random.binomial(1, p_list, len(p_list))
        self._add_score(theta_idx, item_score)
        score = self._get_score(theta_idx)
        return score

    def _get_next_item(self, theta_idx):
        # 获得自适应抽题的下一道题
        est_theta = self._get_theta(theta_idx)
        items = self._get_can_use_items(theta_idx)
        slop = self._get_slop(theta_idx)
        threshold = self._get_threshold(theta_idx)
        test_info = self._model(slop, threshold).info(est_theta)
        info_list = []
        for _item in items:
            _slop, _threshold = _item['params']
            item_info = self._model(_slop, _threshold).info(est_theta)
            info_list.append(np.linalg.det(test_info + item_info))
        max_info_idx = np.array(info_list).argmax()
        item = items[max_info_idx]
        idx = list(self._get_can_use_idx(theta_idx))[max_info_idx]
        self._add_answered_item_idx(theta_idx, [idx])
        return item

    def sim(self):
        thetas = self.random_thetas
        theta_list = []
        for i, theta in enumerate(thetas):
            try:
                est_theta = np.nan
                self._first_random(theta, i)
                for j in range(self._max_sec_item_size):
                    est_theta = self._second_random(theta, i)
                print(u'第{0}个被试模拟成功！'.format(i + 1))
            except Exception as e:
                print(e)
                continue
            theta_list.append(est_theta)
        mean_error = self._get_mean_error(np.array(theta_list))
        print('模拟结束，平均误差{0}'.format(mean_error))
        return theta_list

