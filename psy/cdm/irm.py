# coding=utf-8
from itertools import product
import numpy as np
import progressbar
from psy.exceptions import ConvergenceError
from psy.utils import cached_property, get_log_beta_pd, get_log_normal_pd, get_log_lognormal_pd


class Dina(object):

    def __init__(self, attrs, score=None):
        self._attrs = attrs
        self._score = score

    @cached_property
    def _people_size(self):
        return self._score.shape[0]

    @cached_property
    def item_size(self):
        # 题量
        return self._attrs.shape[1]

    @cached_property
    def _skills_size(self):
        return self._attrs.shape[0]

    def get_yita(self, skills):
        # dina模型下的yita值
        _yita = np.dot(skills, self._attrs)
        _aa = np.sum(self._attrs * self._attrs, axis=0)
        _yita[_yita < _aa] = 0
        _yita[_yita == _aa] = 1
        return _yita

    @staticmethod
    def _get_p(yita, no_slip, guess):
        # dina模型下的答题正确的概率值
        return no_slip ** yita * guess ** (1 - yita)

    def get_p(self, yita, no_slip, guess):
        p_val = self._get_p(yita, no_slip, guess)
        p_val[p_val <= 0] = 1e-10
        p_val[p_val >= 1] = 1 - 1e-10
        return p_val


class BaseEmDina(Dina):

    def _loglik(self, p_val):
        # dina模型的对数似然函数
        log_p_val = np.log(p_val)
        log_q_val = np.log(1 - p_val)
        score = self._score
        return np.dot(log_p_val, score.transpose()) + np.dot(log_q_val, (1 - score).transpose())

    def _get_all_skills(self):
        size = self._skills_size
        return np.array(list(product([0, 1], repeat=size)))


class MlDina(BaseEmDina):

    def __init__(self, guess, no_slip, *args, **kwargs):
        super(MlDina, self).__init__(*args, **kwargs)
        self._guess = guess
        self._no_slip = no_slip

    def solve(self):
        skills = self._get_all_skills()
        yita = self.get_yita(skills)
        p_val = self.get_p(yita, self._no_slip, self._guess)
        loglik = self._loglik(p_val)
        return skills[loglik.argmax(axis=0)]


class EmDina(BaseEmDina):

    def __init__(self, guess_init=None, no_slip_init=None, max_iter=100, tol=1e-5, *args, **kwargs):
        super(EmDina, self).__init__(*args, **kwargs)
        self._skills = self._get_all_skills()
        self._no_slip_init = np.zeros((1, self.item_size)) + 0.7 if no_slip_init is None else no_slip_init
        self._guess_init = np.zeros((1, self.item_size)) + 0.3 if guess_init is None else guess_init
        self._max_iter = max_iter
        self._tol = tol

    def _posterior(self, p_val):
        attr_size = self._skills_size
        return np.exp(self._loglik(p_val) + 1.0 / attr_size)

    def _posterior_normalize(self, p_val):
        posterior = self._posterior(p_val)
        return posterior / np.sum(posterior, axis=0)

    @staticmethod
    def _skill_dis(posterior_normalize):
        return np.sum(posterior_normalize, axis=1)

    def _get_skill_01_dis(self, posterior_normalize):
        skill0_dis = np.repeat(self._skill_dis(posterior_normalize), self.item_size)
        skill0_dis.shape = posterior_normalize.shape[0], self.item_size
        skill1_dis = skill0_dis.copy()
        return skill0_dis, skill1_dis

    def em(self):
        skills = self._get_all_skills()
        yita_val = self.get_yita(skills)
        score = self._score
        max_iter = self._max_iter
        tol = self._tol
        guess = self._guess_init
        no_slip = self._no_slip_init
        for i in range(max_iter):
            p_val = self.get_p(yita_val, no_slip, guess)
            post_normalize = self._posterior_normalize(p_val)
            skill0_dis, skill1_dis = self._get_skill_01_dis(post_normalize)
            skill0_item1_post_normalize = np.dot(post_normalize, score)
            skill1_item1_post_normalize = skill0_item1_post_normalize.copy()

            skill0_item1_dis, skill0_item_dis, skill1_item1_dis, skill1_item_dis = self._get_skill_item_dis(
                skill0_dis, skill0_item1_post_normalize, skill1_dis, skill1_item1_post_normalize, yita_val
            )

            guess_temp = self._get_est_guess(skill0_item1_dis, skill0_item_dis)
            no_slip_temp = self._get_est_no_slip(skill1_item1_dis, skill1_item_dis)

            if max(np.max(np.abs(guess - guess_temp)), np.max(np.abs(no_slip - no_slip_temp))) < tol:
                return no_slip_temp, guess_temp

            no_slip = no_slip_temp
            guess = guess_temp

        raise ConvergenceError('no Convergence')

    @staticmethod
    def _get_skill_item_dis(skill0_dis, skill0_item1_post_normalize, skill1_dis,
                            skill1_item1_post_normalize, yita_val):
        skill0_dis[yita_val == 1] = 0
        skill0_item_dis = np.sum(skill0_dis, axis=0)
        skill0_item_dis[skill0_item_dis <= 0] = 1e-10
        skill0_item1_post_normalize[yita_val == 1] = 0
        skill0_item1_dis = np.sum(skill0_item1_post_normalize, axis=0)
        skill1_dis[yita_val == 0] = 0
        skill1_item_dis = np.sum(skill1_dis, axis=0)
        skill1_item1_post_normalize[yita_val == 0] = 0
        skill1_item1_dis = np.sum(skill1_item1_post_normalize, axis=0)
        return skill0_item1_dis, skill0_item_dis, skill1_item1_dis, skill1_item_dis

    @staticmethod
    def _get_est_guess(skill0_item1_dis, skill0_item_dis):
        guess = skill0_item1_dis / skill0_item_dis
        guess[guess <= 0] = 1e-10
        return guess

    @staticmethod
    def _get_est_no_slip(skill1_item1_dis, skill1_item_dis):
        no_slip = 1 - (skill1_item_dis - skill1_item1_dis) / skill1_item_dis
        no_slip[no_slip >= 1] = 1 - 1e-10
        return no_slip


class BaseMcmcDina(Dina):

    def __init__(self, thin=1, burn=3000, max_iter=10000, *args, **kwargs):
        super(BaseMcmcDina, self).__init__(*args, **kwargs)
        self.max_iter = max_iter * thin
        self.burn = burn
        self.thin = thin

    def _get_item_params_tran(self, skills, no_slip, guess, next_no_slip, next_guess):
        yita_val = self.get_yita(skills)
        pre = self._get_loglik(yita_val, no_slip, guess, axis=0) + get_log_beta_pd(no_slip, guess)
        nxt = self._get_loglik(yita_val, next_no_slip, next_guess, axis=0) + get_log_beta_pd(next_no_slip, next_guess)
        res = np.exp(nxt - pre)
        res[res > 1] = 1
        return res

    def _get_loglik(self, yita, no_slip, guess, axis):
        score = self._score
        p_val = self.get_p(yita, no_slip, guess)
        return np.sum(score * np.log(p_val) + (1 - score) * np.log(1 - p_val), axis)

    def _get_item_params_init(self, size):
        skills = np.ones((self._people_size, self._skills_size))
        skills_list = np.zeros((size, self._people_size, self._skills_size))
        no_slip = np.zeros((1, self.item_size)) + 0.7
        guess = np.zeros((1, self.item_size)) + 0.3
        no_slip_list = np.zeros((size, self.item_size))
        guess_list = np.zeros((size, self.item_size))
        return guess, guess_list, no_slip, no_slip_list, skills, skills_list

    def _get_item_params_tran_res(self, skills, no_slip, guess):
        next_no_slip = np.random.uniform(no_slip - 0.1, no_slip + 0.1)
        next_no_slip[next_no_slip <= 0.4] = 0.4 + 1e-10
        next_no_slip[next_no_slip >= 1] = 1 - 1e-10
        next_guess = np.random.uniform(guess - 0.1, guess + 0.1)
        next_guess[next_guess <= 0] = 1e-10
        next_guess[next_guess >= 0.6] = 0.6 - 1e-10
        tran_param = self._get_item_params_tran(skills, no_slip, guess, next_no_slip, next_guess)
        param_r = np.random.uniform(0, 1, tran_param.shape)
        no_slip[tran_param >= param_r] = next_no_slip[tran_param >= param_r]
        guess[tran_param >= param_r] = next_guess[tran_param >= param_r]
        return no_slip, guess


class McmcDina(BaseMcmcDina):

    def _get_skills_tran(self, skills, no_slip, guess, next_skills):
        yita_val = self.get_yita(skills)
        pre = self._get_loglik(yita_val, no_slip, guess, axis=1)
        next_yita_val = self.get_yita(next_skills)
        nxt = self._get_loglik(next_yita_val, no_slip, guess, axis=1)
        res = np.exp(nxt - pre)
        res[res > 1] = 1
        return res

    def mcmc(self):
        size = self.max_iter
        bar = progressbar.ProgressBar()
        guess, guess_list, no_slip, no_slip_list, skills, skills_list = self._get_item_params_init(size)
        for i in bar(range(size)):
            skills = self._get_skills_tran_res(skills, no_slip, guess)
            no_slip, guess = self._get_item_params_tran_res(skills, no_slip, guess)
            skills_list[i] = skills
            no_slip_list[i] = no_slip
            guess_list[i] = guess
        est_skills = np.mean(skills_list[self.burn::self.thin], axis=0)
        est_no_slip = np.mean(no_slip_list[self.burn::self.thin], axis=0)
        est_guess = np.mean(guess_list[self.burn::self.thin], axis=0)
        return est_skills, est_no_slip, est_guess

    def _get_skills_tran_res(self, skills, no_slip, guess):
        next_skills = np.random.binomial(1, 0.5, skills.shape)
        tran_skills = self._get_skills_tran(skills, no_slip, guess, next_skills)
        skills_r = np.random.uniform(0, 1, tran_skills.shape)
        skills[tran_skills >= skills_r] = next_skills[tran_skills >= skills_r]
        return skills


class McmcHoDina(BaseMcmcDina):

    @staticmethod
    def get_skills_p(lam0, lam1, theta):
        exp_z = np.exp(theta * lam1 + lam0)
        p_val = exp_z / (1.0 + exp_z)
        return p_val

    def _get_skills_pd(self, skills, theta, lam0, lam1, axis):
        p_val = self.get_skills_p(lam0, lam1, theta)
        p_val[p_val <= 0] = 1e-10
        p_val[p_val >= 1] = 1 - 1e-10
        return np.sum(skills * np.log(p_val) + (1 - skills) * np.log(1 - p_val), axis)

    def _get_skills_tran(self, skills, no_slip, guess, theta, lam0, lam1, next_skills):
        yita_val = self.get_yita(skills)
        pre = self._get_loglik(yita_val, no_slip, guess, axis=1) + \
            self._get_skills_pd(skills, theta, lam0, lam1, 1)
        next_yita_val = self.get_yita(next_skills)
        nxt = self._get_loglik(next_yita_val, no_slip, guess, axis=1) + \
            self._get_skills_pd(next_skills, theta, lam0, lam1, 1)
        res = np.exp(nxt - pre)
        res[res > 1] = 1
        return res

    def _get_theta_tran(self, skills, theta, lam0, lam1, next_theta):
        pre = self._get_skills_pd(skills, theta, lam0, lam1, 1) + get_log_normal_pd(theta)[:, 0]
        nxt = self._get_skills_pd(skills, next_theta, lam0, lam1, 1) + get_log_normal_pd(next_theta)[:, 0]
        res = np.exp(nxt - pre)
        res[res > 1] = 1
        return res

    def _get_lam_tran(self, skills, theta, lam0, lam1, next_lam0, next_lam1):
        pre = self._get_skills_pd(skills, theta, lam0, lam1, 0) + get_log_normal_pd(lam0) + \
            get_log_lognormal_pd(lam1)
        nxt = self._get_skills_pd(skills, theta, next_lam0, next_lam1, 0) + get_log_normal_pd(next_lam0) + \
            get_log_lognormal_pd(next_lam1)
        res = np.exp(nxt - pre)
        res[res > 1] = 1
        return res

    def _get_skills_tran_res(self, skills, no_slip, guess, theta, lam0, lam1):
        next_skills = np.random.binomial(1, 0.5, skills.shape)
        tran_skills = self._get_skills_tran(skills, no_slip, guess, theta, lam0, lam1, next_skills)
        skills_r = np.random.uniform(0, 1, tran_skills.shape)
        skills[tran_skills >= skills_r] = next_skills[tran_skills >= skills_r]
        return skills

    def mcmc(self):
        size = self.max_iter
        bar = progressbar.ProgressBar()
        guess, guess_list, no_slip, no_slip_list, skills, skills_list = self._get_item_params_init(size)
        theta = np.zeros((self._people_size, 1))
        theta_list = np.zeros((size, self._people_size, 1))
        lam0 = np.zeros(5)
        lam0_list = np.zeros((size, 5))
        lam1 = np.ones(5)
        lam1_list = np.zeros((size, 5))
        for i in bar(range(size)):

            next_lam0 = np.random.uniform(lam0 - 0.3, lam0 + 0.3)
            next_lam1 = np.random.uniform(lam1 - 0.3, lam1 + 0.3)
            next_lam1[next_lam1 <= 0] = 1e-10
            next_lam1[next_lam1 > 4] = 4
            tran_lam = self. _get_lam_tran(skills, theta, lam0, lam1, next_lam0, next_lam1)
            lam_r = np.random.uniform(0, 1, tran_lam.shape)
            lam0[tran_lam >= lam_r] = next_lam0[tran_lam >= lam_r]
            lam1[tran_lam >= lam_r] = next_lam1[tran_lam >= lam_r]
            lam0_list[i] = lam0
            lam1_list[i] = lam1

            next_theta = np.random.normal(theta, 0.1)
            tran_theta = self._get_theta_tran(skills, theta, lam0, lam1, next_theta)
            theta_r = np.random.uniform(0, 1, tran_theta.shape)
            theta[tran_theta >= theta_r] = next_theta[tran_theta >= theta_r]
            theta_list[i] = theta

            skills = self._get_skills_tran_res(skills, no_slip, guess, theta, lam0, lam1)
            no_slip, guess = self._get_item_params_tran_res(skills, no_slip, guess)

            skills_list[i] = skills
            no_slip_list[i] = no_slip
            guess_list[i] = guess

        est_lam0 = np.mean(lam0_list[self.burn::self.thin], axis=0)
        est_lam1 = np.mean(lam1_list[self.burn::self.thin], axis=0)
        est_theta = np.mean(theta_list[self.burn::self.thin], axis=0)
        est_skills = np.mean(skills_list[self.burn::self.thin], axis=0)
        est_no_slip = np.mean(no_slip_list[self.burn::self.thin], axis=0)
        est_guess = np.mean(guess_list[self.burn::self.thin], axis=0)
        return est_lam0, est_lam1, est_theta, est_skills, est_no_slip, est_guess
