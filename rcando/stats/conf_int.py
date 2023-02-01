#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Confidence Intervals unified API
version: 1.0
type: module
keywords: [confidence interval for mean, mean estimation, CLT, bootstrap, proportions]
description: |
    A unified API for various methods of obtaining condifence intervals
    (currently only applied for the mean).
    For each method of mean estimation (and its CI) separate class is created,
    but it's structure is always the same.
    The main class is CI and it applies Central Limit Theorem
    (the most classic approach to mean and its CI estimation).
    All other classes inherits from CI
    and are adopted to different methods of mean estimation
    via redefinition of __init__().
    All the methods from parent are still valid.
remarks:
    - This unified API may (and should) be used for other statistics
      like e.g. variance estimator;
todo:
    - arithmetic on CIs: see CI_diff and CI_ratio for provisional solutions;
    - applying solutions of Behrens-Fisher problem to CI_diff;
sources:
file:
    usage:
        interactive: True
        terminal: True
    name: builtin.py
    path: D:/ROBOCZY/Python/RCanDo/ak/
    date: 2019-11-20
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - akasp666@google.com
              - arek@staart.pl
"""

# %%
from typing import Union
from copy import deepcopy

import numpy as np
import pandas as pd

import scipy.stats as st
from statsmodels.stats import proportion

import common.builtin as bi


# %%
class CI():

    def __init__(self, data: pd.Series, alpha: float = .05, method: str = 'normal'):
        """
        data: sample from any distribution with finite variance
        alpha: significance level (1 - alpha is confidence level)
        method: str;
            one of 'student' or 'normal' (default)
            if 'normal' (default) then standard normal quantile is used to get CI
            if 'student' then Student t quantile is used
         Returns
        confidence interval for the mean of data (!)
        derived from Central Limit Theorem i.e. approx with normal distribution
        (Student t distr. if `student=True` which is more conservative and proper
        as it takes into account additional uncertainty resulting from estimation of variance)
         Basic subroutines:
        scipy.stats.norm
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
        scipy.stats.t
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
        """
        self.ci_type = "default (CLT)"
        self.method = method
        self.alpha = alpha
        #
        self.N = len(data) if data is not None else 1
        self.mean = np.mean(data) if data is not None else 0
        self.std = np.std(data) if data is not None else 1

        if method == 'student':
            # we use Student t distr as variance is estimated
            self.q_alpha = st.t.ppf(1 - alpha / 2, self.N - 1)
        elif method == 'normal':
            self.q_alpha = st.norm.ppf(1 - alpha / 2)
        else:
            raise Exception("method must be one of 'student' or 'normal'")

        self.half_ci = self.q_alpha * self.std / np.sqrt(self.N)
        self.ci = self.mean + np.array((-self.half_ci, self.half_ci))
        self.left = self.ci[0]
        self.right = self.ci[1]

        self.mean0 = self.mean  # see .scale() below
        self.s = 1              # "

    def __repr__(self):
        left = bi.adaptive_round(self.left, 4)
        right = bi.adaptive_round(self.right, 4)
        ss = f"CI {self.ci_type}: ({left}, {right})"
        return ss

    @property
    def relative_width(self):
        return self.half_ci * 2 / self.mean

    @property
    def copy(self):
        return deepcopy(self)

    @property
    def summary_dict(self):
        summary = {
            "CI": (self.left, self.right),
            "mean": self.mean,
            "std": self.std,
            "N": self.N,
            "width of CI": 2 * self.half_ci,
            "CI width to mean": self.relative_width,
            "original mean": self.mean0,
            "scale": self.s,
            "method": self.method, }
        return summary

    def summary(self, r: int = 4, sep: str = ":"):
        print(f"summary for CI {self.ci_type}")
        for k, v in self.summary_dict.items():
            print(k.rjust(17), sep, bi.adaptive_round(v, r))

    # %%
    def size_for_width(self, width: float = .1, as_ratio=True) -> int:
        """
        How large should be N (sample size) to get CI having width about the size of
            width * self.mean
        where  width of CI  is  `self.right - self.left = 2 * self.half_ci`
        as_ratio: bool
            if True (default) then `width` is treated as ratio of self.mean;
            if False then `width` is treated as number = the value of width in demand;
         Returns
        n: integer
        """
        if as_ratio:
            n = (2 * self.std * self.q_alpha / (width * self.mean)) ** 2
        else:
            n = (2 * self.std * self.q_alpha / width) ** 2
        return int(n)

    def scale(self, s: float = None, all: bool = False):
        """
        scaling the whole thing;
        original value of  self.mean  is retained in  self.mean0

        if one wants to scale also  .mean0  set `all=True`
        this is like scaling first data on which CI is estimated,
        thus in this case self.s is not multiplied by `scale` (`scale` is not remembered)
        in order for self.s to be always a valid scaling between self.mean0 and sel.mean

        scaling by  sample size = self.N (default, when `s=None`)
        may be good for comparison with bootstrap CI for sum.
        """
        s = self.N if s is None else s
        new = deepcopy(self)
        if all:
            new.mean0 = new.mean0 * s
        else:
            new.s = new.s * s
        new.mean = new.mean * s
        new.std = new.std * abs(s)
        new.half_ci = new.half_ci * abs(s)
        new.ci = new.ci * s
        new.ci.sort()               # !!! inplace
        new.left = new.ci[0]
        new.right = new.ci[1]
        return new

    def __mul__(self, a: Union[float, int]):
        return self.copy.scale(a, all=True)

    def __rmul__(self, a: Union[float, int]):
        return self.copy.scale(a, all=True)

    def move(self, b: Union[float, int]):
        """
        move the whole thing by a given value,
        i.e. add `b` to mean and confidence intervals;
        it is like adding `b` to the original data.
        """
        new = self.copy
        new.mean = new.mean + b
        new.ci = new.ci + b
        new.left = new.ci[0]
        new.right = new.ci[1]
        return new

    def __add__(self, a):
        return self.move(a)

    def __radd__(self, a):
        return self.move(a)


# %%
class CI_CLT(CI):

    def __init__(self, data: pd.Series, alpha: float = .05, method: str = 'normal'):
        """
        data: sample from any distribution with finite variance
        alpha: significance level (1 - alpha is confidence level)
        method: str;
            one of 'student' or 'normal' (default)
            if 'normal' (default) then standard normal quantile is used to get CI
            if 'student' then Student t quantile is used
         Returns
        confidence interval for the mean of data (!)
        derived from Central Limit Theorem i.e. approx with normal distribution
        (Student t distr. if `student=True` which is more conservative and proper
        as it takes into account additional uncertainty resulting from estimation of variance)
         Basic subroutines:
        scipy.stats.norm
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
        scipy.stats.t
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
        """
        super().__init__(data, alpha, method)
        self.ci_type = "CLT"


# %%
class CI_binom(CI):

    def __init__(self, n_success: int, n: int, alpha: float, method: str = 'normal'):
        """
        n_success: number of successes
        n: number of trials
        alpha: significance level
        method : {'normal', 'agresti_coull', 'beta', 'wilson', 'binom_test'}
            default: 'normal'
             - `normal` : asymptotic normal approximation
             - `agresti_coull` : Agresti-Coull interval
             - `beta` : Clopper-Pearson interval based on Beta distribution
             - `wilson` : Wilson Score interval
             - `jeffreys` : Jeffreys Bayesian Interval
             - `binom_test` : experimental, inversion of binom_test
        if method 'normal' then the result is simply appl. of CLT approx
        (as in CI_CLT class below which however needs whole data series)
         Returns
        two-sided confidence interval for binomial distribution
         Basic subroutine:
        statsmodels.stats.proportion.proportion_confint()
        https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html#statsmodels.stats.proportion.proportion_confint

        """
        self.ci_type = "binom"
        self.method = method
        self.alpha = alpha
        #
        self.N = n
        self.n_success = n_success
        self.mean = n_success / n
        self.p = self.mean
        self.std = np.sqrt(self.p * (1 - self.p) / n)
        self.q_alpha = st.norm.ppf(1 - alpha / 2)  # is used internally if method='normal' (but not only)
        # may be useful for some checks and for reference
        #
        self.ci = np.array(proportion.proportion_confint(count=n_success, nobs=n, alpha=alpha, method=method))
        self.left = self.ci[0]
        self.right = self.ci[1]
        self.half_ci = (self.right - self.left) / 2
        #
        self.mean0 = self.mean  # see .scale() below
        self.s = 1              # "


# %%
class CI_boot(CI):

    def __init__(
            self,
            data: pd.Series,
            statistic: callable,
            alpha: float,
            n_resamples=9999,
            batch=None,
            method="BCa", ):
        """
        data:  Each element of data is a sample from an underlying distribution
        statistic: statistic for which the confidence interval is to be calculated
        confidence_level: confidence_level
        method: one of "percentile", "basic", "BCa"
         Returns
        bootstrap object with two sided confidence interval
         Basic subroutine:
        scipy.stats.bootstrap()
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html?highlight=bootstrap#scipy.stats.bootstrap
        """
        self.ci_type = "boot"
        self.method = method
        self.alpha = alpha
        #
        self.N = len(data)
        self.value = statistic(data)
        self.statistic_name = statistic.__name__
        #
        self.n_resamples = n_resamples
        self.batch = batch
        boot = st.bootstrap((data,), statistic, confidence_level=1 - alpha, n_resamples=n_resamples,
                            method=method, batch=batch)
        self.ci = np.array(boot.confidence_interval)
        self.left = self.ci[0]
        self.right = self.ci[1]
        self.half_ci = (self.right - self.left) / 2
        #
        self.std = boot.standard_error   # this gives very small values in comparison with CL_CLT; WHY???
        #
        self.mean = self.ci.mean()   # for consistency with other CI_..
        self.mean0 = self.mean  # see .scale() below
        self.s = 1              # "


# %%
class CI_ratio(CI):

    def __init__(self, nom, denom):
        """
        confidence interval of "the ratio of two condidence intervals";
        nom: CI for some stat_X on X variable
        denom: CI for some stat_Y on Y variable
            where CI is object having attributes .left, .right and .ci_type
            e.g. one of the class defined above
         Returns
        approximate confidence interval for the ratio
            stat_X / stat_Y
        !!! It is assumed that both CI are calculated for the same confidence or significance level !!!
        Otherwise it has no sense.
        Both CI should also have the same .ci_type (but it's not critical).
         Remark
        The whole methodology here is very naive!
        The truth is that for getting all the values here one needs to know distributions.
        Especially standard deviation is very tricky to obtain and it may even not exist!
        Thus we do not propose any crude method for it (leave it None).
        """
        self.ci_type = f"ratio ({nom.ci_type} / {denom.ci_type})"
        self.method = f"{nom.method} / {denom.method}"
        self.alpha = nom.alpha

        self.N_nom = nom.N
        self.N_denom = denom.N
        self.N = min(self.N_nom, self.N_denom)

        self.left = nom.left / denom.right
        self.right = nom.right / denom.left
        self.left, self.right = sorted((self.left, self.right))     # it's not obvious for ratio
        self.ci = np.array(sorted((self.left, self.right)))
        self.half_ci = (self.right - self.left) / 2

        try:
            self.mean = nom.mean / denom.mean
        except AttributeError:
            self.mean = None
        self.std = None
        try:
            self.q_alpha = max(nom.q_alpha, denom.q_alpha)
        except AttributeError:
            self.q_alpha = None
        #
        self.mean0 = self.mean   # only for consistency with other CI_..
        self.s = 1               # "

    @property
    def summary_dict(self):
        summary = super().summary_dict
        summary["N nominator"] = self.N_nom
        summary["N denominator"] = self.N_denom
        return summary


class CI_diff(CI):

    def __init__(self, diff_left, diff_right):
        """
        confidence interval of "the difference of two condidence intervals";
        diff_left: CI for some stat_X on X variable
        diff_right: CI for some stat_Y on Y variable
            where CI is object having attributes .left, .right and .ci_type
            e.g. one of the class defined above
         Returns
        approximate confidence interval for the difference
            stat_X - stat_Y
        !!! It is assumed that both CI are calculated for the same confidence or significance level !!!
        Otherwise it has no sense.
        Both CI should also have the same .ci_type (but it's not critical).
         Remark
        The proper solution stems from Behrens-Fisher problem:
        https://en.wikipedia.org/wiki/Behrens%E2%80%93Fisher_problem
        """
        self.ci_type = f"diff ({diff_left.ci_type} - {diff_right.ci_type})"
        self.method = f"{diff_left.method} / {diff_right.method}"
        self.alpha = diff_left.alpha

        self.N_diff_left = diff_left.N
        self.N_diff_right = diff_right.N
        self.N = min(self.N_diff_left, self.N_diff_right)

        self.left = diff_left.left - diff_right.right
        self.right = diff_left.right - diff_right.left
        self.ci = np.array((self.left, self.right))
        self.half_ci = (self.right - self.left) / 2

        try:
            self.mean = diff_left.mean - diff_right.mean
        except AttributeError:
            self.mean = None
        try:
            self.std = np.sqrt((diff_left.std) ** 2 + (diff_right.std) ** 2)
        except AttributeError:
            self.std = None
        try:
            self.q_alpha = max(diff_left.q_alpha, diff_right.q_alpha)
        except AttributeError:
            self.q_alpha = None
        #
        self.mean0 = self.mean   # only for consistency with other CI_..
        self.s = 1               # "

    @property
    def summary_dict(self):
        summary = super().summary_dict
        summary["N left"] = self.N_diff_left
        summary["N right"] = self.N_diff_right
        return summary


# %%
