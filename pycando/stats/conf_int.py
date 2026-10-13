#! python3
# -*- coding: utf-8 -*-
"""
---
title: Confidence Intervals unified API
version: 1.0
type: module
keywords: [confidence interval for mean, mean estimation, CLT, bootstrap, proportions]
description: |
    A unified API for various methods for obtaining condifence intervals
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
    date: 2022-11-20
    authors:
        - nick: arek
          fullname: Arkadiusz Kasprzyk
          email:
              - arkasiusz.kasprzyk@quantup.pl
"""

# %%
from __future__ import annotations
import warnings

from typing import Union
from copy import deepcopy

import numpy as np
import pandas as pd

import scipy.stats as st
from statsmodels.stats import proportion

import utils.builtin as bi


# %%
class CI:

    def __init__(
            self,
            data: pd.Series | None,
            alpha: float = .05,
            method: str = 'normal',     # "student"
    ):
        """
        Common API for confidence interval at the confidence level `1 - alpha`
        for mean (or other statistics) obtained by various methods.

        In this class the default method is application of Central Limit Theorem
        with quantiles at given significance level `alpha` obtained from normal distribution,
        `method = 'normal'` (default), or from Student t-distribution, `method = 'student'`.

        All other types of CIs in this module, obtained by different procedures,
        inherit all methods and attributes from this class;
        however, their initialisation (thus also set and types of arguments)
        is adopted to the specific method.

        Arguments
        ---------
        data: pd.Series | None
            sample from any distribution with finite variance
        alpha: float = .05
            significance level (1 - alpha is confidence level)
        method: str = 'normal'
            one of 'student' or 'normal' (default)
            if 'normal' (default) then standard normal quantile is used to get CI
            if 'student' then Student t quantile is used

        Returns
        -------
        Confidence Interval for the mean of data (!)
        derived from Central Limit Theorem applied to ordinary sample mean,
        i.e. quantiles of order `alpha/2` and `1 - alpha/2` are derived for the normal distribution, or
        Student t distr. if `student=True` (which is more conservative and proper
        as it takes into account additional uncertainty resulting from estimation of variance).

        Basic subroutines
        -----------------
        scipy.stats.norm
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
        scipy.stats.t
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
        """

        self.ci_type = "default (CLT)"
        self.method = method
        self.alpha = alpha
        #
        self.N = len(data) if data is not None else 0

        if self.N > 0:

            self.mean = np.mean(data)
            self.std = np.std(data)

            if method == 'student':
                # we use Student t distr as variance is estimated
                self.q_alpha = st.t.ppf(1 - alpha / 2, self.N - 1)
            elif method == 'normal':
                self.q_alpha = st.norm.ppf(1 - alpha / 2)
            else:
                raise Exception("method must be one of 'student' or 'normal'")

            self.half_ci = self.q_alpha * self.std / np.sqrt(self.N)
            self.ci = self.mean + np.array((-self.half_ci, self.half_ci))

        else:
            warnings.warn(
                "No data to calculate mean and its confidence interval. All attributes set to `np.nan`."
            )
            self.mean = np.nan
            self.std = np.nan
            self.q_alpha = np.nan
            self.half_ci = np.nan
            self.ci = np.array((np.nan, np.nan))

        self.left = self.ci[0]
        self.right = self.ci[1]

        self.mean0 = self.mean  # see .scale() below
        self.s = 1              # "

    def __repr__(self) -> str:
        left = bi.adaptive_round(self.left, 4)
        right = bi.adaptive_round(self.right, 4)
        ss = f"CI {self.ci_type}: ({left}, {right})"
        return ss

    @property
    def upper(self) -> float:
        return self.right

    @property
    def lower(self) -> float:
        return self.left

    @property
    def relative_width(self) -> float:
        return self.half_ci * 2 / self.mean

    @property
    def copy(self) -> CI:
        return deepcopy(self)

    @property
    def summary_dict(self) -> dict[str, float]:
        """
        Returns summary dictionary for this CI, with the following entries:
        - CI: interval as tuple (left, right);
        - mean: estimated mean;
        - std: estimated standard deviation of the mean estimate;
        - width of CI: i.e. right - left;
        - CI width to mean: i.e. mean/width;
        - original mean: if the mean was scaled here its original value is remembered;
        - scale: what scale was applied to the mean, see .scale() method;
        - method: what specific method was applied to get the CI.
        """
        summary = {
            "CI": (self.left, self.right),
            "mean": self.mean,
            "std": self.std,
            "N": self.N,
            "width of CI": 2 * self.half_ci,
            "CI width to mean": self.relative_width,
            "original mean": self.mean0,
            "scale": self.s,
            "method": self.method,
        }
        return summary

    def summary(self, precision: int = 3, sep: str = ":") -> None:
        """
        Prints summary dictionary for this CI.

        Arguments
        ---------
        precision : int = 3,
            number of significant digits to display (only for fractions);
        sep : str = ":",
            seperator between key and value on the printout;

        Returns
        -------
        None
        """

        print(f"summary for CI {self.ci_type}")
        for k, v in self.summary_dict.items():
            print(k.rjust(17), sep, bi.adaptive_round(v, precision))

    # %%
    def size_for_width(
            self,
            width: float = .1,
            as_ratio: bool = True,
    ) -> int:
        """
        How large should be N (sample size) to get CI having width about the size of
        `width * self.mean`

        Arguments
        ---------
        width: float = .1,
            width of CI  is  `self.right - self.left = 2 * self.half_ci`
        as_ratio: bool = True
            if True (default) then `width` is treated as ratio of self.mean;
            if False then `width` is treated as number = the value of width in demand;

        Returns
        -------
        n: integer
        """
        if as_ratio:
            n = (2 * self.std * self.q_alpha / (width * self.mean)) ** 2
        else:
            n = (2 * self.std * self.q_alpha / width) ** 2
        return int(n)

    def scale(
            self,
            s: float | None = None,
            all: bool = False,
    ) -> CI:
        """
        Scaling the whole thing by the factor of `s`;
        original value of `self.mean` is retained in `self.mean0`.

        If one wants to scale also `.mean0` set `all=True`;
        this is like scaling first data on which CI is estimated,
        Thus in this case self.s is not multiplied by `scale` (`scale` is not remembered)
        in order for self.s to be always a valid scaling between self.mean0 and self.mean

        scaling by  sample size = `self.N` (default, when `s=None`)
        is good for comparison with bootstrap CI for sum.
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
        new.ci.sort()               # inplace
        new.left = new.ci[0]
        new.right = new.ci[1]
        return new

    def __mul__(self, a: Union[float, int]) -> CI:
        return self.copy.scale(a, all=True)

    def __rmul__(self, a: Union[float, int]) -> CI:
        return self.copy.scale(a, all=True)

    def move(self, b: Union[float, int]) -> CI:
        """
        Move the whole thing by a given value,
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

    def __init__(
            self,
            data: pd.Series[float] | None,
            alpha: float = .05,
            method: str = 'normal',
    ):
        """
        Confidence Interval at the confidence level `1 - alpha`
        for the mean of data derived from Central Limit Theorem applied to ordinary sample mean.

        Arguments
        ---------
        data: pd.Series | None,
            sample from any distribution with finite variance
        alpha: float = .05,
            significance level (1 - alpha is confidence level)
        method: str = 'normal',
            one of 'student' or 'normal' (default);
            if 'normal' (default) then standard normal quantile is used to get CI;
            if 'student' then Student t quantile is used.

        Returns
        -------
        Confidence Interval for the mean of data
        derived from Central Limit Theorem applied to ordinary sample mean,
        i.e. quantiles of order `alpha/2` and `1 - alpha/2` are derived for the normal distribution, or
        Student t distr. if `student=True` (which is more conservative and proper
        as it takes into account additional uncertainty resulting from estimation of variance).

        Basic subroutines
        -----------------
        scipy.stats.norm
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm
        scipy.stats.t
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html#scipy.stats.t
        """
        super().__init__(data, alpha, method)
        self.ci_type = "CLT"


# %%
class CI_binom(CI):

    def __init__(
            self,
            n_success: int,
            n: int,
            alpha: float = .05,
            method: str = 'normal',
    ):
        """
        Confidence interval at the confidence level `1 - alpha`
        for the probability of success `p` in the Bernoulli schema
        with `n` trials where observed number of successes where `n_success`.

        Arguments
        ---------
        n_success: int,
            number of successes;
        n: int,
            number of Bernoulli trials;
            if `n <= 0` then
        alpha: float
            significance level,
        method: str = 'normal',
            one of 'normal' (default), 'agresti_coull', 'beta', 'wilson', 'binom_test';
             - `normal` : asymptotic normal approximation;
             - `agresti_coull` : Agresti-Coull interval;
             - `beta` : Clopper-Pearson interval based on Beta distribution;
             - `wilson` : Wilson Score interval;
             - `jeffreys` : Jeffreys Bayesian Interval;
             - `binom_test` : experimental, inversion of binom_test;
        if method 'normal' then the result is simply appl. of CLT approx
        (as in CI_CLT class below which, however, needs the whole data series).

        Returns
        -------
        Two-sided confidence interval for `p` for binomial distribution `B(n, p)`.

        Basic subroutine
        ----------------
        statsmodels.stats.proportion.proportion_confint()
            https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html#statsmodels.stats.proportion.proportion_confint

        """
        self.ci_type = "binom"
        self.method = method
        self.alpha = alpha
        #
        self.N = n
        self.n_success = n_success

        if self.N > 0 and self.n_success <= self.N:

            self.mean = n_success / n
            self.p = self.mean
            self.std = np.sqrt(self.p * (1 - self.p) / n)
            self.q_alpha = st.norm.ppf(1 - alpha / 2)  # is used internally if method='normal' (but not only)
            # may be useful for some checks and for reference
            #
            self.ci = np.array(proportion.proportion_confint(count=n_success, nobs=n, alpha=alpha, method=method))

        else:
            if self.N <= 0:
                warnings.warn(
                    "No data to calculate mean (p) and its confidence interval. All attributes set to `np.nan`."
                )
            elif self.n_success > self.N:
                warnings.warn("There cannot be more successes then trials. All attributes set to `np.nan`.")

            self.mean = np.nan
            self.p = np.nan
            self.std = np.nan
            self.q_alpha = np.nan
            self.ci = np.array((np.nan, np.nan))

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
            data: pd.Series | None,
            statistic: callable[[pd.Series | np.array], float] = np.mean,
            alpha: float = .05,
            n_resamples: int = 9999,
            batch: int | None = None,
            method: str = "BCa",
            **kwargs
    ):
        """
        Confidence Interval at the confidence level `1 - alpha`
        for any `statistic` from the `data`
        obtained via bootstrap sampling.

        Arguments
        ---------
        data: pd.Series | None,
            Each element of data is a sample from an underlying distribution;
        statistic: callable,
            statistic for which the confidence interval is to be calculated;
        alpha: float = .05,
            significance level; then confidence_level = 1 - alpha;
        n_resamples: int = 9999,
            The number of resamples performed to form the bootstrap distribution of the statistic.
        batch: int = None,
            The number of resamples to process in each vectorized call to statistic.
            Memory usage is `O(batch * n)`, where `n` is the sample size.
            Default is None, in which case `batch = n_resamples`
            (or `batch = max(n_resamples, n)` for `method='BCa'`).
        method: str = "BCa",
            one of "percentile", "basic", "BCa",
        **kwargs
            rest of the arguments passed to scipy.stats.bootstrap()

        Returns
        -------
        bootstrap object with two sided confidence interval.

        Basic subroutine
        ----------------
        scipy.stats.bootstrap
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html?highlight=bootstrap#scipy.stats.bootstrap
        """
        self.ci_type = "boot"
        self.method = method
        self.alpha = alpha
        #
        self.N = len(data) if data is not None else 0

        if self.N > 0:

            self.value = statistic(data)
            self.statistic_name = statistic.__name__
            #
            self.n_resamples = n_resamples
            self.batch = batch
            boot = st.bootstrap(
                (data,), statistic,
                confidence_level=1 - alpha,
                n_resamples=n_resamples, method=method, batch=batch
            )
            self.ci = np.array(boot.confidence_interval)
            #
            self.std = boot.standard_error   # this gives very small values in comparison with CL_CLT; WHY???
            #
            self.mean = self.ci.mean()   # for consistency with other CI_..

        else:
            warnings.warn(
                "No data to calculate statistics and its confidence interval. All attributes set to `np.nan`."
            )

            self.value = np.nan
            self.statistic_name = np.nan
            #
            self.n_resamples = np.nan
            self.batch = np.nan
            self.ci = np.array((np.nan, np.nan))
            self.std = np.nan
            self.mean = np.nan

        self.left = self.ci[0]
        self.right = self.ci[1]
        self.half_ci = (self.right - self.left) / 2

        self.mean0 = self.mean  # see .scale() below
        self.s = 1              # "


# %%
class CI_ratio(CI):

    def __init__(self, nom: CI, denom: CI) -> CI:
        """
        Confidence Interval of "the ratio of two confidence intervals".

        Arguments
        ---------
        nom: CI,
            CI for some stat_X on X variable;
        denom: CI,
            CI for some stat_Y on Y variable
            where CI is object having attributes .left, .right and .ci_type
            e.g. one of the class defined above;

        Returns
        -------
        approximate confidence interval for the ratio
            stat_X / stat_Y
        !!! It is assumed that both CI are calculated for the same confidence or significance level !!!
        Otherwise it has no sense.
        Both CI should also have the same .ci_type (but it's not critical).

        Remark
        ------
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

    def __init__(self, diff_left: CI, diff_right: CI) -> CI:
        """
        Confidence Interval of "the difference of two condidence intervals";

        Arguments
        ---------
        diff_left: CI,
            CI for some stat_X on X variable
        diff_right: CI,
            CI for some stat_Y on Y variable
            where CI is object having attributes .left, .right and .ci_type
            e.g. one of the class defined above

        Returns
        -------
        approximate confidence interval for the difference
            stat_X - stat_Y
        !!! It is assumed that both CI are calculated for the same confidence or significance level !!!
        Otherwise it has no sense.
        Both CI should also have the same .ci_type (but it's not critical).

        Remark
        ------
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
    def summary_dict(self) -> dict[str, float]:
        summary = super().summary_dict
        summary["N left"] = self.N_diff_left
        summary["N right"] = self.N_diff_right
        return summary


# %%
