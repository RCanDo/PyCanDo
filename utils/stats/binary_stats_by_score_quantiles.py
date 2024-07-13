#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Binary statistics by scores-quantiles
version: 1.0
type: module
keywords: [score quantiles, binary classifier, lift, gain, count, ...]
description: |
remarks:
todo:
sources:
    - https://www.geeksforgeeks.org/understanding-gain-chart-and-lift-chart/
    - https://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
file:
    date: 2024-01-12
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%

# from functools import partialmethod
from typing import Sequence
import numpy as np
import pandas as pd


class BinaryStatsByScoresQuantiles(pd.DataFrame):
    """
    Don't use it directly!
    It's only a helper parent class for more specific statistics on scores from binary model.
    Some assumptions are made here which are not checked.
    """
    def __init__(
            self, quantiles, stats,
            quantiles_name: str = None, stats_name: str = None,
            *args, **kwargs     # passed to DataFrame init
    ):

        if len(quantiles) != len(stats):
            raise Exception("Length of `quantiles` and `stats` cannot differ.")

        quantiles = pd.Series(quantiles)
        stats = pd.Series(stats)
        if not all(stats.index == quantiles.index):
            stats.index = quantiles.index

        quantiles_name = quantiles_name or quantiles.name or "quantiles"
        stats_name = stats_name or stats.name or "statistics"

        super().__init__({quantiles_name: quantiles, stats_name: stats}, *args, **kwargs)

        self.quantiles_name = quantiles_name
        self.stats_name = stats_name

    def to_series(self, quantile_index: bool = True):
        if quantile_index:
            return self.set_index(self.quantiles_name)[self.stats_name]
        else:
            return self[self.stats_name]

    def to_dict(self, quantile_index: bool = True):
        return self.to_series(quantile_index).to_dict()

    def to_list(self):
        return self[self.stats_name].to_list()

    def to_tuple(self):
        return tuple(self.to_list())


class BinaryCountsByScoresQuantiles(BinaryStatsByScoresQuantiles):
    """
    For a model for binary target y with predictions y_hat as probabilities or scores
    (i.e. prediction is continuous on (0, 1), like in logistic regression model),
    we count number of ones falling in each of q quantile intervals of y_hat.
    I.e. we sort data [y, y_hat] wrt y_hat scores (in increasing order)
    and count number of 1s in each quantile interval.

    Example
    -------
    # fake data
    np.random.seed(22)
    N = 33
    y = np.random.choice([0, 1], N, p=[.7, .3])
    y_hat = np.random.sample(N)
    bc = BinaryCountsByScoresQuantiles(y, y_hat, 4)

    bc
    #   score_quantiles  y
    # 1  (-0.001, 0.25]  2
    # 2     (0.25, 0.5]  5
    # 3     (0.5, 0.75]  3
    # 4     (0.75, 1.0]  2

    bc.to_series()
    # score_quantiles
    # (-0.001, 0.25]    2
    # (0.25, 0.5]       5
    # (0.5, 0.75]       3
    # (0.75, 1.0]       2
    # Name: y, dtype: int64

    bc.to_series(False)
    # 1    2
    # 2    5
    # 3    3
    # 4    2
    # Name: y, dtype: int64

    bc.to_dict()
    # {'(-0.001, 0.25]': 2, '(0.25, 0.5]': 5, '(0.5, 0.75]': 3, '(0.75, 1.0]': 2}
    bc.to_dict(False)
    # {1: 2, 2: 5, 3: 3, 4: 2}

    bc.to_list()
    # [2, 5, 3, 2]

    bc.nobs     # 33
    bc.nobs1    # 12
    """
    def __init__(
        self, y: Sequence, y_hat: Sequence[float],
        q_order: int = 10,
        ones_label: int | str = 1,      # all other labels are considered belonging to class 0
        y_name: str = None,
        y_hat_name: str = None,
        quantiles_name: str = None,     # overwrites  `y_hat_name + "_quantiles"`
        model_name: str = "binary_model",
        *args, **kwargs     # passed to DataFrame init
    ):

        if len(y) != len(y_hat):
            raise Exception("Length of `y` and `y_hat` cannot differ.")

        y = pd.Series(y).reset_index(drop=True)
        y_hat = pd.Series(y_hat).reset_index(drop=True)

        y = (y == ones_label).astype(int)

        if q_order == 0:
            # just sorting wrt. score, y (ascending)
            qdf = pd.DataFrame({"q_hat": y_hat, "y": y})  # names are temporary
            gqdf = qdf.sort_values(["q_hat", "y"])
            gqdf.index = range(1, len(y) + 1)

        else:
            if q_order > 0:
                # theoretically proper score quantiles
                quantiles = np.linspace(0, 1, q_order + 1)
                quantiles = y_hat.quantile(quantiles).round(5)
                quantiles[0] = 0
                quantiles[-1] = 1
                quantiles = np.unique(quantiles)

            else:
                # dividing score values space, i.e. interval [0, 1], into q equal intervals
                # -- not proper score quantiles but rather score distribution what is more informative
                quantiles = np.linspace(0, 1, -q_order + 1)

            q_hat = pd.cut(y_hat, quantiles, include_lowest=True)

            qdf = pd.DataFrame({"q_hat": q_hat, "y": y})  # names are temporary
            gqdf = qdf.groupby("q_hat", as_index=False).sum()
            gqdf["q_hat"] = gqdf["q_hat"].astype(str)
            gqdf.index = range(1, len(quantiles))

        y_hat_name = y_hat_name or y_hat.name or "score"
        quantiles_name = quantiles_name or y_hat_name + ("_quantiles" if q_order > 0 else "")
        y_name = y_name or y.name or "y"

        super().__init__(
            gqdf['q_hat'], gqdf['y'],
            quantiles_name, y_name,
            *args, **kwargs
        )

        self.quantiles_name = quantiles_name
        self.y_name = y_name
        self.y_hat_name = y_hat_name

        self.q_order = q_order
        self.nobs = len(y)
        self.nobs1 = sum(y)
        self.model_name = model_name


class Gain(BinaryStatsByScoresQuantiles):

    def __init__(
        self, y: Sequence, y_hat: Sequence[float],
        q_order: int = 10,
        ones_label: int | str = 1,      # all other labels are considered belonging to class 0
        y_name: str = None,
        y_hat_name: str = None,
        quantiles_name: str = None,     # overwrites  `y_hat_name + "_quantiles"`
        model_name: str = "binary_model",
        *args, **kwargs     # passed to DataFrame init
    ):

        bc = BinaryCountsByScoresQuantiles(
            y, y_hat, q_order, ones_label, y_name, y_hat_name, quantiles_name, model_name,
            *args, **kwargs
        )
        bcr = bc[::-1]

        bcr[bc.y_name] = bcr[bc.y_name].cumsum() / bc.nobs1

        super().__init__(
            bcr[bc.quantiles_name], bcr[bc.y_name],
            *args, **kwargs     # passed to DataFrame init
        )

        self.quantiles_name = bc.quantiles_name
        self.y_name = bc.y_name
        self.y_hat_name = bc.y_hat_name

        self.q_order = bc.q_order
        self.nobs = bc.nobs
        self.nobs1 = bc.nobs1
        self.model_name = bc.model_name


class Lift(BinaryStatsByScoresQuantiles):

    def __init__(
        self, y: Sequence, y_hat: Sequence[float],
        q_order: int = 10,
        ones_label: int | str = 1,      # all other labels are considered belonging to class 0
        y_name: str = None,
        y_hat_name: str = "score",
        quantiles_name: str = None,     # overwrites  `y_hat_name + "_quantiles"`
        model_name: str = "binary_model",
        *args, **kwargs     # passed to DataFrame init
    ):

        bc = BinaryCountsByScoresQuantiles(
            y, y_hat, q_order, ones_label, y_name, y_hat_name, quantiles_name, model_name,
            *args, **kwargs
        )
        bcr = bc[::-1]

        cum_random_model_gain = np.linspace(0, bc.nobs1, len(bcr) + 1)[1:]
        bcr[bc.y_name] = bcr[bc.y_name].cumsum() / cum_random_model_gain

        super().__init__(
            bcr[bc.quantiles_name], bcr[bc.y_name],
            *args, **kwargs     # passed to DataFrame init
        )

        self.quantiles_name = bc.quantiles_name
        self.y_name = bc.y_name
        self.y_hat_name = bc.y_hat_name

        self.q_order = bc.q_order
        self.nobs = bc.nobs
        self.nobs1 = bc.nobs1
        self.model_name = bc.model_name
