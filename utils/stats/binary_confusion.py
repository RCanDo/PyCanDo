#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Binary Confusion Matrix
version: 1.0
type: module
keywords: [confusion matrix, binary classifier, ...]
description: |
    Binary Confusion Matrix as pd.DataFrame
    with nice rows and columns names.
    All derivatives easily obtainable through proper methods.
remarks:
todo:
sources:
    - link: https://en.wikipedia.org/wiki/Confusion_matrix
file:
    date: 2022-01-19
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%

from functools import partialmethod
from typing import Union, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# to be used as partialmethod to extend pd.DataFrame
def ravel_binary_confusion(
    self: pd.DataFrame, to: str = "dict"
) -> Union[dict, list, tuple, np.array, pd.Series, pd.DataFrame]:
    """
    Meant only for BINARY confusion matrix with order of levels (F,T) or (-,+) or (0,1)
    and where  real  valus goes along 0-axis (rows idx) and  predicted  values goes along 1-axis (columns idx).
    """
    TN = self.iloc[0, 0]
    FP = self.iloc[0, 1]
    FN = self.iloc[1, 0]
    TP = self.iloc[1, 1]
    if to:
        if isinstance(to, bool):
            to = "dict"
        #
        if to == "dict":
            conf = {"TN": TN, "FP": FP, "FN": FN, "TP": TP}
        elif to == "list":
            conf = [TN, FP, FN, TP]
        elif to == "tuple":
            conf = (TN, FP, FN, TP)
        elif to in ["nparray", "array"]:
            pass
        elif to in ["series", "pandas"]:
            conf = pd.Series({"TN": TN, "FP": FP, "FN": FN, "TP": TP})
    else:
        conf = self
    return conf


class BinaryConfusion(pd.DataFrame):
    """
    DataFrame for binary confusion matrix.
    Also with methods for most important derivatives like  accuracy, sensitivity, f1, ...
    so there's no need to load more helpers from scikit.

    Uses  sklearn.metrics.confusion_matrix  which has the following order of elements:
     predicted     0    1
     real      0   TN   FP
               1   FN   TN
    """

    def __init__(
        self, yy: Sequence, yy_hat: Sequence,
        as_int: bool = True,
        names: tuple[str] = ('0', '1'),
        model_name: str = "binary_model",
        *args, **kwargs
    ):

        if len(yy) != len(yy_hat):
            raise Exception("Length of `yy` and `yy_hat` cannot differ.")

        if as_int:
            try:
                yy = yy.astype(int)
                yy_hat = yy_hat.astype(int)
            except Exception as e:
                print(e)

        conf = confusion_matrix(yy, yy_hat)
        super().__init__(conf, *args, **kwargs)
        #
        self.index = list(names)
        self.columns = list(names)
        self.rename_axis(index="real", columns="prediction", inplace=True)
        self.model_name = "binary_model"
        #
        self.n_obs = self.sum().sum()

        self.TN, self.FP, self.FN, self.TP = self.ravel("tuple")

    ravel = partialmethod(ravel_binary_confusion)

    def to_dict(self):
        return self.ravel("dict")

    def to_series(self):
        return self.ravel("series")

    def to_list(self):
        return self.ravel("list")

    def to_tuple(self):
        return self.ravel("tuple")

    def derivatives(
        self, as_series: bool = False,
        stats: tuple[str] = None,
        name: str = None,
    ) -> Union[dict, pd.Series]:
        """"""
        derivs = {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "sensitivity": self.sensitivity,    # TPR
            "specificity": self.specificity,    # 1 - FPR
            "fpr": self.FPR,
            "roc_ratio": self.roc_ratio,        # TPR / FPR
            "precision": self.precision,
            "fdr": self.false_discovery_rate,   # false discovery rate
            "nobs": self.n_obs
        }
        stats = derivs.keys() if stats is None else stats
        derivs = {k: derivs[k] for k in stats}
        if as_series:
            derivs = pd.Series(derivs, name=(name or self.model_name))
        return derivs

    @property
    def sensitivity(self):
        """TPR, recall"""
        return self.TP / (self.TP + self.FN)

    @property
    def TPR(self):
        return self.sensitivity

    @property
    def specificity(self):
        """TNR = 1 - FPR"""
        return self.TN / (self.FP + self.TN)

    @property
    def FPR(self):
        return 1 - self.specificity

    @property
    def positive_likelihood_ratio(self):
        return self.sensitivity / (1 - self.specificity)    # TPR / FPR

    @property
    def roc_ratio(self):
        return self.positive_likelihood_ratio

    @property
    def precision(self):
        """Positive predictive value"""
        return self.TP / (self.TP + self.FP)

    @property
    def accuracy(self):
        return (self.TP + self.TN) / self.n_obs

    @property
    def false_discovery_rate(self):
        return 1 - self.precision

    @property
    def f1(self):
        return 2 * self.TP / (2 * self.TP + self.FN + self.FP)


ConfusionMatrix = BinaryConfusion
