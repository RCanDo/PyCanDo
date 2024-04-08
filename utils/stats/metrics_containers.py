#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Container classes for metrics for regression and binary models
version: 1.0
type: module
keywords: [performance metrics, regression model, binary classifier, ...]
description: |
    Container classes for metrics for regression and binary models -- ModelsReg, ModelsBin --
    based on test and train data.
    Common interface (almost) and methods, espacially those helpful for reporting like
    gathering all statistics (metrics) into  dict, pd.Series, list, tuple.
    Also, common pretty representation.
remarks:
todo:
    - add container for general classifier model: MetricsClas
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
# import json
# import yaml
from typing import Any, Sequence
import abc

import numpy as np
import pandas as pd

# from scipy.stats import entropy
from sklearn.metrics import r2_score, roc_auc_score  # , roc_curve,
from sklearn.metrics import mean_squared_error, explained_variance_score
# from sklearn.model_selection import cross_val_score
from utils.stats import ConfusionMatrix

import utils.builtin as bi


# %%
class TrainTestMetric(bi.Repr):
    """
    Simple container with pretty representation (inherits from Repr)
    for one metric on model performance
    in two cases: for train and test data.
    Only two attributes: 'train' and 'test' for storing relevant metric values,
    which may be accsessed via indexing, i.e. string name.
    Methods for representing these data as list, dictionary and pd.Series.
    Notice that some metrics may be more then one number, e.g. confusion matrix.
    """
    def __init__(self):
        self.test: Any = None
        self.train: Any = None

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setitem__(self, name, value):
        self.__dict__[name] = value

    def to_dict(self) -> dict:
        """Represent data as dict, recursively (e.g. in case of confusion matrix)
        """
        def to_dict(x):
            try:
                return x.to_dict()
            except AttributeError:
                return x
        dic = {k: to_dict(v) for k, v in self.__dict__.items()}
        return dic

    def to_dict_orig(self) -> dict:
        """Represent data as dict but only one level, e.g. matrices are left intact
        """
        return dict(test=self.test, train=self.train)

    def to_series(self) -> pd.Series:
        """Represent data as pd.Series but only one level, e.g. matrices are left intact
        """
        ss = pd.Series([self.test, self.train], index=("test", "train"))
        return ss

    def to_list(self) -> list:
        """Represent data as list but only one level, e.g. matrices are left intact
        """
        return [self.test, self.train]


class Metrics(abc.ABC, bi.Repr):
    """
    Abstract container for many metrics on model performance,
    each stored in TrainTestMetric.
    It is base class for container for metrics on regression model, MetricsReg, and binary model, MetricsBin.
    It gives pretty representation, elements access via indexing,
    and methods for representing data as dict, pd.Series and pd.DataFrame.
    Also forces to implement .compute() method for computing all metrics from provided data.
    """

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def to_dict(self, stats: Sequence[str] | None = None) -> dict:
        """
        Represent data as dict recursively, relying on .to_dict() method of each level.
        stats: Sequence[str] | None= None
            which metrics to take; if None takes all;
        """
        def to_dict(x):
            try:
                return x.to_dict()
            except AttributeError:
                return x
        if stats is None:
            stats = [k for k in self.__dict__.keys() if not k.startswith("_")]
        dic = {k: to_dict(v) for k, v in self.__dict__.items() if (k in stats and v is not None)}
        return dic

    def to_series(self, stats: Sequence[str] | None = None) -> pd.Series:
        """
        Represent data as pd.Series recursively, except 'confusion',
        relying on .to_series() method of each metric (TestTrainMetric).
        stats: Sequence[str] | None = None
            which metrics to take; if None takes all except 'confusion' (because it's matrix');
        """
        if stats is None:
            stats = [k for k in self.__dict__.keys() if k != "confusion" and not k.startswith("_")]
        ss = pd.concat(
            {k: v.to_series() for k, v in self.__dict__.items() if (k in stats and v is not None)},
            axis=0
        )
        return ss

    def to_frame(self, stats: Sequence[str] | None = None) -> pd.DataFrame:
        """
        Represent data as pd.DataFrame
        where one column is for 'test' and second is for 'train' versions of metrics;
        relies on .to_series() method of each metric (TestTrainMetric).
        stats: Sequence[str] | None = None
            which metrics to take; if None takes all except 'confusion' (because it's matrix');
        """
        df = self.to_series(stats).unstack()
        return df

    @abc.abstractmethod
    def compute(
        self,
        y_train: pd.Series,
        y_train_hat: pd.Series,
        y_test: pd.Series = None,
        y_test_hat: pd.Series = None,
    ) -> None:
        """
        Compute all metrics (stats) for provided data;
        """


class MetricsReg(Metrics):
    """
    Container for metrics on performance of regression model.
    Inherits from Metrics thus is endowed with pretty representation, elements access via indexing,
    and methods for representing data as dict, pd.Series and pd.DataFrame.
    Object is empty at initialisation.
    One needs to run
        obj.compute(y_train, y_train_hat, y_test, y_test_hat)
    to fill it with metrics values, hence it is up to user securing consistency of data provided
    (that they are for the same data and model).
    y_test, y_test_hat  may be left empty.
    """

    STATS = {
        "evs": explained_variance_score,    # Explained Variance Score
        "r2": r2_score,                     # adjusted R2 (coeff. of determination)
        "rmse": lambda y, y_hat: np.sqrt(mean_squared_error(y, y_hat)),     # Residual Mean Squared Error
        "nobs": lambda y, _: len(y)
        # add more
    }

    def __init__(
        self,
        stats: tuple[str] = None,
        *args, **kwargs,
    ):
        """
        stats: tuple[str] = None,
            which metrics (stats) to consider;
            if None takes all predefined in MetricsReg.STATS;
        """
        self._stats = stats if stats is not None else list(MetricsReg.STATS.keys())

        for k in self._stats:
            self.__dict__[k] = TrainTestMetric()

    def compute(
        self,
        y_train: pd.Series,
        y_train_hat: pd.Series,
        y_test: pd.Series = None,
        y_test_hat: pd.Series = None,
    ) -> None:
        """
        Compute all metrics (stats) for provided data;
        y_train: pd.Series,
            target for train data
        y_train_hat: pd.Series,
            prediction of the target for train data
        y_test: pd.Series = None,
            target for test data
        y_test_hat: pd.Series = None,
            prediction of the target for test data
        """
        for k in self._stats:
            self.__dict__[k].train = MetricsReg.STATS[k](y_train, y_train_hat)

        if y_test is not None:
            for k in self._stats:
                self.__dict__[k].test = MetricsReg.STATS[k](y_test, y_test_hat)


class MetricsBin(Metrics):
    """
    Container for metrics on performance of binary classification model.
    Inherits from Metrics thus is endowed with pretty representation, elements access via indexing,
    and methods for representing data as dict, pd.Series and pd.DataFrame.
    Object is empty at initialisation.
    One needs to run
        obj.compute(y_train, y_train_hat, y_test, y_test_hat)
    to fill it with metrics values, hence it is up to user securing consistency of data provided
    (that they are for the same data and model).
    y_test, y_test_hat  may be left empty.
    """

    STATS = [
        "auc", "gini",
        "confusion", "accuracy", "f1", "sensitivity", "specificity", "fpr", "roc_ratio", "precision", "fdr", "nobs",
    ]

    def __init__(
        self,
        stats: tuple[str] = None,
        threshold: float = .5,
    ):
        """
        stats: tuple[str] = None,
            which metrics (stats) to consider;
            if None takes all listed in MetricsBin.STATS;
        threshold: float = .5,
            binarisation threshold for raw predicttions from binary model;
        """

        self._stats = stats if stats is not None else MetricsBin.STATS
        self._threshold = threshold

        if "auc" in self._stats or "gini" in self._stats:
            self.auc = TrainTestMetric()
            self.gini = TrainTestMetric()
        self.confusion = TrainTestMetric()  # regardles of stats

        for k in self._stats:
            self.__dict__[k] = TrainTestMetric()

        if "nobs" in self._stats:
            self.nobs0 = TrainTestMetric()
            self.nobs1 = TrainTestMetric()

    def compute(
        self,
        y_train: pd.Series,
        y_train_hat: pd.Series,
        y_test: pd.Series = None,
        y_test_hat: pd.Series = None,
        threshold: float = None,
    ) -> None:
        """
        Compute all metrics (stats) for provided data;
        y_train: pd.Series,
            target for train data
        y_train_hat: pd.Series,
            prediction of the target for train data;
            this must be raw prediction (.predict_proba) before applying threshold;
        y_test: pd.Series = None,
            target for test data
        y_test_hat: pd.Series = None,
            prediction of the target for test data
            this must be raw prediction (.predict_proba) before applying threshold;
        threshold: float = None,
            threshold value for determining which prediction may be considered 1 and which 0:
            y_hat > threshold -> 1.
        """
        if threshold:
            self._threshold = threshold
        else:
            threshold = self._threshold

        self._compute_0('train', y_train, y_train_hat, threshold)

        if y_test is not None:
            self._compute_0('test', y_test, y_test_hat, threshold)

    def _compute_0(
        self,
        mode: str,
        y: pd.Series,
        y_hat: pd.Series,
        threshold: float = .5,
    ) -> None:
        """
        Helper for .compute() method, depends on self.stats.
        For true values of a target y and it's prediction from a model y_hat,
        computes all the metrics relative to the type of target (and model) binary or continuous.
        mode: str
            'train' or 'test'
        """
        if "auc" in self._stats or "gini" in self._stats:
            auc = roc_auc_score(y, y_hat)
            self.auc[mode] = auc
            self.gini[mode] = 2 * auc - 1

        if "nobs" in self._stats:
            self.nobs0[mode] = (y == 0).sum()
            self.nobs1[mode] = (y == 1).sum()

        y_hat = self.binarise(y_hat, threshold)     # only _hat needs binarisation
        self.confusion[mode] = ConfusionMatrix(y, y_hat)

        stats = tuple(set(self._stats).difference({'confusion', 'auc', 'gini'}))
        values = self.confusion[mode].derivatives(stats=stats).values()
        for k, v in zip(stats, values):
            self.__dict__[k][mode] = v

    def binarise(
        self,
        y_hat: pd.Series,
        threshold: float,
        as_int: bool = True,
    ) -> pd.Series:
        """
        Applying threshold value for determining which prediction may be considered 1 and which 0.
        y_hat: pd.Series,
            raw predictions for binary target;
        threshold: float,
        as_int: bool = True,
            if True then condition `y_hat > threshold` is turned into integer;
            otherwise left as bool;
        """
        y_hat = y_hat > threshold
        if as_int:
            y_hat = y_hat.astype(int)
        return y_hat

    def confusion_df(
        self,
        concat: int = 0,
    ) -> pd.DataFrame:
        """
        Returns confusion matrices for both test and train data concatenated into one matrix.
        concat: int; 0 or 1
            along with which axis to concatenate confusion matrices for test and train
        """
        confusion = pd.concat(self.confusion.to_dict_orig(), axis=concat)
        return confusion
