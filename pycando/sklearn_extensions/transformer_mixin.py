#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: sklearn extensions
version: 1.0
type: module
keywords: [sklearn, extension, TransformerMixin, BaseEstimator, selector, threshold]
description: |
content:
    -
remarks:
todo:
sources:
    - link: https://stackoverflow.com/questions/28296670/remove-a-specific-feature-in-scikit-learn
file:
    date: 2021-11-05
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

from __future__ import annotations
import datetime as dt
import warnings
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator  # type: ignore # [import-untyped]   # ClassifierMixin


class ManualFeatureSelector(TransformerMixin):
    """
    source: https://stackoverflow.com/questions/28296670/remove-a-specific-feature-in-scikit-learn
    Transformer for manual selection of features.
    """

    def __init__(self, features: Iterable[str]):
        """
        features: list[str]
            features (column names) to retain
        """
        self.features = pd.Index(features)

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> ManualFeatureSelector:
        self.features = X.columns.intersection(self.features)
        if len(self.features) == 0:
            raise Exception('There is no feature from `features` within data provided – data would become empty.')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.features is not None
        return X.loc[:, self.features]  # type: ignore  # nvalid index type ?

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        _ = self.fit(X, y)
        return self.transform(X)


class ManualFeatureExcluder(TransformerMixin, BaseEstimator):  # type: ignore
    """
    source: https://stackoverflow.com/questions/28296670/remove-a-specific-feature-in-scikit-learn
    Transformer for manual exclusion of features.
    """

    def __init__(self, exclude: list[str]):
        """
        exclude: list[str]
            features (column names) to exclude
        """
        self.exclude = exclude

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> ManualFeatureExcluder:
        self.features = X.columns.difference(self.exclude)
        if len(self.features) == 0:
            raise Exception('No columns left within data provided after excluding those from `exclude`.')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.features is not None
        return X.loc[:, self.features]  # type: ignore # [index]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        _ = self.fit(X, y)
        return self.transform(X)


class MinUniqueValues(TransformerMixin, BaseEstimator):
    """
    Leaves only features (variables) for which there are at least
    `threshold` unique values.
    """

    def __init__(self, threshold: int = 2, dtypes: Optional[list[str]] = None):
        """
        Arguments
        ---------
        threshold : the minimum number of unique values for a feature to be retained
        dtypes : list of dtypes for which the min unique value threshold will be applied;
            E.g. one may wish to retain only those numeric features which have at least `threshold=99` values:
            > MinUniqueValues(99, dtypes=['float64', 'float32', 'int64', 'int32'])

        Remarks
        -------
        It's expensive!
        NaNs are not counted as values.
        """
        self.threshold = threshold
        self.dtypes = dtypes

    def _check_uniques(self, feature: pd.Series) -> bool:

        if self.dtypes:
            if feature.dtype in self.dtypes:
                res = len(feature.dropna().unique()) >= self.threshold
            else:
                res = True
        else:
            res = len(feature.dropna().unique()) >= self.threshold
        return res

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> MinUniqueValues:

        if X.shape[0] < self.threshold:
            warnings.warn(f'`threshold` is set to {self.threshold} what is more  then `len(data)`.\n'
                          'Filtering formula cannot be applied - all columns passed.')
            self.features = X.columns
        else:
            self.features = pd.Index([c for c in X.columns if self._check_uniques(X[c])])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.features]  # type: ignore # [index]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        _ = self.fit(X, y)
        return self.transform(X)


class MostCommonThreshold(TransformerMixin, BaseEstimator):
    """
    Leaves only features (variables) for which the most common value
    is less frequent then `threshold`.
    """

    def __init__(self, threshold: float = .9, dtypes: Optional[list[str]] = None):
        """
        Arguments
        ---------
        threshold : minimum ratio of the most common value to all observations
        dtypes : list of dtypes for which the most common value threshold will be applied;
            E.g. one may wish to retain only those numeric features
            for which the most common value is no more then 10% of all observations:
            > MostCommonThreshold(.1, dtypes=['float64', 'float32', 'int64', 'int32'])

        Remarks
        -------
        It's expensive!
        NaNs are not counted as values.             # !!!???
        """
        if not 0. <= threshold <= 1.:
            raise Exception('`threshold` must be float between 0 and 1.')

        self.threshold = threshold
        self.dtypes = dtypes

    def _most_common_ratio(self, feature: pd.Series) -> float:
        """"""
        mc = feature.value_counts().max()
        return mc / self.N

    def _below_most_common_ratio(self, feature: pd.Series) -> bool:
        """
        feature: pd.Series;
        below or EQUAL to threshold
        """
        if self.dtypes:
            if feature.dtype in self.dtypes:
                res = self._most_common_ratio(feature) <= self.threshold
            else:
                res = True
        else:
            res = self._most_common_ratio(feature) <= self.threshold
        return res

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> MostCommonThreshold:
        self.N = X.shape[0]

        if 1 / self.N >= self.threshold:
            warnings.warn(f'`threshold` is set to {self.threshold} what is less then `1 / len(data)`.\n'
                          'Filtering formula cannot be applied - all columns passed.')
            self.features = X.columns
        else:
            self.features = pd.Index([c for c in X.columns if self._below_most_common_ratio(X[c])])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.features is not None
        return X.loc[:, self.features]  # type: ignore # [index]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        _ = self.fit(X, y)
        return self.transform(X)


class ElapsedMonthsDT(TransformerMixin, BaseEstimator):
    """ !!!  REVISIN REQUIRED
    Transforms dates into months elapsed
    since given date to another specified date `date_stop` (defult `datetime.date.today()`).
    Dates may be given in arbitrary format `format` which could be read by `datetime.date.strptime(., format)`.
    Specified date `date_stop` (if different from default) must be given in ISO format "YYYY-MM-DD".
    """

    def __init__(
            self,
            features: Iterable[str] | str,
            date_stop: Optional[str | dt.date] = None,
            format: str = '%Y-%m-%d',
            nans: float = np.nan
    ):
        """
        Arguments
        ---------
        features: Iterable[str] | str,
            column names with dates (pd.Timestamp) to be transformed to the time elapsed
            since these dates to dates in `date_stop` column;
        date_stop: Optional[str | dt.date] = None,
            date passed either as str in ISO format "YYYY-MM-DD" or as datetime.date,
            to which elapsed time (in month) from dates in `features` will be measured.
        format: str = '%Y-%m-%d',
            format in which dates in `features` are passed;
        nans: float = np.nan
        """
        self.features = features
        self.date_stop = date_stop
        self.format = format
        self.nans = nans

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> ElapsedMonthsDT:
        """ could be used to set up better default `date_stop` date.
        """
        self.date_stop = dt.date.today() if self.date_stop is None else dt.date.fromisoformat(str(self.date_stop))
        self.year = self.date_stop.year
        self.month = self.date_stop.month
        return self

    def _months_elapsed(self, since: str) -> float:
        months: float
        try:
            since_dt = dt.datetime.strptime(since, self.format)
            months = (self.year - since_dt.year) * 12 + self.month - since_dt.month
        except Exception:   # whatever
            months = self.nans
        return months

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._features = [self.features] if isinstance(self.features, str) else self.features
        self._features = X.columns.intersection(pd.Index(self._features))
        stats = [X[c].applymap(self._months_elapsed) for c in self._features]
        stats = pd.concat(stats, axis=1)
        return pd.DataFrame(stats)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.transform(X)


class ElapsedMonthsPD(TransformerMixin, BaseEstimator):  # type: ignore
    """
    Transforms dates into months elapsed
    since date given in `features` to another specified date `date_stop` given in another column.
    All dates taken from columns of `X` assumed to be pd.Timestamp (pd.datetime format).
    """

    def __init__(self, features: list[str] | str, date_stop: str, precision: int = 0):
        """
        Arguments
        ---------
        features: list[str] | str,
            column names with dates (pd.Timestamp) to be transformed to the time elapsed
            since these dates to dates in `date_stop` column;
        date_stop: str
            name of a column having pd.datetime format ('datetime64[ns]') containing for each record a date
            to which elapsed time (in month) from dates in `features` will be measured, i.e.
            `(X[date_stop] - X[c]).dt.days / 30  for c in features`.
        """
        self.features = features
        self.date_stop = date_stop
        self.precision = precision

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> ElapsedMonthsPD:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._features = self.features if isinstance(self.features, list) else [self.features]
        self._features = X.columns.intersection(self._features)  # type: ignore
        stats = [(X[self.date_stop] - X[c]).dt.days / 30 for c in self._features]
        stats = pd.concat([ss.round(self.precision) for ss in stats], axis=1)
        return stats

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return self.transform(X)


class NullsThreshold(TransformerMixin, BaseEstimator):
    """
    Leaves only features (variables) for which the ratio of NaNs is less than `threshold`.
    """

    def __init__(self, threshold: float = .2):
        """
        Arguments
        ---------
        threshold : float in [0, 1] - the maximum ratio of NaNs in a feature
        """
        if not 0. <= threshold <= 1.:
            raise Exception('`threshold` must be float between 0 and 1.')
        self.threshold = threshold

    def _nans_ratio(self, feature: pd.Series) -> float:
        return (self.N - feature.count()) / self.N

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> 'NullsThreshold':
        self.N = len(X)
        self.features = [c for c in X.columns if self._nans_ratio(X[c]) <= self.threshold]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.features is not None
        return X.loc[:, self.features]  # type: ignore # [index]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        _ = self.fit(X, y)
        return self.transform(X)
