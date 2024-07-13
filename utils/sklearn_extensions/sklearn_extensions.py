#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: sklearn extensions
version: 1.0
type: module
keywords: [sklearn, extension, selector, threshold]
description: |
content:
    -
remarks:
todo:
sources:
    - link: https://stackoverflow.com/questions/28296670/remove-a-specific-feature-in-scikit-learn
file:
    usage:
        interactive: False
        terminal: False
    date: 2021-11-05
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import datetime as dt

from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


# %%
def classifier_quality(model, X_test=None, Y_test=None):
    """
    Meant for regression classification.
    If `model` is Grid it uses best model.
    """
    if isinstance(model, GridSearchCV):
        result = pd.DataFrame(
            {k: model.cv_results_[k] for k in
                 ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']})
        print(result)
        print()
        print(f"best params: {model.best_params_}")
        print("best score: {:f}".format(model.best_score_))
        print()

    #    Y_hat = model.predict(X_test)


# %%
def binary_classifier_quality(model, X_test, Y_test):
    """
    Meant for binary classification.
    If `model` is Grid it uses best model.
    """
    if isinstance(model, GridSearchCV):
        result = pd.DataFrame(
            {k: model.cv_results_[k] for k in
                 ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']})
        print(result)
        print()
        print(f"best params: {model.best_params_}")
        print("best score: {:f}".format(model.best_score_))
        print()

    Y_hat = model.predict(X_test)
    print("Confusion matrix (true x pred):")
    print(confusion_matrix(Y_test, Y_hat))
    print("Sensitivity: {:f}".format(sum(Y_hat[Y_test == 1]) / sum(Y_test)))
    print("Specificity: {:f}".format(sum(1 - Y_hat[Y_test == 0]) / sum(Y_test == 0)))
    print("Accuracy score on test data: {:f}".format(accuracy_score(Y_test, Y_hat)))
    print("F1 score on test data: {:f}".format(f1_score(Y_test, Y_hat)))

    # print(confusion_matrix(grid.predict(X_test), y_test))
    ConfusionMatrixDisplay.from_predictions(Y_test, Y_hat)
    RocCurveDisplay.from_estimator(model.best_estimator_, X_test, Y_test)


# %%
class ManualFeatureSelector(TransformerMixin):
    """
    source: https://stackoverflow.com/questions/28296670/remove-a-specific-feature-in-scikit-learn
    Transformer for manual selection of features.
    """

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.features]

    def fit_transform(self, X, y=None):
        return self.transform(X)


# %%
# class ManualFeatureExcluder(TransformerMixin):
#  moved to common/models


# %%
class ElapsedMonths(TransformerMixin):
    """
    Transforms dates into months elapsed
    since given date to another specified date `upto` (defult `datetime.date.today()`).
    Dates may be given in arbitrary format `format` which could be read by `datetime.date.strptime(., format)`.
    Specified date `upto` (if different from default) must be given in ISO format "YYYY-MM-DD".
    """
    def __init__(self, features, upto=None, format='%Y-%m-%d', nans=np.nan):
        """
        features   list of feature names or single feature name
        upto
        """
        self.features = features if isinstance(features, list) else [features]
        self.upto = dt.date.today() if upto is None else dt.date.fromisoformat(upto)
        self.year = self.upto.year
        self.month = self.upto.month
        self.nans = nans
        self.format = format

    def fit(self, X, y=None):
        """ coud be used to set up better default `upto` date.
        """
        return self

    def _months_elapsed(self, since):
        try:
            since = dt.datetime.strptime(since, self.format)
            months = (self.year - since.year) * 12 + self.month - since.month
        except Exception:
            months = self.nans
        return months

    def transform(self, X):
        features = list(set(X.columns.tolist()).intersection(self.features))
        X[features] = X[features].applymap(self._months_elapsed)
        return X

    def fit_transform(self, X, y=None):
        return self.transform(X)


# %%
class NullsThreshold(TransformerMixin):
    """
    """

    def __init__(self, threshold=.2):
        """
        threshold : float in [0, 1]
        """
        if not 0. <= threshold <= 1.:
            raise Exception('`threshold` must be float between 0 and 1.')

        self.threshold = threshold
        self.features = None
        self.N = None

    def _nans_ratio(self, feature):
        N = self.N if self.N else len(feature)
        return (N - feature.count()) / N

    def fit(self, X, y=None):
        self.N = X.shape[0]
        self.features = [c for c in X.columns if self._nans_ratio(X[c]) <= self.threshold]
        return self

    def transform(self, X):
        return X.loc[:, self.features]

    def fit_transform(self, X, y=None):
        _ = self.fit(X, y)
        return self.transform(X)


# %%
class MinUniqueValues(TransformerMixin):
    """
    Leaves only features (variables) for which there are at least
    `threshold` unique values.
    """

    def __init__(self, threshold=2, dtypes=None):
        """
        threshold : int
        dtypes : List[str]
            list of dtypes for which the min unique value threshold will be applied;
            E.g. one may wish to retain only those numeric features which have at least `threshold=99` values:
            > MinUniqueValues(99, dtypes=['float64', 'float32', 'int64', 'int32'])

        Remarks
        It's expensive!
        NaNs are not counted as values.
        """
        self.threshold = threshold
        self.dtypes = dtypes
        self.features = None

    def _check_uniques(self, feature):

        if self.dtypes:
            if feature.dtype in self.dtypes:
                res = len(feature.dropna().unique()) >= self.threshold
            else:
                res = True
        else:
            res = len(feature.dropna().unique()) >= self.threshold
        return res

    def fit(self, X, y=None):

        if X.shape[0] < self.threshold:
            warnings.warn(f'`threshold` is set to {self.threshold} what is more  then `len(data)`.\n'
                          'Filtering formula cannot be applied - all columns passed.')
            self.features = X.columns
        else:
            self.features = [c for c in X.columns if self._check_uniques(X[c])]

        return self

    def transform(self, X):
        return X.loc[:, self.features]

    def fit_transform(self, X, y=None):
        _ = self.fit(X, y)
        return self.transform(X)


# %%
class MostCommonThreshold(TransformerMixin):
    """
    Leaves only features (variables) for which the most common value
    is less frequent then `threshold`.
    """

    def __init__(self, threshold=.9, dtypes=None):
        """
        threshold : float in [0, 1]
        dtypes : List[str]
            list of dtypes for which the most common value threshold will be applied;
            E.g. one may wish to retain only those numeric features
            for which the most common value is no more then 10% of all observations:
            > MostCommonThreshold(.1, dtypes=['float64', 'float32', 'int64', 'int32'])

        Remarks
        It's expensive!
        NaNs are not counted as values.             # !!!???
        """
        if not 0. <= threshold <= 1.:
            raise Exception('`threshold` must be float between 0 and 1.')

        self.threshold = threshold
        self.dtypes = dtypes
        self.features = None
        self.N = None

    def _most_common_ratio(self, feature: pd.Series):
        """"""
        N = self.N if self.N else len(feature)
        mc = feature.value_counts().max()
        return mc / N

    def _below_most_common_ratio(self, feature: pd.Series):
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

    def fit(self, X: pd.DataFrame, y=None):
        self.N = X.shape[0]

        if 1 / self.N >= self.threshold:
            warnings.warn(f'`threshold` is set to {self.threshold} what is less then `1 / len(data)`.\n'
                          'Filtering formula cannot be applied - all columns passed.')
            self.features = X.columns
        else:
            self.features = [c for c in X.columns if self._below_most_common_ratio(X[c])]

        return self

    def transform(self, X: pd.DataFrame):
        return X.loc[:, self.features]

    def fit_transform(self, X: pd.DataFrame, y=None):
        _ = self.fit(X, y)
        return self.transform(X)


# %%
