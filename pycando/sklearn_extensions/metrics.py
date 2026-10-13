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

import warnings
from typing import Any, Optional, cast

import numpy as np

warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.model_selection import GridSearchCV  # type: ignore # [import-untyped]
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix  # type: ignore # [import-untyped]
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # type: ignore # [import-untyped]


def classifier_quality(
        model: Any,
        X_test: Optional[np.ndarray | pd.DataFrame] = None,
        Y_test: Optional[np.ndarray | pd.DataFrame] = None
) -> None:
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


def binary_classifier_quality(
        model: Any,
        X_test: Optional[np.ndarray | pd.DataFrame] = None,
        Y_test: Optional[np.ndarray | pd.DataFrame] = None
) -> None:
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

    Y_hat: np.ndarray = model.predict(X_test)
    print("Confusion matrix (true x pred):")
    print(confusion_matrix(Y_test, Y_hat))
    print("Sensitivity: {:f}".format(np.sum(Y_hat[Y_test == 1]) / np.sum(cast(np.ndarray, Y_test))))
    print("Specificity: {:f}".format(np.sum(1 - Y_hat[Y_test == 0]) / np.sum(Y_test == 0)))
    print("Accuracy score on test data: {:f}".format(accuracy_score(Y_test, Y_hat)))
    print("F1 score on test data: {:f}".format(f1_score(Y_test, Y_hat)))

    ConfusionMatrixDisplay.from_predictions(Y_test, Y_hat)
    RocCurveDisplay.from_estimator(model.best_estimator_, X_test, Y_test)
