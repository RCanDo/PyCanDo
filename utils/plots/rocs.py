#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
title: Receiver Operating Characteristic (ROC)
version: 1.0
type: sub-module
keywords: [ROC, AUROC, Gini coefficient]
description: |
content:
remarks:
todo:
sources:
    - title: Receiver Operating Characteristic (ROC)
      link: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc
file:
    name: rocs.py
    date: 2022-11-13
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""

# %%
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from itertools import cycle

from sklearn import svm  # , datasets
from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.metrics import roc_auc_score

from matplotlib.colors import ListedColormap  # LinearSegmentedColormap


# %%
def rocs(variable, factor, method="direct"):
    """"""
    classes = np.unique(factor)
    n_classes = len(classes)

    # get scores
    if n_classes > 2:
        fpr = dict()
        tpr = dict()
        thresh = dict()     # we don't need it but good to remember
        auroc = dict()

        y = label_binarize(factor.astype(str), classes=classes)
        if method == "direct":
            x = np.array(variable, ndmin=2).T
            y_score = np.tile(x, (1, n_classes))
        else:
            classifier = OneVsRestClassifier(
                svm.SVC(kernel="linear", probability=True)      # xgboost would do better
            )
            y_score = classifier.fit(x, y).decision_function(x)  # .predict()
            # y ~ x (model) -> y_score -> y_hat
        #
        for c in classes:
            fpr[c], tpr[c], thresh[c] = roc_curve(y[:, c], y_score[:, c])
            auroc[c] = auc(fpr[c], tpr[c])

    else:
        if method == "direct":
            y = np.array(factor.astype(int))
            y_score = np.array(variable)
        else:
            classifier = svm.SVC(kernel="linear", probability=True)      # xgboost would do better
            y_score = classifier.fit(x, y).decision_function(x)  # .predict()
        fpr, tpr, thresh = roc_curve(y, y_score)
        auroc = auc(fpr, tpr)

    return fpr, tpr, thresh, auroc


# %%
def cats_and_colors(factor, most_common=None, cmap="Set2"):
    """helper function
    most_common : None; pd.Series
        table of most common value counts = variable.value_counts[:most_common]
        got from  to_factor(..., most_common: int)
    Returns
    -------
    cats : categories (monst common) of a factor
    cat_colors : list of colors of length len(cats)
    cmap : listed color map as defined in matplotlib
    """
    cats = factor.cat.categories.to_list()
    if most_common is not None:
        cats = [cat for cat in cats if cat in most_common]
    #
    cat_colors = plt.colormaps[cmap](np.linspace(0.1, 0.9, len(cats)))
    cmap = ListedColormap(cat_colors)
    return cats, cat_colors, cmap


def plot_rocs(ax, variable, factor, cats, cat_colors, cmap):
    """"""
    fpr, tpr, thresh, auroc = rocs(variable, factor)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.plot([0, 1], [0, 1], color="darkgray", lw=2, linestyle=":")

    if len(cats) > 2:
        for cat, col in reversed(list(zip(cats, cat_colors))):
            ax.plot(fpr[cat], tpr[cat], color=col)
    else:
        ax.plot(fpr, tpr, color=cat_colors[1], label="True")

    ax.legend(loc="lower right")

    return fpr, tpr, thresh, auroc


# %%
