#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Project specific helpers
version: 1.0
type: module
keywords: [helper functinos, utilieties, subroutines, ...]
description: |
    Project sepecific helpers, used by ModelSpec class or using it.
content:
remarks:
todo:
sources:
file:
    date: 2022-10-11
    authors:
        - fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@quantup.pl
"""
# %%
from typing import Iterable
from copy import deepcopy
import dill as pickle
import pathlib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import xgboost as xgb

# from pygam import LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, s, f, l, te # InvGauss
# from pygam.terms import TermList

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, roc_curve, r2_score, mean_squared_error, confusion_matrix
# from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score

from utils.time_data import WindowsAtMoments
# import utils.df as cdf
import utils.plots as pl
# import utils.data_prep as dp
# import utils.setup as su
# from utils.project import ModelSpec
# from utils.sklearn_extensions import classifier_quality

# import utils.transformations as tr


# %%
def to_dict(x, series_as_list=True):
    """
    crude generality for transforming to dictionary;
    retains lists so it works almost like transforming to json;
    Examples
    --------
    obj = {'a': [{"x": 2, "y":["a", {11: 'aa', 22:"bb"}, pd.DataFrame({'a':[1, 2], 'b':[33, 22]})]}, 2],
           "b": {'p': (3, 4), "g": 2},
           "c": pd.DataFrame({'a':[1, 2], 'b':[33, 22]}),
           "d": pd.Series([1, 2, 3, 4, 5])
           }
    obj
    to_dict(obj)
    to_dict(obj, False)
    obj = {'a': [2, 3, 2], "b": {'p': (3, 4), "g": 2}, "c": None, "d": pd.Timestamp('2022-02-22 22:22')}
    to_dict(obj)
    """
    if isinstance(x, pathlib.PosixPath):
        res = str(x)
    elif isinstance(x, str):
        res = x
    elif isinstance(x, pd.Series) and series_as_list:
        res = x.to_list()
    elif isinstance(x, Iterable):
        try:
            res = {k: to_dict(v, series_as_list) for k, v in x.items()}
        except Exception:
            res = [to_dict(v, series_as_list) for v in x]
    else:
        try:
            res = float(x)
        except Exception:
            res = str(x)

    return res


# %%
def plots_compare(
        events: pd.DataFrame,
        wam: WindowsAtMoments,
        variables: Iterable[str],   # list/tuple/set of variables names
        most_common: int = 33, ) -> None:
    """"""
    for v in variables:
        print('---------------------------------------------------------------------')
        print(f"{v} [before]")
        pl.plot_variable(
            v, data=events.loc[wam.union('window'), :], horizontal=False,
            most_common=most_common, sort_levels=True, title_suffix=" [before]")
        print('-----------------------')
        print(f"{v} [between]")
        pl.plot_variable(
            v, data=events.loc[wam.complement('all'), :], horizontal=False,
            most_common=most_common, sort_levels=True, title_suffix=" [between]")


# %% comparison and visualisation of probabilities in both cases
def normalise_within(df, level=0, axis=1) -> pd.Series:
    """
    For DataFrame `df` with MultiIndex on given `axis` (default 1 == columns)
    returns pd.Series with variables along the given MultiIndex `level`
    (levels of a `level`... :P -- YES! it's Pandas naming inconsistency,
     see https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
    being normalised within, s.t. values for levels (categories) of each variable
    are probabilities within this variable.
    First the sum along the second axis is performed (obviously),
    so that we work on summary pd.Series with MultiIndex.
    """
    df_sum = df.sum(axis=not axis)
    df_trans = df_sum.groupby(level=level).transform(lambda x: x / sum(x))
    return df_trans


def collate_probs(columns=None, **kwargs):
    """ e.g.
    collate_probs(columns_common, before=frame_before, other=frame_other)
    """
    if columns is not None:
        kw = {k: normalise_within(v[columns]) for k, v in kwargs.items()}
    else:
        kw = {k: normalise_within(v) for k, v in kwargs.items()}
    coll = pd.concat(kw, axis=1).fillna(0)
    coll['diff'] = coll.iloc[:, 0] - coll.iloc[:, 1]
    return coll


def plot_bars(probs_df):
    """
    Plots all columns of `probs_df` (each value is plotted as bar)
    which is indexed with MultiIndex where
    level 0 is "variable" and level 1 are categories of each variable.
    Each "variable" from level 0 is plotted separately (separate figure)
    which has as many subplots as there are columns of `probs_df`
    so that one may compare values for each column within each "variable";
    Meant for comparing probabilities for normalized summary of
    categorical data (factors);
    However, values may not be porbabilities.
    """
    for v in probs_df.index.levels[0]:
        fig, axs = plt.subplots(3, 1, figsize=(8, 13))
        for k, c in enumerate(probs_df.columns):
            var = probs_df.loc[v][c]
            axs[k].bar(var.index, var)
            axs[k].grid(color='darkgray', alpha=.3)
            axs[k].tick_params(axis='x', labelrotation=75.)
            axs[k].set_title(c, color='grey')
        fig.suptitle(v, fontsize=15)
        plt.tight_layout()


# %%
# %%  XGBOOST
# %%

# !!! generalise it for any type of model
def xgboost_repeat(
        INFO, xx, yy, xx_0=None, yy_0=None, inverse=None, k=None,
        model=None,
        # if model is None then it will be via XGBRegressor with following parameters:
        objective='reg:squarederror',
        colsample_bytree=0.8,
        subsample=0.4,
        eta=0.05,
        max_depth=3,
        n_estimators=80,
        seed=None,
        importance_type="gain",
        silent=True,  # ?
        #
        plot=True,
        ID=None,
        write=True,
        *args, **kwargs):
    """"""
    if seed is None:
        seed = np.random.randint(9999)

    INFO0 = INFO.copy()
    if ID is not None:
        INFO0.id = ID
    if k is not None:
        INFO0.id = INFO.id + "_" + str(k).rjust(2, "0")
    INFO0.randstate = seed
    INFO0.update()

    if model is None:

        model0 = xgb.XGBRegressor(
            objective=objective,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            eta=eta,
            max_depth=max_depth,       # tree depth
            n_estimators=n_estimators,    # boosting rounds
            random_state=seed,
            importance_type=importance_type,   # !!!
            silent=silent,  # ?
            verbosity=0, )

    else:

        model0 = deepcopy(model)
        model0.random_state = seed
        model0.silent = silent  # ?
        model0.verbosity = 0

    model0.fit(xx, yy, eval_set=[(xx, yy)], verbose=False)  # , early_stopping_rounds=10)
    INFO0.model_details = model0.__str__()

    importances = pd.Series(model0.feature_importances_, index=xx.columns).sort_values(ascending=False)
    INFO0.importances = importances

    yy_hat = pd.Series(model0.predict(xx))
    yy_0_hat = pd.Series(model0.predict(xx_0)) if xx_0 is not None else None

    INFO0.performance(yy, yy_hat, yy_0, yy_0_hat, inverse)

    INFO0.cv_score(model0, xx, yy, scoring=['explained_variance', 'neg_root_mean_squared_error'])

    INFO0.path['plot_importance'] = {
        imp_type: INFO0.folder / f"{INFO0.name_id}_importance_{imp_type}.png"
        for imp_type in ["gain", "weight", "cover"]}

    if plot:
        plt.close('all')
        for importance_type in ["weight", "gain", "cover"]:
            fig, ax = plt.subplots(figsize=(14, 11))
            xgb.plot_importance(model0, ax=ax, grid=False, importance_type=importance_type,
                                title=f"{INFO0.name_id} - importance - {importance_type}")
            plt.tight_layout()
            plt.savefig(INFO0.path['plot_importance'][importance_type])

    if write:
        pickle.dump((model0, INFO0), open(INFO0.path['model'], "wb"))

    return model0, INFO0


def xgboost_loop(N, INFO, xx, yy, xx_0=None, yy_0=None, inverse=None, plot=False, write=False, *args, **kwargs):
    """"""
    models_dict = dict()
    for n in range(N):
        print(n)
        model_n, INFO_n = xgboost_repeat(INFO, xx, yy, xx_0, yy_0,
                                         inverse=inverse, k=n, plot=plot, write=write, *args, **kwargs)
        # !!!  DEFINITION OF models_dict :
        models_dict[n] = dict(model=model_n, info=INFO_n)
    return models_dict


def get_importances(models_dict):
    """DO it numeric !!!
    """
    N = len(models_dict)
    res = pd.concat(
        [models_dict[n]['info'].importances.to_frame().reset_index().iloc[:, 0] for n in range(N)],
        axis=1)
    return res


def rank_variables(importances):
    """"""
    rank_df = pd.DataFrame()
    for n, ss in importances.iteritems():
        ss = ss.reset_index()
        ss.index = ss['index']
        ss = ss.level_0
        rank_df = pd.concat([rank_df, ss], axis=1)
        ranking = rank_df.sum(axis=1).sort_values()
    return ranking


# %%
# %%  GAM
# %% ...


# %%
# %%  common
# %%

def print_metrics(models_dict, metric):
    """"""
    for k, item in models_dict.items():
        print(item['info'].metrics[metric])


def print_metrics_raw(models_dict, metric):
    """"""
    for k, item in models_dict.items():
        print(item['info'].metrics_raw[metric])


def metrics_df(models_dict, metric, raw=False):
    """ `key` in `models_dict` may be arbitrary !!!  v.nice
    """

    if raw:
        test_dic = {k: item['info'].metrics_raw[metric].test
                    for k, item in models_dict.items()}
        train_dic = {k: item['info'].metrics_raw[metric].train
                     for k, item in models_dict.items()}

    else:
        test_dic = {k: item['info'].metrics[metric].test
                    for k, item in models_dict.items()}
        train_dic = {k: item['info'].metrics[metric].train
                     for k, item in models_dict.items()}

    metrics = pd.DataFrame({"test": pd.Series(test_dic), "train": pd.Series(train_dic)})

    return metrics

# %%
